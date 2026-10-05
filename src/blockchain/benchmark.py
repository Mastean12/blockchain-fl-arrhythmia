"""Benchmarks for the isolated FL round ledger (synthetic, deterministic FL-shaped records).

No FL run is read or modified. Each round block records 5 synthetic client update
hashes and one aggregation hash, mirroring the 5-client FedAvg setup. Timings are
local wall-clock measurements (time.perf_counter) on one machine, without load
control, reported as medians over repeats.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import pandas as pd

from .ledger import Ledger, canonical_json, hash_payload, sha256_hex
from .tamper import run_tamper_trials


def fixed_clock(start="2026-10-05T00:00:00+00:00", step_seconds=1):
    """Deterministic clock for reproducible ledgers and tests."""
    current = [datetime.fromisoformat(start)]

    def clock():
        value = current[0].isoformat(timespec="microseconds")
        current[0] += timedelta(seconds=step_seconds)
        return value
    return clock


def synthetic_hash(*parts):
    return sha256_hex("|".join(map(str, parts)).encode("utf-8"))


def build_ledger(rounds, clients, seed, clock=None, ledger_id="fl_round_ledger_v1", limits=None):
    """Genesis + `rounds` round blocks with deterministic synthetic hashes."""
    ledger = Ledger(ledger_id, f"synthetic_fedavg_s{seed}", synthetic_hash(seed, "initial_model"),
                    clock=clock or fixed_clock(), limits=limits)
    counts = [16406, 18698, 16563, 13946, 11937]
    for r in range(1, rounds + 1):
        participants = [{"client_id": f"client_{k:02d}", "num_samples": counts[k % len(counts)],
                         "update_hash": synthetic_hash(seed, r, k, "update")} for k in range(clients)]
        ledger.append_round(r, participants, synthetic_hash(seed, r, "aggregate"),
                            metadata={"aggregation": "FedAvg", "participants": clients})
    return ledger


def _median_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def run_benchmarks(config_path="configs/blockchain/ledger_v1.json"):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    bench, limits = config["benchmark"], config["limits"]
    root = Path(config["output_root"])
    for sub in ("tables", "metrics", "figures", "ledgers"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    seed, clients, repeats = bench["seed"], bench["clients_per_round"], bench["repeats"]

    # Hashing overhead for payload sizes seen in Days 10 and 14 (synthetic random bytes of that size).
    rng = np.random.default_rng(seed)
    hashing = []
    for label, size in bench["payload_bytes"].items():
        payload = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
        seconds, _ = _median_time(lambda: [hash_payload(payload) for _ in range(bench["hash_repeats"])], repeats)
        per_hash = seconds / bench["hash_repeats"]
        hashing.append({"payload": label, "payload_bytes": size, "seconds_per_hash": per_hash,
                        "throughput_MB_per_s": size / per_hash / 1e6,
                        "seconds_per_round_5_clients_plus_aggregate": per_hash * (clients + 1)})

    # Block creation, validation, persistence vs chain length.
    scaling = []
    for length in bench["chain_lengths"]:
        rounds = length - 1
        create_s, ledger = _median_time(lambda: build_ledger(rounds, clients, seed, limits=limits), repeats)
        validate_s, (ok, problems) = _median_time(ledger.validate, repeats)
        if not ok:
            raise RuntimeError(f"Benchmark ledger invalid: {problems[:3]}")
        anchor_s, _ = _median_time(lambda: ledger.validate(anchor=ledger.anchor()), repeats)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            save_s, size = _median_time(lambda: ledger.save(path), repeats)
            load_s, loaded = _median_time(lambda: Ledger.load(path), repeats)
            if [b.to_dict() for b in loaded.blocks] != [b.to_dict() for b in ledger.blocks]:
                raise RuntimeError("save/load round trip changed the chain")
        block_bytes = [len(canonical_json(b.to_dict())) for b in ledger.blocks[1:]]
        scaling.append({"chain_length_blocks": length, "round_blocks": rounds, "clients_per_block": clients,
                        "create_seconds_total": create_s, "create_ms_per_block": 1e3 * create_s / length,
                        "validate_seconds_total": validate_s, "validate_ms_per_block": 1e3 * validate_s / length,
                        "validate_with_anchor_seconds": anchor_s, "save_seconds": save_s, "load_seconds": load_s,
                        "file_bytes": size, "file_bytes_per_block": size / length,
                        "round_block_canonical_bytes_mean": float(np.mean(block_bytes))})

    # Hashing share of block creation: SHA-256 over the canonical block bytes alone.
    sample = build_ledger(1000, clients, seed, limits=limits)
    encoded = [canonical_json(b.content()) for b in sample.blocks]
    hash_only_s, _ = _median_time(lambda: [sha256_hex(e) for e in encoded], repeats)
    encode_s, _ = _median_time(lambda: [canonical_json(b.content()) for b in sample.blocks], repeats)

    # Tamper detection.
    tamper_ledger = build_ledger(bench["tamper_chain_length"] - 1, clients, seed, limits=limits)
    tamper = run_tamper_trials(tamper_ledger, bench["tamper_trials_per_scenario"], seed)

    # Deterministic example ledger (genesis + 20 rounds) and its anchor.
    example = build_ledger(20, clients, seed, limits=limits)
    example_size = example.save(root / "ledgers" / "example_20_rounds.jsonl")
    (root / "ledgers" / "example_20_rounds_anchor.json").write_text(
        json.dumps(example.anchor(), indent=2) + "\n", encoding="utf-8")

    hashing, scaling, tamper = pd.DataFrame(hashing), pd.DataFrame(scaling), pd.DataFrame(tamper)
    hashing.to_csv(root / "tables" / "hashing_overhead.csv", index=False)
    scaling.to_csv(root / "tables" / "ledger_scaling.csv", index=False)
    tamper.to_csv(root / "tables" / "tamper_detection.csv", index=False)
    summary = {
        "config": config, "seed": seed,
        "hashing_overhead": hashing.to_dict("records"),
        "block_hashing_share": {"blocks": len(encoded), "sha256_seconds": hash_only_s,
                                "canonical_encoding_seconds": encode_s,
                                "sha256_us_per_block": 1e6 * hash_only_s / len(encoded),
                                "encoding_us_per_block": 1e6 * encode_s / len(encoded)},
        "scaling": scaling.to_dict("records"), "tamper_detection": tamper.to_dict("records"),
        "example_ledger": {"path": str(root / "ledgers" / "example_20_rounds.jsonl").replace("\\", "/"),
                           "bytes": example_size, **example.anchor()},
        "timing_note": "Median wall-clock over repeats on one machine; load not controlled.",
    }
    (root / "metrics" / "blockchain_benchmark.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                encoding="utf-8")
    from src.federated import plots
    plots.plot_ledger_scaling(scaling, root / "figures" / "ledger_scaling.png")
    return summary
