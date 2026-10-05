"""Day 16 report: FedAvg with vs without blockchain recording (real runs, seeds 42/123/2024).

Reads saved outputs only. The "without" arm is the Day 11 natural FedAvg run with the
same seed (same code path, partition, initialization and training order).
"""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from src.federated.diagnostics import write_diagnostic_reports
from src.federated.robustness import load_robustness_config, planned_runs, write_reports
from src.federated import plots
from .fl_recorder import state_hash, verify_against_off_chain
from .ledger import Ledger
from .tamper import run_tamper_trials


def _dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _offchain_tamper_trials(ledger, off_chain_dir, trials, seed):
    """Perturb one value of one archived client or global state; the recomputed hash must differ."""
    rng = random.Random(seed)
    archives = {b.round: torch.load(Path(off_chain_dir) / f"round_{b.round:03d}.pt", map_location="cpu",
                                    weights_only=False) for b in ledger.blocks}
    detected = 0
    for _ in range(trials):
        block = rng.choice(ledger.blocks[1:])
        archive = archives[block.round]
        if rng.random() < 0.5:
            record = rng.choice(block.participants)
            state, expected = copy.deepcopy(archive["client_states"][record["client_id"]]), record["update_hash"]
        else:
            state, expected = copy.deepcopy(archive["global_state"]), block.aggregation_hash
        key = rng.choice([k for k, v in state.items() if v.is_floating_point()])
        flat = state[key].view(-1)
        flat[rng.randrange(flat.numel())] += float(np.float32(1e-6))
        detected += state_hash(state) != expected
    return {"scenario": "perturb_one_off_chain_value", "trials": trials, "detected": detected,
            "rate": detected / trials}


def write_integration_reports(config_path="configs/blockchain/fedavg_blockchain_v1.json", tamper_trials=200):
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    comparator = config["comparator"]
    write_reports(config_path)
    write_diagnostic_reports(config_path)
    base_root = Path(comparator["runs_table"]).parents[1]
    rows, verification, tamper_rows = [], [], []
    for run_id, _, _, seed in planned_runs(config):
        base_id = comparator["run_id_template"].format(seed=seed)
        metrics, base_metrics = root / "metrics" / "federated", base_root / "metrics" / "federated"
        summary = json.loads((metrics / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        base_summary = json.loads((base_metrics / f"{base_id}_run_summary.json").read_text(encoding="utf-8"))
        bc = json.loads((metrics / f"{run_id}_blockchain_summary.json").read_text(encoding="utf-8"))
        timings = pd.read_csv(metrics / f"{run_id}_blockchain_round_timings.csv")
        rounds_only = timings[timings["round"] > 0]

        # Equivalence with the paired run without blockchain.
        hist = pd.read_csv(metrics / f"{run_id}_round_history.csv").drop(columns="elapsed_seconds")
        base_hist = pd.read_csv(base_metrics / f"{base_id}_round_history.csv").drop(columns="elapsed_seconds")
        test = json.loads((metrics / f"{run_id}_test_metrics.json").read_text(encoding="utf-8"))
        base_test = json.loads((base_metrics / f"{base_id}_test_metrics.json").read_text(encoding="utf-8"))
        model = torch.load(root / "models/federated" / f"{run_id}_global.pt", weights_only=False)["model_state_dict"]
        base_model_path = base_root / "models/federated" / f"{base_id}_global.pt"
        base_pred_path = base_metrics / f"{base_id}_test_predictions.npz"
        model_equal = preds_equal = None
        if base_model_path.exists():
            base_model = torch.load(base_model_path, weights_only=False)["model_state_dict"]
            model_equal = all(torch.equal(model[k], base_model[k]) for k in model)
        if base_pred_path.exists():
            pa, pb = np.load(metrics / f"{run_id}_test_predictions.npz"), np.load(base_pred_path)
            preds_equal = all(np.array_equal(pa[k], pb[k]) for k in pb.files)

        # Verification of every on-chain hash against the off-chain archive.
        ledger = Ledger.load(root / "ledgers" / f"{run_id}.jsonl")
        anchor = json.loads((root / "anchors" / f"{run_id}_anchor.json").read_text(encoding="utf-8"))
        ok_chain, _ = ledger.validate(anchor=anchor)
        checks = verify_against_off_chain(ledger, root / "offchain" / run_id)
        verified = sum(c["matches"] for c in checks)
        verification.append({"seed": seed, "run_id": run_id, "chain_valid_with_anchor": ok_chain,
                             "hashes_checked": len(checks), "hashes_matching": verified,
                             "client_hashes_checked": sum(c["item"] not in ("global_state", "archive") for c in checks),
                             "global_hashes_checked": sum(c["item"] == "global_state" for c in checks),
                             "selected_round": bc["selected_round"],
                             "selected_checkpoint_hash_matches_ledger": bc["selected_checkpoint_hash_matches_ledger"]})

        # Tamper detection on the real ledger and the off-chain archive.
        for row in run_tamper_trials(ledger, tamper_trials, seed):
            tamper_rows.append({"seed": seed, **row})
        off = _offchain_tamper_trials(ledger, root / "offchain" / run_id, tamper_trials, seed)
        tamper_rows.append({"seed": seed, "scenario": off["scenario"], "trials": off["trials"],
                            "detected_internal": off["detected"], "detected_with_anchor": off["detected"],
                            "internal_rate": off["rate"], "anchored_rate": off["rate"]})

        bc_seconds = float(timings["hash_seconds"].sum() + timings["block_seconds"].sum())
        rows.append({
            "seed": seed, "run_id": run_id, "paired_run_without_blockchain": base_id,
            "selected_round": summary["selected_round_by_validation_loss"],
            "selected_round_without": base_summary["selected_round_by_validation_loss"],
            "test_accuracy": test["metrics"]["accuracy"], "test_accuracy_without": base_test["metrics"]["accuracy"],
            "test_macro_f1": test["metrics"]["macro_f1"], "test_macro_f1_without": base_test["metrics"]["macro_f1"],
            "test_weighted_f1": test["metrics"]["weighted_f1"],
            "test_weighted_f1_without": base_test["metrics"]["weighted_f1"],
            "test_metrics_identical": test["metrics"] == base_test["metrics"],
            "round_history_identical": hist.equals(base_hist),
            "selected_model_bit_identical": model_equal, "test_predictions_identical": preds_equal,
            "hash_ms_per_round": 1e3 * float(rounds_only["hash_seconds"].mean()),
            "hash_ms_per_state": 1e3 * float((rounds_only["hash_seconds"] / rounds_only["hashed_states"]).mean()),
            "block_creation_ms_per_round": 1e3 * float(rounds_only["block_seconds"].mean()),
            "off_chain_save_ms_per_round": 1e3 * float(rounds_only["off_chain_save_seconds"].mean()),
            "blockchain_seconds_total_hash_plus_blocks": bc_seconds,
            "validate_ms_full_chain": 1e3 * bc["validate_seconds"],
            "blocks": bc["blocks"], "ledger_bytes": bc["ledger_bytes"],
            "ledger_bytes_per_round_block": float(rounds_only["block_bytes"].mean()),
            "off_chain_archive_bytes": _dir_bytes(root / "offchain" / run_id),
            "training_seconds_with": summary["training_elapsed_seconds"],
            "training_seconds_without": base_summary["training_elapsed_seconds"],
            "blockchain_share_of_training_time": bc_seconds / summary["training_elapsed_seconds"],
        })
    per_seed, verification, tamper = pd.DataFrame(rows), pd.DataFrame(verification), pd.DataFrame(tamper_rows)
    tables = root / "tables"
    per_seed.to_csv(tables / "blockchain_integration_per_seed.csv", index=False)
    verification.to_csv(tables / "blockchain_integration_verification.csv", index=False)
    tamper.to_csv(tables / "blockchain_integration_tamper_detection.csv", index=False)
    plots.plot_blockchain_overhead(per_seed, root / "figures" / "blockchain_integration_overhead.png")
    return per_seed, verification, tamper
