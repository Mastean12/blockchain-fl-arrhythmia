"""Record real FedAvg rounds on the hash-chained ledger, with model states kept off-chain.

On-chain per round (via ``Ledger.append_round``): round number, participating
client IDs, client sample counts, SHA-256 of each client's uploaded model state,
SHA-256 of the aggregated global state, timestamp, and previous block hash.

Off-chain: the exact client states and the global state of every round are saved
under an archive directory, so every on-chain hash can be re-verified against the
real artifact. Nothing but hashes, identifiers, counts, and scalar metadata goes on-chain.

The recorder only *reads* tensors (no RNG use, no in-place changes), so recording
cannot alter training or aggregation; integration tests check this bit for bit.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from .ledger import Ledger, utc_timestamp


def canonical_state_bytes(state):
    """Deterministic serialization of a state dict: per entry, key, dtype, shape, then raw little-endian bytes."""
    chunks = []
    for key, tensor in state.items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        header = json.dumps({"key": key, "dtype": array.dtype.str, "shape": list(array.shape)},
                            sort_keys=True, separators=(",", ":")).encode("ascii")
        chunks.append(len(header).to_bytes(4, "little") + header)
        chunks.append(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
    return b"".join(chunks)


def state_hash(state):
    return hashlib.sha256(canonical_state_bytes(state)).hexdigest()


class FLLedgerRecorder:
    def __init__(self, ledger_id, experiment_id, initial_state, off_chain_dir, clock=utc_timestamp, limits=None,
                 metadata=None):
        self.off_chain_dir = Path(off_chain_dir)
        self.off_chain_dir.mkdir(parents=True, exist_ok=True)
        self.timings = []
        t0 = time.perf_counter()
        initial_hash = state_hash(initial_state)
        t1 = time.perf_counter()
        self.ledger = Ledger(ledger_id, experiment_id, initial_hash, clock=clock, limits=limits,
                             genesis_metadata=metadata)
        t2 = time.perf_counter()
        torch.save({"round": 0, "global_state": initial_state}, self.off_chain_dir / "round_000.pt")
        t3 = time.perf_counter()
        genesis = self.ledger.blocks[0]
        self.timings.append({"round": 0, "hash_seconds": t1 - t0, "block_seconds": t2 - t1,
                             "off_chain_save_seconds": t3 - t2, "hashed_states": 1,
                             "block_bytes": len(json.dumps(genesis.to_dict(), sort_keys=True, separators=(",", ":")))})

    def record_round(self, round_number, client_ids, sample_counts, client_states, global_state, metadata=None):
        t0 = time.perf_counter()
        client_hashes = [state_hash(s) for s in client_states]
        global_hash = state_hash(global_state)
        t1 = time.perf_counter()
        participants = [{"client_id": cid, "num_samples": int(n), "update_hash": h}
                        for cid, n, h in zip(client_ids, sample_counts, client_hashes)]
        block = self.ledger.append_round(round_number, participants, global_hash, metadata=metadata)
        t2 = time.perf_counter()
        torch.save({"round": round_number, "global_state": global_state,
                    "client_states": dict(zip(client_ids, client_states))},
                   self.off_chain_dir / f"round_{round_number:03d}.pt")
        t3 = time.perf_counter()
        self.timings.append({"round": round_number, "hash_seconds": t1 - t0, "block_seconds": t2 - t1,
                             "off_chain_save_seconds": t3 - t2, "hashed_states": len(client_states) + 1,
                             "block_bytes": len(json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":")))})
        return block

    def finalize(self, ledger_path, anchor_path):
        t0 = time.perf_counter()
        ok, problems = self.ledger.validate()
        validate_seconds = time.perf_counter() - t0
        if not ok:
            raise RuntimeError(f"Ledger invalid at finalize: {problems[:3]}")
        size = self.ledger.save(ledger_path)
        anchor = self.ledger.anchor()
        Path(anchor_path).parent.mkdir(parents=True, exist_ok=True)
        Path(anchor_path).write_text(json.dumps(anchor, indent=2) + "\n", encoding="utf-8")
        return {"validate_seconds": validate_seconds, "ledger_bytes": size, "anchor": anchor,
                "blocks": len(self.ledger.blocks)}


def verify_against_off_chain(ledger, off_chain_dir):
    """Recompute every on-chain hash from the off-chain archive. Returns per-item rows."""
    rows = []
    for block in ledger.blocks:
        path = Path(off_chain_dir) / f"round_{block.round:03d}.pt"
        if not path.exists():
            rows.append({"round": block.round, "item": "archive", "matches": False})
            continue
        archive = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({"round": block.round, "item": "global_state",
                     "matches": state_hash(archive["global_state"]) == block.aggregation_hash})
        states = archive.get("client_states", {})
        for record in block.participants:
            state = states.get(record["client_id"])
            rows.append({"round": block.round, "item": record["client_id"],
                         "matches": state is not None and state_hash(state) == record["update_hash"]})
    return rows
