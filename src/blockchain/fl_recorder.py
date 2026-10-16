"""Record federated rounds on the hash-chained ledger, with model states kept off-chain.

On-chain per round (via ``Ledger.append_round``): round number, participating
client IDs, client sample counts, SHA-256 of each client's uploaded model state,
SHA-256 of the aggregated global state, timestamp, and previous block hash.

Off-chain: the exact client states and the global state of every round are saved
under an archive directory for plain FedAvg. The HE+DP v2 lifecycle path instead
commits fixed public weights and update hashes without archiving client states or
publishing sample counts.

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
                 metadata=None, expected_roster=None, expected_weights=None):
        self.off_chain_dir = Path(off_chain_dir)
        self.off_chain_dir.mkdir(parents=True, exist_ok=True)
        self.timings = []
        roster = list(expected_roster) if expected_roster is not None else None
        self.expected_roster = sorted(roster) if roster is not None else None
        if expected_weights is not None and (roster is None or len(roster) != len(expected_weights)):
            raise ValueError("expected_weights must match the configured fixed roster")
        self.expected_weights = (dict(zip(roster, map(float, expected_weights)))
                                 if expected_weights is not None else None)
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

    def prepare_round(self, round_number, client_ids, sample_counts, client_states, previous_global_state,
                      metadata=None):
        """Commit update hashes, then verify roster and payloads before aggregation."""
        started = time.perf_counter()
        if not (len(client_ids) == len(sample_counts) == len(client_states)):
            raise ValueError("round submissions have inconsistent lengths")
        if self.expected_roster is not None and sorted(client_ids) != self.expected_roster:
            raise ValueError("round submission is missing, duplicated, or outside the fixed participant roster")
        valid, problems = self.ledger.validate()
        if not valid:
            raise RuntimeError(f"ledger is invalid before commitment: {problems[:3]}")
        if round_number != self.ledger.blocks[-1].round + 1:
            raise RuntimeError(f"round {round_number} is stale or out of sequence after "
                               f"round {self.ledger.blocks[-1].round}")
        if self.ledger.blocks[-1].aggregation_hash != state_hash(previous_global_state):
            raise RuntimeError("submitted updates are based on a stale or uncommitted global model")
        participants = [{"client_id": cid, "num_samples": int(n), "update_hash": state_hash(state)}
                       for cid, n, state in zip(client_ids, sample_counts, client_states)]
        hash_finished = time.perf_counter()
        block = self.ledger.append_round_prepare(round_number, participants, state_hash(previous_global_state),
                                                  metadata=metadata)
        append_finished = time.perf_counter()
        if not self.verify_prepared_round(round_number, client_ids, sample_counts, client_states):
            raise RuntimeError("prepared update commitments failed verification; aggregation is blocked")
        finished = time.perf_counter()
        self.timings.append({"round": round_number, "phase": "prepare", "hash_seconds": hash_finished - started,
                             "block_seconds": append_finished - hash_finished,
                             "verify_seconds": finished - append_finished,
                             "block_bytes": len(json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":")))})
        return block

    def verify_prepared_round(self, round_number, client_ids, sample_counts, client_states):
        ok, _ = self.ledger.validate(allow_pending_prepared=True)
        if not ok or not (len(client_ids) == len(sample_counts) == len(client_states)):
            return False
        if not self.ledger.blocks:
            return False
        prepared = self.ledger.blocks[-1]
        if (prepared.round != round_number or prepared.metadata.get("phase") != "prepared"
                or len(self.ledger.blocks) < 2
                or prepared.round != self.ledger.blocks[-2].round + 1
                or prepared.aggregation_hash != self.ledger.blocks[-2].aggregation_hash):
            return False
        if self.expected_roster is not None and sorted(client_ids) != self.expected_roster:
            return False
        expected = sorted(({"client_id": cid, "num_samples": int(n), "update_hash": state_hash(state)}
                           for cid, n, state in zip(client_ids, sample_counts, client_states)),
                          key=lambda row: row["client_id"])
        return expected == prepared.participants

    def aggregate_if_prepared(self, round_number, client_ids, sample_counts, client_states, aggregate):
        """Run the aggregation callback only after ledger integrity and payload checks pass."""
        if not self.verify_prepared_round(round_number, client_ids, sample_counts, client_states):
            raise RuntimeError("ledger commitment verification rejected aggregation")
        return aggregate()

    def complete_round(self, round_number, client_ids, sample_counts, client_states, global_state, metadata=None):
        started = time.perf_counter()
        if not self.verify_prepared_round(round_number, client_ids, sample_counts, client_states):
            raise RuntimeError("prepared commitments no longer verify; model broadcast is blocked")
        participants = [{"client_id": cid, "num_samples": int(n), "update_hash": state_hash(state)}
                       for cid, n, state in zip(client_ids, sample_counts, client_states)]
        append_started = time.perf_counter()
        block = self.ledger.append_round_complete(round_number, participants, state_hash(global_state), metadata)
        ok, problems = self.ledger.validate()
        if not ok or block.aggregation_hash != state_hash(global_state):
            raise RuntimeError(f"global-model commitment failed before broadcast: {problems[:3]}")
        append_finished = time.perf_counter()
        torch.save({"round": round_number, "global_state": global_state,
                    "client_states": dict(zip(client_ids, client_states))},
                   self.off_chain_dir / f"round_{round_number:03d}.pt")
        finished = time.perf_counter()
        self.timings.append({"round": round_number, "phase": "complete",
                             "verify_and_hash_seconds": append_started - started,
                             "block_seconds": append_finished - append_started,
                             "off_chain_save_seconds": finished - append_finished,
                             "block_bytes": len(json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":")))})
        return block

    def _fixed_weight_participants(self, client_ids, client_states, weights):
        if len(client_ids) != len(client_states) or len(client_ids) != len(weights):
            raise ValueError("fixed-roster submissions have inconsistent lengths")
        if self.expected_roster is None or sorted(client_ids) != self.expected_roster:
            raise ValueError("round submission is missing, duplicated, or outside the fixed participant roster")
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("round submission is missing, duplicated, or outside the fixed participant roster")
        if self.expected_weights is None:
            raise ValueError("fixed public aggregation weights were not configured")
        participants = []
        for client_id, state, weight in zip(client_ids, client_states, weights):
            expected_weight = self.expected_weights.get(client_id)
            if expected_weight is None or float(weight) != expected_weight:
                raise ValueError("submitted aggregation weights differ from the configured public weights")
            participants.append({"client_id": client_id, "update_hash": state_hash(state),
                                 "weight": float(weight)})
        return sorted(participants, key=lambda record: record["client_id"])

    def prepare_private_round(self, round_number, client_ids, client_states, previous_global_state, weights,
                              metadata=None):
        """Commit fixed-weight HE+DP updates without counts or an off-chain update archive."""
        started = time.perf_counter()
        valid, problems = self.ledger.validate()
        if not valid:
            raise RuntimeError(f"ledger is invalid before commitment: {problems[:3]}")
        tip = self.ledger.blocks[-1]
        if tip.metadata.get("phase") == "prepared":
            raise RuntimeError("previous round has an uncompleted commitment; submissions are stale")
        if round_number != tip.round + 1:
            raise RuntimeError(f"round {round_number} is stale or out of sequence after round {tip.round}")
        previous_hash = state_hash(previous_global_state)
        if tip.aggregation_hash != previous_hash:
            raise RuntimeError("submitted updates are based on a stale or uncommitted global model")
        participants = self._fixed_weight_participants(client_ids, client_states, weights)
        hashed = time.perf_counter()
        block = self.ledger.append_round_prepare(
            round_number, participants, previous_hash,
            metadata={**dict(metadata or {}), "commitment_format": "fixed_public_weight"})
        appended = time.perf_counter()
        if not self.verify_private_prepared_round(round_number, client_ids, client_states, weights,
                                                  previous_global_state):
            raise RuntimeError("prepared update commitments failed verification; aggregation is blocked")
        verified = time.perf_counter()
        self.timings.append({
            "round": round_number, "phase": "prepare", "hash_seconds": hashed - started,
            "block_seconds": appended - hashed, "verify_seconds": verified - appended,
            "block_bytes": len(json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":"))),
        })
        return block

    def verify_private_prepared_round(self, round_number, client_ids, client_states, weights,
                                      previous_global_state):
        valid, _ = self.ledger.validate(allow_pending_prepared=True)
        if not valid:
            return False
        tip = self.ledger.blocks[-1]
        if tip.round != round_number or tip.metadata.get("phase") != "prepared":
            return False
        if len(self.ledger.blocks) < 2 or self.ledger.blocks[-2].round + 1 != round_number:
            return False
        if tip.aggregation_hash != state_hash(previous_global_state):
            return False
        try:
            expected = self._fixed_weight_participants(client_ids, client_states, weights)
        except (TypeError, ValueError):
            return False
        return (tip.metadata.get("commitment_format") == "fixed_public_weight"
                and expected == tip.participants)

    def aggregate_private_if_prepared(self, round_number, client_ids, client_states, weights,
                                      previous_global_state, aggregate):
        """Run HE+DP aggregation only after ledger and fixed-weight payload verification."""
        if not self.verify_private_prepared_round(round_number, client_ids, client_states, weights,
                                                  previous_global_state):
            raise RuntimeError("ledger commitment verification rejected aggregation")
        return aggregate()

    def complete_private_round(self, round_number, client_ids, client_states, weights, previous_global_state,
                               global_state, metadata=None):
        """Commit the noised global model and archive it without client update tensors."""
        started = time.perf_counter()
        if not self.verify_private_prepared_round(round_number, client_ids, client_states, weights,
                                                  previous_global_state):
            raise RuntimeError("prepared commitments no longer verify; model broadcast is blocked")
        participants = self._fixed_weight_participants(client_ids, client_states, weights)
        append_started = time.perf_counter()
        block = self.ledger.append_round_complete(
            round_number, participants, state_hash(global_state),
            metadata={**dict(metadata or {}), "commitment_format": "fixed_public_weight"})
        valid, problems = self.ledger.validate()
        if not valid or block.aggregation_hash != state_hash(global_state):
            raise RuntimeError(f"global-model commitment failed before broadcast: {problems[:3]}")
        append_finished = time.perf_counter()
        torch.save({"round": round_number, "global_state": global_state},
                   self.off_chain_dir / f"round_{round_number:03d}.pt")
        finished = time.perf_counter()
        self.timings.append({
            "round": round_number, "phase": "complete", "verify_and_hash_seconds": append_started - started,
            "block_seconds": append_finished - append_started, "off_chain_save_seconds": finished - append_finished,
            "block_bytes": len(json.dumps(block.to_dict(), sort_keys=True, separators=(",", ":"))),
        })
        return block

    def finalize(self, ledger_path, anchor_path):
        t0 = time.perf_counter()
        ok, problems = self.ledger.validate()
        validate_seconds = time.perf_counter() - t0
        if not ok:
            raise RuntimeError(f"Ledger invalid at finalize: {problems[:3]}")
        anchor_path = Path(anchor_path)
        if anchor_path.exists():
            raise FileExistsError(f"refusing to overwrite existing ledger anchor: {anchor_path}")
        size = self.ledger.save(ledger_path)
        anchor = self.ledger.anchor()
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        with anchor_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(anchor, indent=2) + "\n")
        return {"validate_seconds": validate_seconds, "ledger_bytes": size, "anchor": anchor,
                "blocks": len(self.ledger.blocks)}


def verify_against_off_chain(ledger, off_chain_dir):
    """Recompute every on-chain hash from the off-chain archive. Returns per-item rows."""
    rows = []
    for block in ledger.blocks:
        archived_round = block.round - 1 if block.metadata.get("phase") == "prepared" else block.round
        path = Path(off_chain_dir) / f"round_{archived_round:03d}.pt"
        if not path.exists():
            rows.append({"round": block.round, "item": "archive", "matches": False})
            continue
        archive = torch.load(path, map_location="cpu", weights_only=False)
        global_state = archive["global_state"]
        rows.append({"round": block.round, "item": "global_state",
                     "matches": state_hash(global_state) == block.aggregation_hash})
        if block.metadata.get("phase") == "prepared":
            update_archive_path = Path(off_chain_dir) / f"round_{block.round:03d}.pt"
            update_archive = (torch.load(update_archive_path, map_location="cpu", weights_only=False)
                              if update_archive_path.exists() else {})
            states = update_archive.get("client_states", {})
        else:
            states = archive.get("client_states", {})
        for record in block.participants:
            if block.metadata.get("commitment_format") == "fixed_public_weight":
                rows.append({"round": block.round, "item": record["client_id"], "matches": None,
                              "verification": "update_tensor_not_archived_by_policy"})
                continue
            state = states.get(record["client_id"])
            rows.append({"round": block.round, "item": record["client_id"],
                         "matches": state is not None and state_hash(state) == record["update_hash"]})
    return rows
