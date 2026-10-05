"""Single-node, append-only, SHA-256 hash-chained ledger of federated-learning round metadata.

Block structure
---------------
index -> previous_hash -> timestamp -> round -> experiment_id -> participants
(client_id, num_samples, update_hash) -> aggregation_hash -> metadata -> block_hash

``block_hash = SHA-256(canonical_json(block without block_hash))``. Canonical JSON
uses sorted keys, compact separators, ASCII escaping, and rejects NaN/Infinity, so
the same content always hashes to the same value.

Only hashes and small scalar metadata are stored. Schema limits reject large or
nested values, so ECG data, model parameters, and (encrypted) updates cannot be
placed on-chain through this API.

What validation can and cannot detect
-------------------------------------
``validate()`` recomputes every block hash and checks the hash links, indices,
genesis, round order, timestamps, and schema. That detects any edit, deletion,
insertion, or reordering that is not followed by consistently recomputing all
later hashes. A party able to rewrite the whole file can recompute every hash, or
silently drop the newest blocks, and still produce an internally valid chain. Such
rewrites are detectable only against an independently stored *anchor* (the expected
length and tip hash), checked by ``validate(anchor=...)``. There is no consensus,
replication, or digital signature: this is a tamper-evident log, not a distributed,
Byzantine-fault-tolerant blockchain.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

GENESIS_PREVIOUS_HASH = "0" * 64
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_LIMITS = {"max_block_bytes": 16384, "max_participants": 1000, "max_metadata_value_chars": 256,
                  "max_client_id_chars": 64}


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def hash_payload(payload):
    """SHA-256 of raw bytes (e.g. a serialized model update); this digest is what goes on-chain."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("hash_payload expects bytes")
    return sha256_hex(bytes(payload))


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class Block:
    index: int
    previous_hash: str
    timestamp: str
    round: int
    experiment_id: str
    participants: list = field(default_factory=list)
    aggregation_hash: str = ""
    metadata: dict = field(default_factory=dict)
    block_hash: str = ""

    def content(self):
        data = asdict(self)
        data.pop("block_hash")
        return data

    def compute_hash(self):
        return sha256_hex(canonical_json(self.content()))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise ValueError(f"Block fields {sorted(data)} != expected {sorted(expected)}")
        return cls(**data)


class ValidationError(Exception):
    pass


class Ledger:
    def __init__(self, ledger_id, experiment_id, initial_model_hash, clock=utc_timestamp, limits=None,
                 genesis_metadata=None):
        self.ledger_id = ledger_id
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.clock = clock
        self.blocks = []
        genesis = Block(index=0, previous_hash=GENESIS_PREVIOUS_HASH, timestamp=clock(), round=0,
                        experiment_id=experiment_id, participants=[], aggregation_hash=initial_model_hash,
                        metadata={"ledger_id": ledger_id, "record": "genesis: initial global model hash",
                                  **(genesis_metadata or {})})
        self._seal_and_append(genesis)

    # ------------------------------------------------------------------ creation
    def _check_block_schema(self, block):
        limits = self.limits
        if not isinstance(block.index, int) or not isinstance(block.round, int) or isinstance(block.round, bool):
            raise ValidationError("index and round must be integers")
        if not isinstance(block.experiment_id, str) or not block.experiment_id:
            raise ValidationError("experiment_id must be a non-empty string")
        for name, value in (("previous_hash", block.previous_hash), ("aggregation_hash", block.aggregation_hash)):
            if not isinstance(value, str) or not HEX64.match(value):
                raise ValidationError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
        try:
            datetime.fromisoformat(block.timestamp)
        except (TypeError, ValueError) as exc:
            raise ValidationError("timestamp must be ISO-8601") from exc
        if not isinstance(block.participants, list) or len(block.participants) > limits["max_participants"]:
            raise ValidationError("participants must be a list within the size limit")
        ids = []
        for record in block.participants:
            if not isinstance(record, dict) or set(record) != {"client_id", "num_samples", "update_hash"}:
                raise ValidationError("participant records must have exactly client_id, num_samples, update_hash")
            if not isinstance(record["client_id"], str) or not 0 < len(record["client_id"]) <= limits["max_client_id_chars"]:
                raise ValidationError("client_id must be a short non-empty string")
            if not isinstance(record["num_samples"], int) or isinstance(record["num_samples"], bool) or record["num_samples"] < 0:
                raise ValidationError("num_samples must be a non-negative integer")
            if not isinstance(record["update_hash"], str) or not HEX64.match(record["update_hash"]):
                raise ValidationError("update_hash must be a SHA-256 hex digest")
            ids.append(record["client_id"])
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValidationError("participants must be sorted by client_id without duplicates")
        if not isinstance(block.metadata, dict):
            raise ValidationError("metadata must be a dict")
        for key, value in block.metadata.items():
            if not isinstance(key, str):
                raise ValidationError("metadata keys must be strings")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValidationError("metadata floats must be finite")
            if not isinstance(value, (str, int, float, bool)) or value is None:
                raise ValidationError("metadata values must be scalars (no arrays or nested data)")
            if isinstance(value, str) and len(value) > limits["max_metadata_value_chars"]:
                raise ValidationError("metadata string too long")
        if len(canonical_json(block.to_dict())) > limits["max_block_bytes"]:
            raise ValidationError("block exceeds max_block_bytes")

    def _seal_and_append(self, block):
        self._check_block_schema(block)
        block.block_hash = block.compute_hash()
        self.blocks.append(block)
        return block

    def append_round(self, round_number, participants, aggregation_hash, metadata=None):
        """Create, hash, and append the block for one completed FL round."""
        tip = self.blocks[-1]
        if round_number <= tip.round:
            raise ValidationError(f"round {round_number} must exceed the previous round {tip.round}")
        for p in participants:
            if not isinstance(p, dict) or set(p) != {"client_id", "num_samples", "update_hash"}:
                raise ValidationError("participant records must have exactly client_id, num_samples, update_hash")
        records = sorted((dict(p) for p in participants), key=lambda r: str(r["client_id"]))
        block = Block(index=tip.index + 1, previous_hash=tip.block_hash, timestamp=self.clock(), round=int(round_number),
                      experiment_id=tip.experiment_id, participants=records, aggregation_hash=aggregation_hash,
                      metadata=dict(metadata or {}))
        return self._seal_and_append(block)

    # ------------------------------------------------------------------ verification
    def anchor(self):
        """Externally storable commitment to the current chain state."""
        return {"ledger_id": self.ledger_id, "length": len(self.blocks), "tip_hash": self.blocks[-1].block_hash}

    def validate(self, anchor=None):
        """Return (ok, list_of_problems). Checks every block; `anchor` adds length/tip checks."""
        problems = []
        if not self.blocks:
            return False, ["empty chain"]
        for position, block in enumerate(self.blocks):
            try:
                self._check_block_schema(block)
            except ValidationError as exc:
                problems.append(f"block {position}: schema: {exc}")
            if block.index != position:
                problems.append(f"block {position}: index {block.index} != position")
            try:
                hash_matches = block.compute_hash() == block.block_hash
            except (TypeError, ValueError) as exc:  # e.g. non-finite or non-serializable values from a corrupted file
                hash_matches = False
                problems.append(f"block {position}: content cannot be canonically encoded ({exc})")
            if not hash_matches:
                problems.append(f"block {position}: stored hash does not match content")
            if position == 0:
                if block.previous_hash != GENESIS_PREVIOUS_HASH or block.round != 0 or block.participants:
                    problems.append("genesis block malformed")
                continue
            previous = self.blocks[position - 1]
            if block.previous_hash != previous.block_hash:
                problems.append(f"block {position}: previous_hash does not link to block {position - 1}")
            if block.round <= previous.round:
                problems.append(f"block {position}: round {block.round} not after {previous.round}")
            try:
                if datetime.fromisoformat(block.timestamp) < datetime.fromisoformat(previous.timestamp):
                    problems.append(f"block {position}: timestamp earlier than block {position - 1}")
            except (TypeError, ValueError):
                pass  # already reported by the schema check
            if block.experiment_id != previous.experiment_id:
                problems.append(f"block {position}: experiment_id changed")
        if anchor is not None:
            if anchor.get("ledger_id") != self.ledger_id:
                problems.append("anchor: ledger_id mismatch")
            if anchor.get("length") != len(self.blocks):
                problems.append(f"anchor: length {len(self.blocks)} != anchored {anchor.get('length')}")
            if anchor.get("tip_hash") != self.blocks[-1].block_hash:
                problems.append("anchor: tip hash mismatch")
        return not problems, problems

    def verify_update(self, round_number, client_id, payload):
        """True if SHA-256(payload) equals the update hash recorded for that client in that round."""
        for block in self.blocks:
            if block.round == round_number:
                for record in block.participants:
                    if record["client_id"] == client_id:
                        return record["update_hash"] == hash_payload(payload)
                return False
        return False

    def verify_aggregate(self, round_number, payload):
        for block in self.blocks:
            if block.round == round_number:
                return block.aggregation_hash == hash_payload(payload)
        return False

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Atomically write the chain as JSON Lines (one canonical block per line) plus a header line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        header = {"ledger_id": self.ledger_id, "format": "fl-ledger-jsonl-v1", "limits": self.limits}
        with tmp.open("wb") as f:
            f.write(canonical_json(header) + b"\n")
            for block in self.blocks:
                f.write(canonical_json(block.to_dict()) + b"\n")
        os.replace(tmp, path)
        return path.stat().st_size

    @classmethod
    def load(cls, path, clock=utc_timestamp):
        """Load a saved chain without trusting it; call validate() afterwards."""
        lines = Path(path).read_bytes().splitlines()
        if not lines:
            raise ValidationError("empty ledger file")
        header = json.loads(lines[0])
        if header.get("format") != "fl-ledger-jsonl-v1":
            raise ValidationError("unknown ledger format")
        ledger = cls.__new__(cls)
        ledger.ledger_id = header["ledger_id"]
        ledger.limits = {**DEFAULT_LIMITS, **header.get("limits", {})}
        ledger.clock = clock
        ledger.blocks = [Block.from_dict(json.loads(line)) for line in lines[1:]]
        return ledger
