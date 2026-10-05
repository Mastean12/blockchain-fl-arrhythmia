"""Deterministic tests for the isolated FL round ledger (fixed clock, synthetic hashes)."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.blockchain.benchmark import build_ledger, fixed_clock, synthetic_hash
from src.blockchain.ledger import (GENESIS_PREVIOUS_HASH, Ledger, ValidationError, canonical_json, hash_payload,
                                   sha256_hex)
from src.blockchain import tamper

ROOT = Path(__file__).resolve().parents[1]
# Pinned digests: any change to the block format, canonical encoding or hashing breaks these on purpose.
GENESIS_HASH_S42 = "93ea1f76c8dad48480a9fe192ae29c108c627db06685f242d649d4959b5d6167"
BLOCK2_HASH_S42 = "d7f5754d7052d331a1033732d561ecf386ecdc6fb02e2b1ab0e8257b6b0b7a26"


def _participants(r, n=5):
    return [{"client_id": f"client_{k:02d}", "num_samples": 100 + k, "update_hash": synthetic_hash(r, k)}
            for k in range(n)]


class HashingAndDeterminismTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent_and_rejects_nan(self):
        self.assertEqual(canonical_json({"b": 1, "a": [1, 2]}), canonical_json({"a": [1, 2], "b": 1}))
        self.assertEqual(canonical_json({"a": 1}), b'{"a":1}')
        with self.assertRaises(ValueError):
            canonical_json({"x": float("nan")})

    def test_sha256_known_vector(self):
        self.assertEqual(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(hash_payload(b"abc"), sha256_hex(b"abc"))
        with self.assertRaises(TypeError):
            hash_payload("not bytes")

    def test_same_inputs_and_clock_give_identical_pinned_hashes(self):
        a, b = build_ledger(2, 5, 42), build_ledger(2, 5, 42)
        self.assertEqual([x.block_hash for x in a.blocks], [x.block_hash for x in b.blocks])
        self.assertEqual(a.blocks[0].block_hash, GENESIS_HASH_S42)
        self.assertEqual(a.blocks[2].block_hash, BLOCK2_HASH_S42)
        self.assertNotEqual(build_ledger(2, 5, 43).blocks[2].block_hash, BLOCK2_HASH_S42)


class BlockCreationAndValidationTests(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger("test", "exp", synthetic_hash("init"), clock=fixed_clock())

    def test_block_structure_and_links(self):
        block = self.ledger.append_round(1, list(reversed(_participants(1))), synthetic_hash("agg", 1))
        genesis = self.ledger.blocks[0]
        self.assertEqual((genesis.index, genesis.previous_hash, genesis.round), (0, GENESIS_PREVIOUS_HASH, 0))
        self.assertEqual((block.index, block.round, block.previous_hash), (1, 1, genesis.block_hash))
        self.assertEqual([p["client_id"] for p in block.participants], [f"client_{k:02d}" for k in range(5)])
        self.assertEqual(block.block_hash, block.compute_hash())
        self.assertEqual(self.ledger.validate(), (True, []))
        self.assertTrue(self.ledger.validate(anchor=self.ledger.anchor())[0])

    def test_rounds_must_increase(self):
        self.ledger.append_round(3, _participants(3), synthetic_hash("agg", 3))
        with self.assertRaises(ValidationError):
            self.ledger.append_round(3, _participants(3), synthetic_hash("agg", 3))

    def test_schema_keeps_data_and_parameters_off_chain(self):
        bad_cases = [
            dict(participants=_participants(1), aggregation_hash="not-a-hash"),
            dict(participants=[{**_participants(1)[0], "weights": [0.1, 0.2]}], aggregation_hash=synthetic_hash(1)),
            dict(participants=_participants(1), aggregation_hash=synthetic_hash(1), metadata={"weights": [0.1] * 10}),
            dict(participants=_participants(1), aggregation_hash=synthetic_hash(1), metadata={"blob": "x" * 300}),
            dict(participants=_participants(1) + _participants(1)[:1], aggregation_hash=synthetic_hash(1)),
            dict(participants=[{**_participants(1)[0], "update_hash": "ABC"}], aggregation_hash=synthetic_hash(1)),
        ]
        for case in bad_cases:
            with self.assertRaises(ValidationError, msg=str(case)[:80]):
                self.ledger.append_round(1, **case)
        small = Ledger("t", "e", synthetic_hash("i"), clock=fixed_clock(), limits={"max_block_bytes": 600})
        with self.assertRaises(ValidationError):
            small.append_round(1, _participants(1, n=10), synthetic_hash("agg"))
        self.assertEqual(len(self.ledger.blocks), 1)  # nothing invalid was appended

    def test_update_and_aggregate_hash_verification(self):
        update, aggregate = b"client-02 serialized update", b"aggregated model bytes"
        participants = _participants(1)
        participants[2]["update_hash"] = hash_payload(update)
        self.ledger.append_round(1, participants, hash_payload(aggregate))
        self.assertTrue(self.ledger.verify_update(1, "client_02", update))
        self.assertFalse(self.ledger.verify_update(1, "client_02", update + b"x"))
        self.assertFalse(self.ledger.verify_update(1, "client_99", update))
        self.assertFalse(self.ledger.verify_update(7, "client_02", update))
        self.assertTrue(self.ledger.verify_aggregate(1, aggregate))
        self.assertFalse(self.ledger.verify_aggregate(1, b"other"))


class TamperDetectionTests(unittest.TestCase):
    INTERNALLY_UNDETECTABLE = {"full_rewrite_with_recomputed_hashes", "truncate_newest_blocks"}
    TIP_DEPENDENT = {"edit_update_hash_and_rehash_block"}  # undetectable internally only when the tip is edited

    def test_every_scenario_detected_with_anchor_and_documented_gaps_without(self):
        ledger = build_ledger(10, 5, 42)
        anchor = ledger.anchor()
        for name, scenario in tamper.SCENARIOS.items():
            for trial in range(10):
                tampered = copy.deepcopy(ledger)
                scenario(tampered, random.Random(f"{name}:{trial}"))
                self.assertFalse(tampered.validate(anchor=anchor)[0], f"{name} not detected with anchor")
                if name in self.TIP_DEPENDENT:
                    continue
                detected_internal = not tampered.validate()[0]
                self.assertEqual(detected_internal, name not in self.INTERNALLY_UNDETECTABLE, name)

    def test_rehashing_an_edited_block_is_caught_by_its_successor_but_not_at_the_tip(self):
        ledger = build_ledger(5, 5, 42)
        for position, expect_internal in ((2, True), (len(ledger.blocks) - 1, False)):
            tampered = copy.deepcopy(ledger)
            tampered.blocks[position].participants[0]["update_hash"] = synthetic_hash("forged")
            tampered.blocks[position].block_hash = tampered.blocks[position].compute_hash()
            self.assertEqual(not tampered.validate()[0], expect_internal, position)
            self.assertFalse(tampered.validate(anchor=ledger.anchor())[0])

    def test_run_tamper_trials_reports_rates(self):
        rows = {r["scenario"]: r for r in tamper.run_tamper_trials(build_ledger(6, 5, 1), trials=5, seed=0)}
        self.assertEqual(rows["edit_update_hash"]["internal_rate"], 1.0)
        self.assertEqual(rows["full_rewrite_with_recomputed_hashes"]["internal_rate"], 0.0)
        self.assertEqual(rows["full_rewrite_with_recomputed_hashes"]["anchored_rate"], 1.0)
        self.assertEqual(rows["file_byte_flip_on_disk"]["internal_rate"], 1.0)
        self.assertTrue(all(r["anchored_rate"] == 1.0 for r in rows.values()))


class PersistenceTests(unittest.TestCase):
    def test_save_load_round_trip_and_on_disk_tamper(self):
        ledger = build_ledger(5, 5, 42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            size = ledger.save(path)
            self.assertEqual(size, path.stat().st_size)
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())  # atomic replace leaves no temp file
            loaded = Ledger.load(path)
            self.assertEqual([b.to_dict() for b in loaded.blocks], [b.to_dict() for b in ledger.blocks])
            self.assertTrue(loaded.validate(anchor=ledger.anchor())[0])
            lines = path.read_text(encoding="ascii").splitlines()
            block = json.loads(lines[3])
            block["participants"][0]["num_samples"] += 1
            lines[3] = canonical_json(block).decode("ascii")
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            ok, problems = Ledger.load(path).validate()
            self.assertFalse(ok)
            self.assertTrue(any("does not match content" in p for p in problems))

    def test_corrupted_numbers_are_reported_not_raised(self):
        ledger = build_ledger(3, 5, 42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            ledger.save(path)
            text = path.read_text(encoding="ascii").replace('"num_samples":16406', '"num_samples":1e406', 1)
            path.write_text(text, encoding="ascii")
            ok, problems = Ledger.load(path).validate()  # must return, not raise
            self.assertFalse(ok)
            self.assertTrue(any("cannot be canonically encoded" in p for p in problems))

    def test_config_declares_requested_block_structure(self):
        config = json.loads((ROOT / "configs/blockchain/ledger_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["hash_algorithm"], "sha256")
        for name in ("index", "previous_hash", "timestamp", "round", "participants", "aggregation_hash", "block_hash"):
            self.assertIn(name, config["block_fields"])
        self.assertEqual(config["output_root"], "results/blockchain")


if __name__ == "__main__":
    unittest.main()
