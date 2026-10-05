"""End-to-end integration of the FL round ledger with the real FedAvg pipeline (synthetic split, 2 rounds).

Proves: (1) on-chain hashes match the off-chain artifacts, (2) tampering is detected,
(3) FedAvg results are bit-identical with and without recording, and (4) no model
or data content enters the ledger.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.blockchain.fl_recorder import canonical_state_bytes, state_hash, verify_against_off_chain
from src.blockchain.ledger import Ledger
from src.blockchain import tamper
from src.federated.fedavg import get_model_state
from src.federated.run_fedavg import run_experiment
from src.models.cnn1d import ECG1DCNN
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/federated/fedavg_v1.json"


def _overrides(split_root, with_blockchain):
    overrides = {"experiment_id": "bc_test", "data": {"split_root": str(split_root)},
                 "partition": {"num_clients": 3, "min_client_segments": 1},
                 "training": {"rounds": 2, "batch_size": 4}, "runtime": {"cpu_threads": 1}}
    if with_blockchain:
        overrides["blockchain"] = {"enabled": True, "ledger_id": "bc_test_ledger"}
    return overrides


class StateHashTests(unittest.TestCase):
    def test_state_hash_is_deterministic_and_sensitive(self):
        torch.manual_seed(0)
        state = get_model_state(ECG1DCNN())
        self.assertEqual(state_hash(state), state_hash(copy.deepcopy(state)))
        changed = copy.deepcopy(state)
        changed["features.0.weight"].view(-1)[0] += 1e-6
        self.assertNotEqual(state_hash(state), state_hash(changed))
        recast = copy.deepcopy(state)
        recast["classifier.1.bias"] = recast["classifier.1.bias"].double()
        self.assertNotEqual(state_hash(state), state_hash(recast))  # dtype is part of the hash
        self.assertIn(b"features.0.weight", canonical_state_bytes(state))


class FedAvgLedgerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(ROOT)  # run_experiment reads frozen baseline files by repo-relative path
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        _make_split(tmp / "split")
        cls.plain = run_experiment(BASE, output_root=tmp / "plain", overrides=_overrides(tmp / "split", False),
                                   log=lambda *a: None)
        cls.chain = run_experiment(BASE, output_root=tmp / "chain", overrides=_overrides(tmp / "split", True),
                                   log=lambda *a: None)
        cls.root = tmp / "chain"
        cls.ledger = Ledger.load(cls.root / "ledgers" / "bc_test.jsonl")
        cls.anchor = json.loads((cls.root / "anchors" / "bc_test_anchor.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls.tmp.cleanup()

    # (1) recorded hashes match the actual off-chain artifacts
    def test_recorded_hashes_match_off_chain_artifacts(self):
        self.assertEqual(self.ledger.validate(anchor=self.anchor), (True, []))
        self.assertEqual(len(self.ledger.blocks), 3)  # genesis + 2 rounds
        rows = verify_against_off_chain(self.ledger, self.root / "offchain" / "bc_test")
        self.assertEqual(len(rows), 1 + 2 * (1 + 3))
        self.assertTrue(all(r["matches"] for r in rows))
        checkpoint = torch.load(self.root / "models/federated/bc_test_global.pt", weights_only=False)
        selected = checkpoint["selected_round_by_validation_loss"]
        block = next(b for b in self.ledger.blocks if b.round == selected)
        self.assertEqual(state_hash(checkpoint["model_state_dict"]), block.aggregation_hash)
        self.assertTrue(self.chain["blockchain"]["selected_checkpoint_hash_matches_ledger"])
        partition = json.loads((self.root / "metrics/federated/bc_test_client_partition.json").read_text())
        for b in self.ledger.blocks[1:]:
            recorded = {p["client_id"]: p["num_samples"] for p in b.participants}
            self.assertEqual(recorded, {c["client_id"]: c["segments"] for c in partition["clients"]})

    # (2) tampering is detected
    def test_tampering_with_ledger_or_off_chain_artifacts_is_detected(self):
        forged = copy.deepcopy(self.ledger)
        forged.blocks[1].participants[0]["num_samples"] += 1
        self.assertFalse(forged.validate()[0])
        rewritten = copy.deepcopy(self.ledger)
        tamper.full_rewrite_from_block(rewritten, __import__("random").Random(0))
        self.assertTrue(rewritten.validate()[0])  # undetectable without an anchor ...
        self.assertFalse(rewritten.validate(anchor=self.anchor)[0])  # ... detected with it
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp)
            for f in (self.root / "offchain" / "bc_test").glob("*.pt"):
                (archive_dir / f.name).write_bytes(f.read_bytes())
            archive = torch.load(archive_dir / "round_001.pt", weights_only=False)
            first = sorted(archive["client_states"])[0]
            archive["client_states"][first]["features.0.weight"].view(-1)[0] += 1e-4
            torch.save(archive, archive_dir / "round_001.pt")
            rows = verify_against_off_chain(self.ledger, archive_dir)
            bad = [r for r in rows if not r["matches"]]
            self.assertEqual(bad, [{"round": 1, "item": first, "matches": False}])

    # (3) FedAvg results remain equivalent
    def test_fedavg_results_bit_identical_with_and_without_recording(self):
        plain_root = Path(self.cls_tmp()) / "plain"
        a = torch.load(plain_root / "models/federated/bc_test_global.pt", weights_only=False)["model_state_dict"]
        b = torch.load(self.root / "models/federated/bc_test_global.pt", weights_only=False)["model_state_dict"]
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        ha = pd.read_csv(plain_root / "metrics/federated/bc_test_round_history.csv").drop(columns="elapsed_seconds")
        hb = pd.read_csv(self.root / "metrics/federated/bc_test_round_history.csv").drop(columns="elapsed_seconds")
        pd.testing.assert_frame_equal(ha, hb, check_exact=True)
        pa = np.load(plain_root / "metrics/federated/bc_test_test_predictions.npz")
        pb = np.load(self.root / "metrics/federated/bc_test_test_predictions.npz")
        self.assertTrue(all(np.array_equal(pa[k], pb[k]) for k in pa.files))
        self.assertEqual(self.plain["test_metrics"], self.chain["test_metrics"])

    # (4) sensitive model/data content never enters the ledger
    def test_no_model_or_data_content_in_ledger(self):
        def leaves(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from leaves(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from leaves(v)
            else:
                yield obj
        raw = (self.root / "ledgers" / "bc_test.jsonl").read_bytes()
        for line in raw.splitlines()[1:]:
            block = json.loads(line)
            self.assertEqual(set(block), {"index", "previous_hash", "timestamp", "round", "experiment_id",
                                          "participants", "aggregation_hash", "metadata", "block_hash"})
            for value in leaves(block):
                self.assertNotIsInstance(value, float)  # no real numbers at all on-chain
                self.assertTrue(isinstance(value, (str, int, bool)))
                if isinstance(value, str):
                    self.assertLessEqual(len(value), 256)
            self.assertLess(len(line), 2048)
        state = torch.load(self.root / "offchain/bc_test/round_002.pt", weights_only=False)["global_state"]
        for key in ("features.0.weight", "classifier.1.weight"):
            self.assertNotIn(state[key].numpy().tobytes()[:64], raw)
            self.assertNotIn(f"{float(state[key].view(-1)[0]):.6f}".encode(), raw)
        self.assertNotIn(b"segments", raw)
        self.assertNotIn(b"weight", raw)

    def test_blockchain_cannot_be_combined_with_dp_or_he(self):
        dp = json.loads((ROOT / "configs/privacy/dp_fedavg_v1.json").read_text(encoding="utf-8"))["extra_overrides"]
        with self.assertRaises(ValueError):
            run_experiment(BASE, evaluate_test=False, output_root=Path(self.cls_tmp()) / "never",
                           overrides={**_overrides(Path(self.cls_tmp()) / "split", True), **dp})

    def cls_tmp(self):
        return type(self).tmp.name


if __name__ == "__main__":
    unittest.main()
