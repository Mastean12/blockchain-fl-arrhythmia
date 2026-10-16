"""Active pre-aggregation commitment gate and hash record tests."""
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from src.blockchain.fl_recorder import FLLedgerRecorder, state_hash, verify_against_off_chain
from src.blockchain.ledger import Ledger
from src.federated.config import load_federated_config
from src.federated.run_fedavg import run_experiment
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]


class LifecycleGateTests(unittest.TestCase):
    def test_invalid_altered_or_missing_commitments_block_aggregation(self):
        state = {"w": torch.tensor([0.0])}
        clients = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([-1.0])}]
        with tempfile.TemporaryDirectory() as temp:
            gate = FLLedgerRecorder("gate-test", "exp", state, Path(temp) / "offchain",
                                    expected_roster=["client_00", "client_01"])
            ids, counts = ["client_00", "client_01"], [3, 5]
            gate.prepare_round(1, ids, counts, clients, state)
            called = []
            self.assertEqual(gate.aggregate_if_prepared(1, ids, counts, clients,
                                                         lambda: called.append(True) or "aggregated"),
                             "aggregated")
            self.assertEqual(called, [True])

            tampered = copy.deepcopy(clients)
            tampered[0]["w"][0] += 0.1
            with self.assertRaisesRegex(RuntimeError, "rejected aggregation"):
                gate.aggregate_if_prepared(1, ids, counts, tampered,
                                           lambda: called.append("bad"))
            self.assertEqual(called, [True])
            with self.assertRaisesRegex(ValueError, "missing, duplicated, or outside"):
                gate.prepare_round(2, ids[:1], counts[:1], clients[:1], state)

    def test_lifecycle_blocks_are_valid_and_detect_tampering(self):
        initial = {"w": torch.tensor([0.0])}
        clients = [{"w": torch.tensor([0.2])}]
        final = {"w": torch.tensor([0.2])}
        with tempfile.TemporaryDirectory() as temp:
            gate = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "offchain",
                                    expected_roster=["client_00"])
            gate.prepare_round(1, ["client_00"], [4], clients, initial)
            gate.aggregate_if_prepared(1, ["client_00"], [4], clients, lambda: None)
            complete = gate.complete_round(1, ["client_00"], [4], clients, final)
            self.assertEqual(complete.metadata["phase"], "complete")
            valid, problems = gate.ledger.validate()
            self.assertTrue(valid, problems)
            self.assertEqual(len(verify_against_off_chain(gate.ledger, Path(temp) / "offchain")), 5)
            forged = copy.deepcopy(gate.ledger)
            forged.blocks[1].participants[0]["update_hash"] = "f" * 64
            self.assertFalse(forged.validate()[0])

    def test_plain_prepared_round_is_accepted_only_at_the_current_tip(self):
        initial = {"w": torch.tensor([0.0])}
        first_updates = [{"w": torch.tensor([0.1])}]
        second_updates = [{"w": torch.tensor([0.2])}]
        with tempfile.TemporaryDirectory() as temp:
            gate = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "offchain",
                                    expected_roster=["client_00"])
            gate.prepare_round(1, ["client_00"], [1], first_updates, initial)
            self.assertTrue(gate.verify_prepared_round(1, ["client_00"], [1], first_updates))
            gate.aggregate_if_prepared(1, ["client_00"], [1], first_updates, lambda: None)
            gate.complete_round(1, ["client_00"], [1], first_updates, first_updates[0])
            gate.prepare_round(2, ["client_00"], [1], second_updates, first_updates[0])

            self.assertFalse(gate.verify_prepared_round(1, ["client_00"], [1], first_updates))
            called = []
            with self.assertRaisesRegex(RuntimeError, "rejected aggregation"):
                gate.aggregate_if_prepared(1, ["client_00"], [1], first_updates,
                                           lambda: called.append(True))
            self.assertEqual(called, [])
            self.assertTrue(gate.verify_prepared_round(2, ["client_00"], [1], second_updates))
            self.assertFalse(gate.ledger.validate()[0])
            self.assertTrue(gate.ledger.validate(allow_pending_prepared=True)[0])

    def test_ledger_rejects_invalid_lifecycle_transitions_and_malformed_metadata(self):
        initial = {"w": torch.tensor([0.0])}
        update = {"w": torch.tensor([0.1])}
        with tempfile.TemporaryDirectory() as temp:
            missing_prepare = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "missing")
            missing_prepare.ledger.append_round(
                1, [{"client_id": "client_00", "num_samples": 1, "update_hash": state_hash(update)}],
                state_hash(update), metadata={"phase": "complete"})
            self.assertFalse(missing_prepare.ledger.validate()[0])

            incomplete = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "incomplete")
            incomplete.ledger.append_round_prepare(
                1, [{"client_id": "client_00", "num_samples": 1, "update_hash": state_hash(update)}],
                state_hash(initial))
            self.assertFalse(incomplete.ledger.validate()[0])

            skipped = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "skipped")
            skipped.ledger.append_round_prepare(
                2, [{"client_id": "client_00", "num_samples": 1, "update_hash": state_hash(update)}],
                state_hash(initial))
            self.assertFalse(skipped.ledger.validate()[0])

            malformed = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "malformed")
            malformed.ledger.blocks[0].metadata = ["not", "a", "mapping"]
            valid, problems = malformed.ledger.validate()
            self.assertFalse(valid)
            self.assertTrue(any("metadata must be a dict" in problem for problem in problems))

            broken_link = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "link")
            broken_link.ledger.append_round(
                1, [{"client_id": "client_00", "num_samples": 1, "update_hash": state_hash(update)}],
                state_hash(update))
            broken_link.ledger.blocks[1].previous_hash = "f" * 64
            self.assertFalse(broken_link.ledger.validate()[0])

    def test_tampered_prior_model_commitment_blocks_aggregation(self):
        initial = {"w": torch.tensor([0.0])}
        update = {"w": torch.tensor([0.1])}
        with tempfile.TemporaryDirectory() as temp:
            gate = FLLedgerRecorder("gate-test", "exp", initial, Path(temp) / "prior-hash",
                                    expected_roster=["client_00"])
            gate.prepare_round(1, ["client_00"], [1], [update], initial)
            prepared = gate.ledger.blocks[-1]
            prepared.aggregation_hash = state_hash({"w": torch.tensor([9.0])})
            prepared.block_hash = prepared.compute_hash()
            self.assertTrue(gate.ledger.validate(allow_pending_prepared=True)[0])
            called = []
            with self.assertRaisesRegex(RuntimeError, "rejected aggregation"):
                gate.aggregate_if_prepared(1, ["client_00"], [1], [update],
                                           lambda: called.append(True))
            self.assertEqual(called, [])

    def test_private_commitment_rejects_missing_duplicate_stale_and_tampered_submissions(self):
        initial = {"w": torch.tensor([0.0])}
        clients = [{"w": torch.tensor([0.2])}, {"w": torch.tensor([-0.1])}]
        ids, weights = ["client_00", "client_01"], [0.6, 0.4]
        with tempfile.TemporaryDirectory() as temp:
            gate = FLLedgerRecorder("private-gate", "exp", initial, Path(temp) / "offchain",
                                    expected_roster=ids, expected_weights=weights)
            with self.assertRaisesRegex(ValueError, "missing, duplicated, or outside"):
                gate.prepare_private_round(1, ids[:1], clients[:1], initial, weights[:1])
            with self.assertRaisesRegex(ValueError, "missing, duplicated, or outside"):
                gate.prepare_private_round(1, [ids[0], ids[0]], clients, initial, weights)
            with self.assertRaisesRegex(RuntimeError, "stale or out of sequence"):
                gate.prepare_private_round(2, ids, clients, initial, weights)
            with self.assertRaisesRegex(RuntimeError, "stale or uncommitted"):
                gate.prepare_private_round(1, ids, clients, {"w": torch.tensor([1.0])}, weights)

            gate.prepare_private_round(1, ids, clients, initial, weights,
                                       metadata={"privacy_protocol": "he_dp_v2_aggregate_noise",
                                                 "he_scheme": "CKKS", "weighting": "fixed_public"})
            called = []
            altered = copy.deepcopy(clients)
            altered[0]["w"][0] += 0.01
            with self.assertRaisesRegex(RuntimeError, "rejected aggregation"):
                gate.aggregate_private_if_prepared(1, ids, altered, weights, initial,
                                                   lambda: called.append(True))
            self.assertEqual(called, [])
            gate.ledger.blocks[-1].participants[0]["update_hash"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "rejected aggregation"):
                gate.aggregate_private_if_prepared(1, ids, clients, weights, initial,
                                                   lambda: called.append(True))
            self.assertEqual(called, [])

    def test_unsupported_blockchain_mode_remains_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Unsupported blockchain mode"):
                run_experiment(
                    ROOT / "configs/blockchain/he_dp_fedavg_v2_lifecycle.json", evaluate_test=False,
                    output_root=Path(temp) / "output",
                    overrides={"blockchain": {"mode": "round_recorder_v1"}}, log=lambda *_: None)

    def test_only_he_dp_v2_ckks_is_supported_with_lifecycle_blockchain(self):
        arm_configs = (
            ROOT / "configs/privacy/dp_fedavg_v1.json",
            ROOT / "configs/he/he_fedavg_v1.json",
            ROOT / "configs/he_dp_fedavg_v1.json",
        )
        with tempfile.TemporaryDirectory() as temp:
            for index, arm_path in enumerate(arm_configs):
                arm = json.loads(arm_path.read_text(encoding="utf-8"))
                with self.subTest(config=arm_path.name):
                    with self.assertRaisesRegex(ValueError, "supports plain FedAvg or HE\\+DP v2"):
                        run_experiment(
                            ROOT / "configs/federated/fedavg_v1.json", evaluate_test=False,
                            output_root=Path(temp) / f"unsupported_{index}",
                            overrides={**arm["extra_overrides"], "blockchain": {
                                "enabled": True, "mode": "lifecycle_gate_v2", "ledger_id": "unsupported"}},
                            log=lambda *_: None)

    def test_fl_training_runs_through_gate_and_reports_overhead(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            _make_split(split)
            common = {"experiment_id": "lifecycle_v2_test", "data": {"split_root": str(split)},
                      "partition": {"num_clients": 3, "min_client_segments": 1, "seed": 19},
                      "training": {"rounds": 1, "local_epochs": 1, "batch_size": 2},
                      "runtime": {"cpu_threads": 1}}
            old_cwd = Path.cwd()
            os.chdir(ROOT)
            try:
                plain = run_experiment(ROOT / "configs/federated/fedavg_v1.json", evaluate_test=False,
                                       output_root=root / "plain", overrides=common, log=lambda *_: None)
                gated = run_experiment(ROOT / "configs/blockchain/fedavg_lifecycle_v2.json", evaluate_test=False,
                                       output_root=root / "gated", overrides={**common, "blockchain": {
                                           "enabled": True, "mode": "lifecycle_gate_v2", "ledger_id": "gate_v2"}},
                                       log=lambda *_: None)
            finally:
                os.chdir(old_cwd)

            ledger = Ledger.load(root / "gated/ledgers/lifecycle_v2_test.jsonl")
            anchor = json.loads((root / "gated/anchors/lifecycle_v2_test_anchor.json").read_text())
            self.assertEqual(ledger.validate(anchor=anchor), (True, []))
            self.assertEqual([b.metadata.get("phase") for b in ledger.blocks[1:]], ["prepared", "complete"])
            self.assertIn("num_samples", ledger.blocks[1].participants[0])
            plain_archive = torch.load(root / "gated/offchain/lifecycle_v2_test/round_001.pt",
                                       map_location="cpu", weights_only=False)
            self.assertIn("client_states", plain_archive)
            plain_validation = {k: v for k, v in plain["selected_round_validation"].items()
                                if k != "elapsed_seconds"}
            gated_validation = {k: v for k, v in gated["selected_round_validation"].items()
                                if k != "elapsed_seconds"}
            self.assertEqual(plain_validation, gated_validation)
            with (root / "gated/metrics/federated/lifecycle_v2_test_blockchain_round_timings.csv").open(
                    newline="", encoding="utf-8") as timing_file:
                timing_rows = list(csv.DictReader(timing_file))
            self.assertTrue(any(row.get("phase") == "prepare" for row in timing_rows))
            self.assertTrue(any(row.get("phase") == "complete" for row in timing_rows))
            summary = json.loads((root / "gated/metrics/federated/lifecycle_v2_test_blockchain_summary.json").read_text())
            self.assertGreater(summary["ledger_bytes"], 0)
            self.assertTrue(gated["blockchain"]["selected_checkpoint_hash_matches_ledger"])

    def test_combined_he_dp_v2_lifecycle_runs_one_round_without_private_ledger_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            _make_split(split)
            base_config = load_federated_config(ROOT / "configs/he_dp_fedavg_v2.json")
            combined_config = load_federated_config(ROOT / "configs/blockchain/he_dp_fedavg_v2_lifecycle.json")
            self.assertEqual(combined_config["privacy"], base_config["privacy"])
            self.assertEqual(combined_config["homomorphic_encryption"], base_config["homomorphic_encryption"])
            self.assertNotEqual(combined_config["output_root"], base_config["output_root"])
            overrides = {
                "experiment_id": "he_dp_v2_blockchain_integration_test",
                "data": {"split_root": str(split)},
                "partition": {"num_clients": 5, "min_client_segments": 1, "seed": 19},
                "training": {"rounds": 1, "local_epochs": 1, "batch_size": 2},
                "runtime": {"cpu_threads": 1},
                "privacy": {"rounds": 1},
                "blockchain": {"anchor_export_path": str(root / "independent-anchor" / "anchor.json")},
            }
            old_cwd = Path.cwd()
            os.chdir(ROOT)
            try:
                summary = run_experiment(
                    ROOT / "configs/blockchain/he_dp_fedavg_v2_lifecycle.json", evaluate_test=False,
                    output_root=root / "output", overrides=overrides, log=lambda *_: None)
            finally:
                os.chdir(old_cwd)

            exp = "he_dp_v2_blockchain_integration_test"
            output = root / "output"
            self.assertEqual(summary["config"]["output_root"], str(output).replace("\\", "/"))
            ledger = Ledger.load(output / "ledgers" / f"{exp}.jsonl")
            anchor_path = root / "independent-anchor" / "anchor.json"
            anchor = json.loads(anchor_path.read_text())
            self.assertEqual(ledger.validate(anchor=anchor), (True, []))
            self.assertFalse((output / "anchors" / f"{exp}_anchor.json").exists())
            self.assertEqual([block.metadata.get("phase") for block in ledger.blocks[1:]],
                             ["prepared", "complete"])
            for block in ledger.blocks[1:]:
                self.assertEqual(block.metadata["privacy_protocol"], "he_dp_v2_aggregate_noise")
                self.assertEqual(block.metadata["he_scheme"], "CKKS")
                self.assertEqual(block.metadata["commitment_format"], "fixed_public_weight")
                self.assertTrue(all("num_samples" not in row for row in block.participants))
                self.assertEqual([row["weight"] for row in block.participants], [0.2] * 5)

            offchain = output / "offchain" / exp
            archived_round = torch.load(offchain / "round_001.pt", map_location="cpu", weights_only=False)
            self.assertEqual(set(archived_round), {"round", "global_state"})
            checked = verify_against_off_chain(ledger, offchain)
            self.assertTrue(all(row["matches"] for row in checked if row["item"] == "global_state"))
            self.assertTrue(all(row.get("verification") == "update_tensor_not_archived_by_policy"
                                for row in checked if row["item"].startswith("client_")))

            checkpoint = torch.load(output / "models/federated" / f"{exp}_global.pt",
                                    map_location="cpu", weights_only=False)
            selected_block = next(block for block in ledger.blocks
                                  if block.round == checkpoint["selected_round_by_validation_loss"]
                                  and block.metadata.get("phase") != "prepared")
            self.assertEqual(state_hash(checkpoint["model_state_dict"]), selected_block.aggregation_hash)
            self.assertTrue(summary["blockchain"]["selected_checkpoint_hash_matches_ledger"])
            self.assertIn(summary["selected_round_by_validation_loss"], (0, 1))

            metrics = output / "metrics/federated"
            with (metrics / f"{exp}_he_dp_v2_costs.csv").open(
                    newline="", encoding="utf-8") as costs_file:
                he_costs = list(csv.DictReader(costs_file))
            self.assertEqual(len(he_costs), 1)
            for key in ("encrypt_seconds_all_clients", "aggregate_seconds_server",
                        "decrypt_seconds_trusted_decryptor"):
                self.assertGreaterEqual(float(he_costs[0][key]), 0.0)
            self.assertGreater(float(he_costs[0]["noise_seconds_trusted_decryptor"]), 0.0)
            with (metrics / f"{exp}_blockchain_round_timings.csv").open(
                    newline="", encoding="utf-8") as timing_file:
                bc_timings = list(csv.DictReader(timing_file))
            self.assertTrue(any(row["phase"] == "prepare" for row in bc_timings))
            self.assertTrue(any(row["phase"] == "complete" for row in bc_timings))
            self.assertGreater(summary["blockchain"]["ledger_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
