"""Protocol and integration tests for fixed-weight aggregate-noise HE+DP v2."""
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.federated.privacy import account, he_dp_v2_account
from src.federated.run_fedavg import run_experiment
from src.he.aggregation import aggregate_he_dp_v2
from src.he.ckks import CKKSSession
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]


class V2AccountAndMechanismTests(unittest.TestCase):
    def test_fixed_weight_sensitivity_and_independent_accountant_crosscheck(self):
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        result = he_dp_v2_account(weights, clipping_norm=1.5, noise_multiplier=2.0,
                                  rounds=4, delta=1e-5, roster_size=5)
        self.assertAlmostEqual(result["sensitivity_l2"], 0.45)
        self.assertAlmostEqual(result["aggregate_noise_std"], 0.9)
        reference = account(4, 2.0, 1e-5)
        self.assertAlmostEqual(result["epsilon"], reference["epsilon"], places=10)
        self.assertGreaterEqual(result["rdp_epsilon_upper_bound"], result["epsilon"])
        with self.assertRaisesRegex(ValueError, "renormalization is forbidden"):
            he_dp_v2_account([0.2] * 4 + [0.19], 1.0, 2.0, 4, 1e-5, 5)

    def test_accountant_rejects_non_finite_parameters_and_non_integer_rounds(self):
        for kwargs in (
                {"clipping_norm": float("nan")},
                {"noise_multiplier": float("inf")},
                {"delta": float("nan")},
                {"rounds": float("nan")},
                {"rounds": 1.5}):
            params = {"clipping_norm": 1.0, "noise_multiplier": 2.0, "rounds": 4, "delta": 1e-5}
            params.update(kwargs)
            with self.assertRaises(ValueError, msg=str(kwargs)):
                he_dp_v2_account([0.2] * 5, params["clipping_norm"], params["noise_multiplier"],
                                 params["rounds"], params["delta"], 5)

    def test_ckks_aggregate_then_single_noise_and_bn_integer_hold(self):
        global_state = OrderedDict(weight=torch.tensor([0.0, 0.0]),
                                  bn_num_batches_tracked=torch.tensor(7, dtype=torch.long))
        updates = [OrderedDict(weight=torch.tensor([2.0, 0.0]),
                               bn_num_batches_tracked=torch.tensor(100, dtype=torch.long)),
                   OrderedDict(weight=torch.tensor([0.0, -2.0]),
                               bn_num_batches_tracked=torch.tensor(200, dtype=torch.long))]
        weights, clipping_norm, multiplier, seed = [0.5, 0.5], 1.0, 2.0, 39
        session = CKKSSession()
        client_encryptor = session.client_encryptor()
        self.assertFalse(client_encryptor.public_context.has_secret_key())
        self.assertFalse(session.server_view().public_context.has_secret_key())
        captured_plaintexts = []
        encrypt = client_encryptor.encrypt
        client_encryptor.encrypt = lambda vector: (captured_plaintexts.append(np.asarray(vector).copy())
                                                    or encrypt(vector))
        session.client_encryptor = lambda: client_encryptor

        noise = torch.normal(0.0, multiplier * clipping_norm * max(weights), size=(2,),
                             generator=torch.Generator().manual_seed(seed), dtype=torch.float64).numpy()
        clean = np.asarray([0.5, -0.5])  # each client delta is clipped to norm 1
        expected = clean + noise
        result, diagnostics = aggregate_he_dp_v2(global_state, updates, weights, session, clipping_norm,
                                                  multiplier, torch.Generator().manual_seed(seed))
        np.testing.assert_allclose(result["weight"].numpy(), expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(captured_plaintexts, [[1.0, 0.0], [0.0, -1.0]], atol=0, rtol=0)
        self.assertEqual(result["bn_num_batches_tracked"].item(), 7)
        self.assertEqual(diagnostics["aggregate_noise_std"], 1.0)
        self.assertLess(diagnostics["ckks_clean_aggregate_max_abs_error"], 1e-5)
        self.assertGreater(diagnostics["uplink_ciphertext_bytes_total"], 0)

    def test_one_round_pipeline_runs_he_dp_together_and_selects_on_validation_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            _make_split(split)
            overrides = {
                "experiment_id": "he_dp_v2_integration_test",
                "data": {"split_root": str(split)},
                "partition": {"num_clients": 5, "min_client_segments": 1, "seed": 19},
                "training": {"rounds": 1, "local_epochs": 1, "batch_size": 2},
                "runtime": {"cpu_threads": 1},
                "privacy": {"rounds": 1, "fixed_roster": [f"client_{i:02d}" for i in range(5)]},
            }
            old_cwd = Path.cwd()
            os.chdir(ROOT)
            try:
                summary = run_experiment(ROOT / "configs/he_dp_fedavg_v2.json", evaluate_test=False,
                                         output_root=root / "output", overrides=overrides, log=lambda *_: None)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(summary["integrated_privacy_pipeline"]["process_isolation_enforced"], False)
            self.assertEqual(summary["selected_round_by_validation_loss"], 0)
            metrics = root / "output/metrics/federated"
            history = list(csv.DictReader((metrics / "he_dp_v2_integration_test_round_history.csv").open()))
            expected_round = min(range(len(history)), key=lambda i: float(history[i]["validation_loss"]))
            self.assertEqual(summary["selected_round_by_validation_loss"], expected_round)
            accounting = json.loads((metrics / "he_dp_v2_integration_test_privacy_accounting.json").read_text())
            self.assertIn("add/remove one client contribution", accounting["accounting"]["adjacency"])
            self.assertFalse((metrics / "he_dp_v2_integration_test_he_round_diagnostics.csv").exists())
            private = root / "output/private/he_dp_v2_integration_test_sensitive_diagnostics.json"
            self.assertTrue(private.exists())
            rows = json.loads(private.read_text())
            self.assertEqual(rows[0]["protocol"],
                             "fixed_weight_clip_ckks_sum_trusted_decryptor_aggregate_gaussian_v2")
            artifact = torch.load(root / "output/models/federated/he_dp_v2_integration_test_global.pt",
                                 map_location="cpu", weights_only=False)
            self.assertNotIn("partition", artifact)
            private_seed_file = root / "output/private/he_dp_v2_integration_test_noise_seeds.json"
            self.assertTrue(private_seed_file.exists())
            private_seed = str(json.loads(private_seed_file.read_text())["trusted_decryptor_noise_seed"])
            output_root = root / "output"
            public_files = (path for path in output_root.rglob("*")
                            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"}
                            and "private" not in path.relative_to(output_root).parts)
            for public_file in public_files:
                self.assertNotIn(private_seed, public_file.read_text(encoding="utf-8"),
                                 str(public_file.relative_to(output_root)))

    def test_he_dp_v2_rejects_mismatched_privacy_and_he_configuration(self):
        config_path = ROOT / "configs/he_dp_fedavg_v2.json"
        cases = (
            ({"privacy": {"enabled": False}}, "privacy.enabled=true"),
            ({"training": {"rounds": 1}, "privacy": {"rounds": 2}}, "privacy.rounds"),
            ({"training": {"rounds": 1}, "privacy": {"rounds": 1.5}}, "privacy.rounds"),
            ({"privacy": {"fixed_roster": ["client_00"] * 5}}, "fixed_roster"),
            ({"privacy": {"noise_placement": "client_before_encryption"}}, "after trusted decryption"),
            ({"homomorphic_encryption": {"encrypted_quantity": "client_update"}},
             "clipped client updates"),
        )
        for index, (overrides, message) in enumerate(cases):
            with self.assertRaisesRegex(ValueError, message):
                run_experiment(config_path, evaluate_test=False, output_root=ROOT / "temp" / f"invalid_he_dp_{index}",
                               overrides=overrides, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
