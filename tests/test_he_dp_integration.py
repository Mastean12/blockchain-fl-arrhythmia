"""Integrated HE+DP round tests: prove clipping/noising and CKKS coexist in one FL run."""
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from src.federated.privacy import account
from src.federated.robustness import load_robustness_config, planned_runs, run_overrides
from src.federated.config import load_federated_config
from src.federated.run_fedavg import apply_overrides, run_experiment
from src.he.ckks import CKKSSession
from src.he.config import validate_he_config
from src.federated.privacy import validate_privacy_config
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]


def _he_config():
    return {
        "enabled": True, "scheme": "CKKS", "library": "tenseal",
        "poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale_bits": 40, "security_level_bits": 128,
        "encrypted_quantity": "clipped_noisy_client_update",
        "aggregation": "sample_weighted_fedavg",
    }


def _dp_config(rounds=1):
    return {
        "enabled": True, "privacy_unit": "client", "clipping_norm": 1.0,
        "noise_multiplier": 2.69, "delta": 1e-5,
        "accountant": "gdp_full_participation", "client_sampling_rate": 1.0,
        "rounds": rounds, "noise_seed_offset": 7919,
        "noise_placement": "client_before_encryption",
    }


class IntegratedPrivacyProtocolTests(unittest.TestCase):
    def test_versioned_study_config_is_reproducible_and_valid(self):
        study = load_robustness_config(ROOT / "configs/he_dp_fedavg_v1.json")
        self.assertEqual([run[-1] for run in planned_runs(study)], [42, 123, 2024])
        base = load_federated_config(ROOT / study["base_config"])
        run_id, strategy, _, seed = planned_runs(study)[0]
        config = apply_overrides(base, run_overrides(run_id, strategy, seed, study["extra_overrides"]))
        self.assertEqual(validate_he_config(config["homomorphic_encryption"]).slots, 4096)
        accounting = validate_privacy_config(config["privacy"], config["training"]["rounds"],
                                             config["participation"]["fraction"])
        self.assertAlmostEqual(accounting["epsilon"], 7.98, places=1)

    def test_public_context_encrypts_but_cannot_decrypt_noise_payload(self):
        session = CKKSSession()
        server = session.server_view()
        self.assertFalse(server.public_context.has_secret_key())
        self.assertFalse(hasattr(server, "secret_context"))
        self.assertFalse(hasattr(server, "decrypt"))
        ciphertext = session.encrypt([0.25, -0.5])
        aggregate = server.aggregate([ciphertext], [1.0])
        self.assertEqual(len(session.decrypt(aggregate, 2)), 2)

    def test_accountant_uses_full_round_composition_and_privacy_unit(self):
        result = account(20, 2.69, 1e-5, sampling_rate=1.0)
        self.assertEqual(result["rounds"], 20)
        self.assertAlmostEqual(result["epsilon"], 7.98, places=1)
        with self.assertRaises(ValueError):
            account(20, 2.69, 1e-5, sampling_rate=0.5)

    def test_training_run_executes_dp_and_he_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "split"
            _make_split(split)
            overrides = {
                "experiment_id": "he_dp_integration_test",
                "data": {"split_root": str(split)},
                "partition": {"num_clients": 5, "min_client_segments": 1, "seed": 19},
                "training": {"rounds": 1, "local_epochs": 1, "batch_size": 2},
                "runtime": {"cpu_threads": 1},
                "privacy": _dp_config(1),
                "homomorphic_encryption": _he_config(),
            }
            old_cwd = Path.cwd()
            os.chdir(ROOT)
            try:
                summary = run_experiment(ROOT / "configs/federated/fedavg_v1.json", evaluate_test=False,
                                         output_root=root / "output", overrides=overrides, log=lambda *_: None)
            finally:
                os.chdir(old_cwd)

            self.assertIn("privacy", summary)
            self.assertIn("homomorphic_encryption", summary)
            self.assertIn("integrated_privacy_pipeline", summary)
            self.assertAlmostEqual(summary["privacy"]["accounting"]["epsilon"],
                                   account(1, 2.69, 1e-5)["epsilon"])
            metrics = root / "output/metrics/federated"
            with (metrics / "he_dp_integration_test_he_round_diagnostics.csv").open(newline="") as stream:
                he_rows = list(csv.DictReader(stream))
            with (metrics / "he_dp_integration_test_dp_round_diagnostics.csv").open(newline="") as stream:
                dp_rows = list(csv.DictReader(stream))
            self.assertEqual(len(he_rows), 1)
            self.assertEqual(len(dp_rows), 1)
            self.assertEqual(he_rows[0]["protocol"], "client_clip_gaussian_noise_then_ckks_weighted_sum")
            self.assertGreater(int(he_rows[0]["uplink_ciphertext_bytes_total"]), 0)
            self.assertGreater(float(he_rows[0]["encrypt_seconds_all_clients"]), 0)
            self.assertGreater(float(he_rows[0]["aggregate_seconds_server"]), 0)
            self.assertGreater(float(he_rows[0]["decrypt_seconds_key_holders"]), 0)
            self.assertLess(float(he_rows[0]["aggregate_max_abs_error"]), 1e-4)
            self.assertTrue((root / "output/private/he_dp_integration_test_noise_seeds.json").exists())
            self.assertFalse((metrics / "he_dp_integration_test_client_round_history.csv").exists())
            accounting = json.loads((metrics / "he_dp_integration_test_privacy_accounting.json").read_text())
            self.assertEqual(accounting["accounting"]["delta"], 1e-5)
            self.assertNotIn("client_seeds", json.dumps(accounting))


if __name__ == "__main__":
    unittest.main()
