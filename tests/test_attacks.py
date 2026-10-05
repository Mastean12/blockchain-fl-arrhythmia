"""Tests for the Day 17 malicious-client attacks (math, determinism, config, and an end-to-end attacked run)."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.federated.attacks import apply_attack, choose_attacker, noise_generator, validate_attack_config
from src.federated.config import load_federated_config
from src.federated.fedavg import get_model_state
from src.federated.run_fedavg import apply_overrides, run_experiment
from src.federated.security import attack_overrides, load_security_config, planned_attack_runs
from src.models.cnn1d import ECG1DCNN
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/security/attacks_v1.json"


def _states():
    torch.manual_seed(0)
    global_state = get_model_state(ECG1DCNN())
    honest = copy.deepcopy(global_state)
    for key, value in honest.items():
        honest[key] = value + 0.01 * torch.randn_like(value) if value.is_floating_point() else value + 7
    return global_state, honest


def _delta(state, reference):
    return torch.cat([(state[k] - reference[k]).reshape(-1).double() for k in reference if reference[k].is_floating_point()])


class AttackMathTests(unittest.TestCase):
    def test_sign_flip_and_scale_are_exact(self):
        global_state, honest = _states()
        honest_copy = copy.deepcopy(honest)
        flipped, d1 = apply_attack(global_state, honest, {"type": "sign_flip", "scale": 1.0})
        scaled, d2 = apply_attack(global_state, honest, {"type": "scale", "scale": 10.0})
        delta = _delta(honest, global_state)
        torch.testing.assert_close(_delta(flipped, global_state), -delta, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(_delta(scaled, global_state), 10 * delta, rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(d1["cosine_malicious_vs_honest"], -1.0, places=6)
        self.assertAlmostEqual(d2["cosine_malicious_vs_honest"], 1.0, places=6)
        self.assertAlmostEqual(d2["malicious_update_norm"], 10 * d2["honest_update_norm"], places=4)
        for key in honest:
            self.assertTrue(torch.equal(honest[key], honest_copy[key]))  # honest state untouched
            if not honest[key].is_floating_point():
                self.assertTrue(torch.equal(flipped[key], honest[key]))  # integer counters unchanged
            self.assertEqual(flipped[key].dtype, honest[key].dtype)

    def test_norm_matched_noise_is_seeded_and_magnitude_preserving(self):
        global_state, honest = _states()
        cfg = {"type": "gaussian_norm_matched", "norm_multiplier": 1.0, "noise_seed_offset": 31337}
        a, da = apply_attack(global_state, honest, cfg, noise_generator(42, 31337, 3))
        b, _ = apply_attack(global_state, honest, cfg, noise_generator(42, 31337, 3))
        c, _ = apply_attack(global_state, honest, cfg, noise_generator(42, 31337, 4))
        self.assertAlmostEqual(da["malicious_update_norm"], da["honest_update_norm"], places=6)
        self.assertLess(abs(da["cosine_malicious_vs_honest"]), 0.1)  # random direction in ~10k dimensions
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertFalse(torch.equal(a["features.0.weight"], c["features.0.weight"]))
        with self.assertRaises(ValueError):
            apply_attack(global_state, honest, cfg, None)


class AttackConfigTests(unittest.TestCase):
    def test_attacker_choice_is_seeded_and_in_range(self):
        picks = [choose_attacker(s, 1717, 5) for s in (42, 123, 2024)]
        self.assertEqual(picks, [choose_attacker(s, 1717, 5) for s in (42, 123, 2024)])
        self.assertTrue(all(0 <= p < 5 for p in picks))

    def test_validation_rejects_bad_attacks(self):
        good = {"enabled": True, "name": "x", "type": "scale", "scale": 10.0, "selection_seed_offset": 1}
        validate_attack_config(good)
        for change in ({"type": "label_flip"}, {"scale": 0}, {"enabled": False},
                       {"type": "gaussian_norm_matched", "norm_multiplier": 1.0}):
            with self.assertRaises(ValueError, msg=str(change)):
                validate_attack_config({**good, **change})

    def test_study_plan_and_single_variable_overrides(self):
        config = load_security_config(CONFIG)
        self.assertEqual(config["seeds"], [42, 123, 2024])
        self.assertEqual(sorted(config["attacks"]), ["random_noise", "scaled", "sign_flip"])
        runs = planned_attack_runs(config)
        self.assertEqual(len(runs), 9)
        self.assertEqual(config["output_root"], "results/security")
        base = load_federated_config(ROOT / config["base_config"])
        for run_id, name, seed in runs:
            attacked = apply_overrides(base, attack_overrides(config, run_id, name, seed))
            self.assertEqual(attacked["attack"]["name"], name)
            attacked.pop("attack")
            plain = apply_overrides(base, {"experiment_id": run_id, "random_seed": seed,
                                           "partition": {"seed": seed, "strategy": config["partition_strategy"]}})
            self.assertEqual(attacked, plain)


class AttackedRunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        _make_split(tmp / "split")
        base = {"experiment_id": "atk", "data": {"split_root": str(tmp / "split")},
                "partition": {"num_clients": 3, "min_client_segments": 1},
                "training": {"rounds": 2, "batch_size": 4}, "runtime": {"cpu_threads": 1}}
        cls.clean = run_experiment(ROOT / "configs/federated/fedavg_v1.json", output_root=tmp / "clean",
                                   overrides=base, log=lambda *a: None)
        attack = {"enabled": True, "name": "scaled", "type": "scale", "scale": 10.0, "selection_seed_offset": 1717,
                  "client_index": 1}
        cls.attacked = run_experiment(ROOT / "configs/federated/fedavg_v1.json", output_root=tmp / "attacked",
                                      overrides={**base, "attack": attack}, log=lambda *a: None)
        cls.root = tmp

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls.tmp.cleanup()

    def test_attack_is_applied_every_round_and_recorded(self):
        diag = pd.read_csv(self.root / "attacked/metrics/federated/atk_attack_round_diagnostics.csv")
        self.assertEqual(diag["round"].tolist(), [1, 2])
        self.assertTrue((diag["attacker"] == "client_01").all())
        np.testing.assert_allclose(diag["malicious_update_norm"], 10 * diag["honest_update_norm"], rtol=1e-6)
        trajectory = np.load(self.root / "attacked/metrics/federated/atk_global_trajectory.npz")["states"]
        self.assertEqual(trajectory.shape[0], 3)  # rounds 0..2
        self.assertEqual(self.attacked["attack"]["attacker_client_id"], "client_01")

    def test_same_initial_model_but_different_trajectory_than_clean(self):
        clean = pd.read_csv(self.root / "clean/metrics/federated/atk_round_history.csv")
        attacked = pd.read_csv(self.root / "attacked/metrics/federated/atk_round_history.csv")
        self.assertEqual(clean.loc[0, "validation_loss"], attacked.loc[0, "validation_loss"])
        self.assertNotEqual(clean.loc[2, "validation_loss"], attacked.loc[2, "validation_loss"])
        self.assertNotIn("attack", self.clean)
        self.assertFalse((self.root / "clean/metrics/federated/atk_attack_round_diagnostics.csv").exists())

    def test_attack_cannot_be_combined_with_dp(self):
        dp = json.loads((ROOT / "configs/privacy/dp_fedavg_v1.json").read_text(encoding="utf-8"))["extra_overrides"]
        attack = {"enabled": True, "name": "s", "type": "sign_flip", "scale": 1.0, "selection_seed_offset": 1}
        with self.assertRaises(ValueError):
            run_experiment(ROOT / "configs/federated/fedavg_v1.json", evaluate_test=False,
                           output_root=self.root / "never", overrides={**dp, "attack": attack})


if __name__ == "__main__":
    unittest.main()
