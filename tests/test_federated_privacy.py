import copy
import json
import math
from pathlib import Path
import unittest

import torch
from scipy.stats import norm

from src.federated.config import load_federated_config
from src.federated.fedavg import fedavg_aggregate, get_model_state
from src.federated.privacy import (account, clip_update, dp_fedavg_aggregate, flatten, gdp_delta, gdp_epsilon, gdp_mu,
                                   noise_generator, noise_multiplier_for_epsilon, rdp_epsilon, validate_privacy_config)
from src.federated.robustness import load_robustness_config, planned_runs, run_overrides
from src.federated.run_fedavg import apply_overrides
from src.models.cnn1d import ECG1DCNN

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/privacy/dp_fedavg_v1.json"


def _privacy():
    return copy.deepcopy(json.loads(STUDY.read_text(encoding="utf-8"))["extra_overrides"]["privacy"])


def _states(n_clients=3, scale=0.0, seed=0):
    torch.manual_seed(seed)
    base = get_model_state(ECG1DCNN())
    clients = []
    for i in range(n_clients):
        state = copy.deepcopy(base)
        for key, value in state.items():
            if value.is_floating_point():
                state[key] = value + scale * (i + 1) * torch.ones_like(value)
        clients.append(state)
    return base, clients


class ClippingTests(unittest.TestCase):
    def test_clip_scales_large_updates_to_norm_and_keeps_small_ones(self):
        large, n = clip_update(torch.tensor([3.0, 4.0]), 1.0)
        self.assertAlmostEqual(n, 5.0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(large)), 1.0, places=6)
        torch.testing.assert_close(large, torch.tensor([0.6, 0.8]))
        small, _ = clip_update(torch.tensor([0.3, 0.4]), 1.0)
        torch.testing.assert_close(small, torch.tensor([0.3, 0.4]))
        zero, _ = clip_update(torch.zeros(3), 1.0)
        torch.testing.assert_close(zero, torch.zeros(3))

    def test_no_clipping_and_no_noise_equals_fedavg(self):
        base, clients = _states(scale=0.01)
        out, diag = dp_fedavg_aggregate(base, clients, [10, 20, 30], clipping_norm=1e9, noise_multiplier=0.0,
                                        generator=torch.Generator().manual_seed(0))
        ref = fedavg_aggregate(clients, [10, 20, 30])
        for key in ref:
            torch.testing.assert_close(out[key], ref[key], rtol=1e-5, atol=1e-6)
        self.assertEqual(diag["clipped"], [False, False, False])

    def test_aggregate_sensitivity_bounded_by_clip_times_max_weight(self):
        base, clients = _states(scale=5.0)  # every update far larger than C
        counts, c = [10, 20, 30], 1.0
        full, _ = dp_fedavg_aggregate(base, clients, counts, c, 0.0, torch.Generator().manual_seed(0))
        keys = [k for k, v in base.items() if v.is_floating_point()]
        weights = [n / sum(counts) for n in counts]
        clipped = [clip_update(flatten(s, keys) - flatten(base, keys), c)[0] for s in clients]
        for k in range(len(clients)):  # remove client k's term with other weights fixed
            reduced = sum(w * u for i, (w, u) in enumerate(zip(weights, clipped)) if i != k)
            change = float(torch.linalg.vector_norm(sum(w * u for w, u in zip(weights, clipped)) - reduced))
            self.assertLessEqual(change, c * max(weights) + 1e-9)


class NoiseTests(unittest.TestCase):
    def test_noise_std_matches_calibration_and_is_reproducible(self):
        base, clients = _states(scale=0.0)  # zero updates: output - global is pure noise (+ var clamp)
        counts, c, z = [10, 20, 30], 0.5, 2.0
        out, diag = dp_fedavg_aggregate(base, clients, counts, c, z, noise_generator(42, 7919, 3))
        expected = z * c * max(counts) / sum(counts)
        self.assertAlmostEqual(diag["noise_std"], expected)
        self.assertAlmostEqual(diag["sensitivity"], c * 30 / 60)
        weight_keys = [k for k in base if k.endswith(".weight")]
        residual = torch.cat([(out[k] - base[k]).reshape(-1).double() for k in weight_keys])
        self.assertAlmostEqual(float(residual.std()), expected, delta=0.03 * expected)
        again, _ = dp_fedavg_aggregate(base, clients, counts, c, z, noise_generator(42, 7919, 3))
        other, _ = dp_fedavg_aggregate(base, clients, counts, c, z, noise_generator(42, 7919, 4))
        for key in out:
            torch.testing.assert_close(out[key], again[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(out["features.0.weight"], other["features.0.weight"]))

    def test_running_variance_is_non_negative_and_counters_follow_fedavg(self):
        base, clients = _states(scale=0.0)
        out, _ = dp_fedavg_aggregate(base, clients, [1, 1, 1], 1.0, 50.0, torch.Generator().manual_seed(1))
        for key in out:
            if key.endswith("running_var"):
                self.assertGreaterEqual(float(out[key].min()), 0.0)
            if key.endswith("num_batches_tracked"):
                self.assertEqual(out[key].dtype, torch.long)


class AccountantTests(unittest.TestCase):
    def test_gdp_closed_form_values(self):
        self.assertAlmostEqual(gdp_mu(20, 2.0), math.sqrt(20) / 2.0)
        self.assertAlmostEqual(gdp_delta(0.0, 1.0), 2 * norm.cdf(0.5) - 1, places=12)  # TV distance of mu-GDP
        eps = gdp_epsilon(1e-5, 1.0)
        self.assertAlmostEqual(gdp_delta(eps, 1.0), 1e-5, delta=1e-10)

    def test_epsilon_monotone_in_noise_and_rdp_is_upper_bound(self):
        eps = [account(20, z, 1e-5)["epsilon"] for z in (1.5, 2.69, 5.0)]
        self.assertTrue(eps[0] > eps[1] > eps[2])
        for z in (1.5, 2.69, 5.0):
            result = account(20, z, 1e-5)
            self.assertGreaterEqual(result["epsilon_rdp_upper_bound"], result["epsilon"])
        self.assertAlmostEqual(rdp_epsilon(1, 1.0, 1e-5)[0], min(a / 2 + math.log(1e5) / (a - 1)
                                                                for a in [x / 100 for x in range(101, 51200)]), places=2)

    def test_declared_noise_multiplier_meets_target(self):
        z_min = noise_multiplier_for_epsilon(8.0, 1e-5, 20)
        self.assertAlmostEqual(z_min, 2.6843, places=3)
        self.assertLessEqual(account(20, 2.69, 1e-5)["epsilon"], 8.0)
        with self.assertRaises(ValueError):
            gdp_mu(20, 2.69, sampling_rate=0.5)


class PrivacyConfigTests(unittest.TestCase):
    def test_study_configuration_is_valid_and_recorded(self):
        result = validate_privacy_config(_privacy(), 20, 1.0)
        self.assertAlmostEqual(result["epsilon"], 7.9795, places=3)
        self.assertEqual(result["client_sampling_rate"], 1.0)

    def test_invalid_configurations_are_rejected(self):
        cases = [("client_sampling_rate", 0.5), ("privacy_unit", "record"), ("rounds", 10), ("delta", 0.0),
                 ("clipping_norm", 0.0), ("accountant", "rdp"), ("noise_multiplier", 1.0)]  # z=1 exceeds eps target
        for key, value in cases:
            privacy = _privacy()
            privacy[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_privacy_config(privacy, 20, 1.0)
        privacy = _privacy()
        del privacy["delta"]
        with self.assertRaises(ValueError):
            validate_privacy_config(privacy, 20, 1.0)
        with self.assertRaises(ValueError):
            validate_privacy_config(_privacy(), 20, 0.6)  # partial participation unsupported

    def test_dp_study_differs_from_fedavg_only_by_privacy(self):
        study = load_robustness_config(STUDY)
        day11 = load_robustness_config(ROOT / "configs/federated/fedavg_v1_robustness.json")
        self.assertEqual((study["seeds"], study["primary_partition"]), (day11["seeds"], day11["primary_partition"]))
        self.assertEqual(study["output_root"], "results/privacy")
        base = load_federated_config(ROOT / study["base_config"])
        for run_id, strategy, _, seed in planned_runs(study):
            dp = apply_overrides(base, run_overrides(run_id, strategy, seed, study["extra_overrides"]))
            plain = apply_overrides(base, run_overrides(run_id, strategy, seed))
            self.assertIn("privacy", dp)
            dp.pop("privacy")
            self.assertEqual(dp, plain)


if __name__ == "__main__":
    unittest.main()
