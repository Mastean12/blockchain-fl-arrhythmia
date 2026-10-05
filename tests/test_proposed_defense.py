"""Tests for the proposed robust aggregation (Day 18)."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.federated.attacks import apply_attack
from src.federated.fedavg import fedavg_aggregate, get_model_state
from src.federated.proposed_defense import partition_keys, robust_aggregate, validate_defense_config
from src.federated.run_fedavg import run_experiment
from src.models.cnn1d import ECG1DCNN
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]
DEFENSE = {"enabled": True, "name": "test", "mode": "defend", "clipping_norm": 100.0, "cosine_threshold": 0.2,
           "norm_ratio_threshold": 3.0, "bn_variance_floor": 1e-5, "reference_direction": "coordinate_wise_median",
           "bn_running_stats": "coordinate_wise_median_of_accepted"}


def _honest_clients(n=5, seed=0):
    """Clients sharing a common update direction plus small individual noise."""
    torch.manual_seed(seed)
    model = ECG1DCNN()
    keys = partition_keys(model)
    global_state = get_model_state(model)
    direction = {k: 0.05 * torch.randn_like(v) for k, v in global_state.items() if v.is_floating_point()}
    clients = []
    for i in range(n):
        state = copy.deepcopy(global_state)
        for k, v in state.items():
            if k in keys[0]:
                state[k] = v + direction[k] * (1 + 0.1 * i) + 0.005 * torch.randn_like(v)
            elif k in keys[1]:
                state[k] = v.abs() + 0.1 * (i + 1)
            else:
                state[k] = v + 10 + i
        clients.append(state)
    return global_state, clients, keys


class PartitionAndEquivalenceTests(unittest.TestCase):
    def test_state_entries_partitioned(self):
        trainable, running, integer = partition_keys(ECG1DCNN())
        self.assertEqual((len(trainable), len(running), len(integer)), (11, 6, 3))
        self.assertTrue(all(k.endswith(("running_mean", "running_var")) for k in running))
        self.assertIn("features.1.weight", trainable)  # BatchNorm affine parameters are trainable

    def test_honest_round_matches_fedavg_on_parameters_and_uses_median_stats(self):
        global_state, clients, keys = _honest_clients()
        counts = [10, 20, 30, 40, 50]
        out, rows, summary = robust_aggregate(global_state, clients, counts, DEFENSE, keys)
        plain = fedavg_aggregate(clients, counts)
        self.assertEqual(summary["accepted_clients"], 5)
        self.assertFalse(any(r["flagged"] or r["clipped"] for r in rows))
        for k in keys[0]:
            torch.testing.assert_close(out[k], plain[k], rtol=1e-5, atol=1e-6)
        for k in keys[1]:
            torch.testing.assert_close(out[k], torch.stack([c[k] for c in clients]).median(dim=0).values)
        for k in keys[2]:
            self.assertTrue(torch.equal(out[k], plain[k]))
        np.testing.assert_allclose([r["defended_weight"] for r in rows], np.array(counts) / 150)

    def test_observe_mode_returns_plain_fedavg(self):
        global_state, clients, keys = _honest_clients()
        attacked = [*clients[:4], apply_attack(global_state, clients[4], {"type": "scale", "scale": 10.0})[0]]
        out, rows, _ = robust_aggregate(global_state, attacked, [1] * 5, {**DEFENSE, "mode": "observe"}, keys)
        plain = fedavg_aggregate(attacked, [1] * 5)
        for k in plain:
            self.assertTrue(torch.equal(out[k], plain[k]))
        self.assertTrue(rows[4]["flagged"])  # statistics are still computed and logged


class ScreeningTests(unittest.TestCase):
    def _run(self, attack, generator=None):
        global_state, clients, keys = _honest_clients()
        malicious, _ = apply_attack(global_state, clients[2], attack, generator)
        states = clients[:2] + [malicious] + clients[3:]
        return robust_aggregate(global_state, states, [10, 20, 30, 40, 50], DEFENSE, keys)

    def test_each_day17_attack_is_flagged_and_honest_clients_accepted(self):
        cases = {"sign_flip": ({"type": "sign_flip", "scale": 1.0}, None, "flag_low_cosine"),
                 "scaled": ({"type": "scale", "scale": 10.0}, None, "flag_high_norm"),
                 "noise": ({"type": "gaussian_norm_matched", "norm_multiplier": 1.0},
                           torch.Generator().manual_seed(1), "flag_low_cosine")}
        for name, (attack, gen, reason) in cases.items():
            out, rows, summary = self._run(attack, gen)
            self.assertTrue(rows[2]["flagged"] and rows[2][reason], name)
            self.assertEqual(rows[2]["defended_weight"], 0.0, name)
            self.assertFalse(any(r["flagged"] for i, r in enumerate(rows) if i != 2), name)
            self.assertAlmostEqual(sum(r["defended_weight"] for r in rows), 1.0, msg=name)
            self.assertEqual(summary["accepted_clients"], 4, name)

    def test_clipping_bounds_each_accepted_update(self):
        global_state, clients, keys = _honest_clients()
        out, rows, _ = robust_aggregate(global_state, clients, [1] * 5, {**DEFENSE, "clipping_norm": 0.01}, keys)
        self.assertTrue(all(r["clipped"] for r in rows))
        ref = torch.cat([global_state[k].reshape(-1).double() for k in keys[0]])
        new = torch.cat([out[k].reshape(-1).double() for k in keys[0]])
        self.assertLessEqual(float(torch.linalg.vector_norm(new - ref)), 0.01 + 1e-6)

    def test_negative_running_variance_is_floored(self):
        global_state, clients, keys = _honest_clients()
        for c in clients[:3]:
            c["features.1.running_var"] = -torch.ones_like(c["features.1.running_var"])
        out, _, summary = robust_aggregate(global_state, clients, [1] * 5, DEFENSE, keys)
        self.assertGreaterEqual(float(out["features.1.running_var"].min()), float(np.float32(1e-5)))  # float32 floor
        self.assertGreater(summary["bn_variance_entries_floored"], 0)

    def test_fallback_when_every_client_is_flagged(self):
        global_state, clients, keys = _honest_clients()
        out, rows, summary = robust_aggregate(global_state, clients, [1] * 5, {**DEFENSE, "cosine_threshold": 0.9999}, keys)
        self.assertTrue(summary["fallback_median"])
        self.assertTrue(all(r["defended_weight"] == 0.0 for r in rows))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values() if v.is_floating_point()))

    def test_deterministic(self):
        global_state, clients, keys = _honest_clients()
        a = robust_aggregate(global_state, clients, [1] * 5, DEFENSE, keys)
        b = robust_aggregate(global_state, clients, [1] * 5, DEFENSE, keys)
        for k in a[0]:
            self.assertTrue(torch.equal(a[0][k], b[0][k]))


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        validate_defense_config(DEFENSE)
        for change in ({"mode": "x"}, {"clipping_norm": 0}, {"norm_ratio_threshold": 1.0}, {"cosine_threshold": 1.0},
                       {"bn_variance_floor": 0}, {"reference_direction": "mean"}, {"enabled": False}):
            with self.assertRaises(ValueError, msg=str(change)):
                validate_defense_config({**DEFENSE, **change})

    def test_frozen_config_if_present(self):
        path = ROOT / "configs/proposed/robust_defense_v1.json"
        if not path.exists():
            self.skipTest("defense parameters not frozen yet")
        config = json.loads(path.read_text(encoding="utf-8"))
        validate_defense_config(config["defense"])
        self.assertEqual(config["defense"]["mode"], "defend")
        self.assertNotIn(config["calibration"]["seed"], config["seeds"])  # calibrated on a non-evaluation seed


class DefendedRunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        _make_split(tmp / "split")
        cls.base = {"experiment_id": "def", "data": {"split_root": str(tmp / "split")},
                    "partition": {"num_clients": 3, "min_client_segments": 1},
                    "training": {"rounds": 2, "batch_size": 4}, "runtime": {"cpu_threads": 1},
                    "attack": {"enabled": True, "name": "scaled", "type": "scale", "scale": 10.0,
                               "selection_seed_offset": 1717, "client_index": 1}}
        cls.defended = run_experiment(ROOT / "configs/federated/fedavg_v1.json", output_root=tmp / "defended",
                                      overrides={**cls.base, "defense": DEFENSE}, log=lambda *a: None)
        cls.root = tmp

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls.tmp.cleanup()

    def test_defense_runs_with_attack_and_logs_client_decisions(self):
        rows = pd.read_csv(self.root / "defended/metrics/federated/def_defense_client_diagnostics.csv")
        self.assertEqual(sorted(rows["round"].unique()), [1, 2])
        attacker = rows[rows["is_attacker"]]
        self.assertEqual(set(attacker["client_id"]), {"client_01"})
        self.assertTrue(attacker["flag_high_norm"].all())
        self.assertTrue((attacker["defended_weight"] == 0).all())
        for _, group in rows.groupby("round"):
            self.assertAlmostEqual(group["defended_weight"].sum(), 1.0)
        trajectory = np.load(self.root / "defended/metrics/federated/def_global_trajectory.npz")["states"]
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertIn("defense", self.defended)

    def test_defense_cannot_be_combined_with_dp(self):
        dp = json.loads((ROOT / "configs/privacy/dp_fedavg_v1.json").read_text(encoding="utf-8"))["extra_overrides"]
        with self.assertRaises(ValueError):
            run_experiment(ROOT / "configs/federated/fedavg_v1.json", evaluate_test=False,
                           output_root=self.root / "never", overrides={**self.base, "defense": DEFENSE, **dp})


if __name__ == "__main__":
    unittest.main()
