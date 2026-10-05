"""Tests for the Day 19 component switches and the ablation study design."""
import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from src.federated.ablation import ablation_overrides, load_ablation_config, new_runs
from src.federated.attacks import apply_attack
from src.federated.fedavg import fedavg_aggregate
from src.federated.proposed_defense import components, robust_aggregate, validate_defense_config
from src.federated.proposed_study import load_proposed_config, run_overrides
from src.federated.security import load_security_config
from test_proposed_defense import DEFENSE, _honest_clients

ROOT = Path(__file__).resolve().parents[1]
OFF = {"clipping": False, "cosine_filter": False, "norm_filter": False, "robust_bn_stats": False}


def _with(**switches):
    return {**DEFENSE, "components": {**OFF, **switches}}


def _attacked(kind):
    global_state, clients, keys = _honest_clients()
    attack = {"sign_flip": {"type": "sign_flip", "scale": 1.0}, "scaled": {"type": "scale", "scale": 10.0}}[kind]
    states = clients[:2] + [apply_attack(global_state, clients[2], attack)[0]] + clients[3:]
    return global_state, states, keys


class SwitchTests(unittest.TestCase):
    def test_defaults_are_the_full_method(self):
        self.assertTrue(all(components(DEFENSE).values()))
        global_state, states, keys = _attacked("scaled")
        a = robust_aggregate(global_state, states, [1] * 5, DEFENSE, keys)
        b = robust_aggregate(global_state, states, [1] * 5, _with(clipping=True, cosine_filter=True, norm_filter=True,
                                                                  robust_bn_stats=True), keys)
        for k in a[0]:
            self.assertTrue(torch.equal(a[0][k], b[0][k]))

    def test_validation_of_switches(self):
        for bad in ({"clip": True}, {"clipping": 1}):
            with self.assertRaises(ValueError):
                validate_defense_config({**DEFENSE, "components": bad})

    def test_all_components_off_matches_fedavg(self):
        global_state, states, keys = _attacked("scaled")
        out, rows, summary = robust_aggregate(global_state, states, [10, 20, 30, 40, 50], {**DEFENSE, "components": OFF}, keys)
        plain = fedavg_aggregate(states, [10, 20, 30, 40, 50])
        for k in plain:
            torch.testing.assert_close(out[k], plain[k], rtol=1e-5, atol=1e-5)
        self.assertEqual(summary["accepted_clients"], 5)
        self.assertTrue(rows[2]["flag_high_norm"] and not rows[2]["flagged"])  # computed but not applied

    def test_clip_only_keeps_all_clients_and_bounds_updates(self):
        global_state, states, keys = _attacked("scaled")
        cfg = {**_with(clipping=True), "clipping_norm": 10.0}  # between honest (~5-7) and scaled (~55) norms
        out, rows, summary = robust_aggregate(global_state, states, [1] * 5, cfg, keys)
        self.assertEqual(summary["accepted_clients"], 5)
        self.assertTrue(rows[2]["clipped"])
        self.assertFalse(any(r["clipped"] for i, r in enumerate(rows) if i != 2))
        np.testing.assert_allclose([r["defended_weight"] for r in rows], [0.2] * 5)
        # BatchNorm statistics include the attacker (weighted mean), as in FedAvg
        k = keys[1][0]
        torch.testing.assert_close(out[k], torch.stack([s[k] for s in states]).mean(dim=0))

    def test_each_filter_catches_its_own_attack_only(self):
        cos_only, norm_only = _with(clipping=True, cosine_filter=True), _with(clipping=True, norm_filter=True)
        sf, sc = _attacked("sign_flip"), _attacked("scaled")
        self.assertTrue(robust_aggregate(*sf[:2], [1] * 5, cos_only, sf[2])[1][2]["flagged"])
        self.assertFalse(robust_aggregate(*sc[:2], [1] * 5, cos_only, sc[2])[1][2]["flagged"])   # cos(10Δ, m) is high
        self.assertTrue(robust_aggregate(*sc[:2], [1] * 5, norm_only, sc[2])[1][2]["flagged"])
        self.assertFalse(robust_aggregate(*sf[:2], [1] * 5, norm_only, sf[2])[1][2]["flagged"])  # same norm as honest

    def test_non_robust_bn_stats_use_accepted_weighted_mean_without_floor(self):
        global_state, states, keys = _attacked("sign_flip")
        for s in states[:2]:
            s["features.1.running_var"] = -torch.ones_like(s["features.1.running_var"])
        out, rows, _ = robust_aggregate(global_state, states, [1] * 5, _with(clipping=True, cosine_filter=True), keys)
        accepted = [s for s, r in zip(states, rows) if not r["flagged"]]
        expected = torch.stack([s["features.1.running_var"].double() for s in accepted]).mean(dim=0).float()
        torch.testing.assert_close(out["features.1.running_var"], expected)


class AblationDesignTests(unittest.TestCase):
    def test_thresholds_unchanged_and_only_components_differ(self):
        config = load_ablation_config(ROOT / "configs/ablation/ablation_v1.json")
        frozen = load_proposed_config(ROOT / config["frozen_defense_config"])
        attacks = load_security_config(ROOT / config["attack_config"])
        self.assertEqual(config["seeds"], frozen["seeds"])
        self.assertEqual(config["conditions"], ["clean", "sign_flip", "scaled", "random_noise"])
        runs = new_runs(config)
        self.assertEqual(len(runs), 4 * 4 * 3)
        for run_id, variant, condition, seed in runs:
            ablated = ablation_overrides(config, run_id, variant, condition, seed)
            day18 = run_overrides(frozen, attacks, run_id, condition, seed)
            for key in ("clipping_norm", "cosine_threshold", "norm_ratio_threshold", "bn_variance_floor"):
                self.assertEqual(ablated["defense"][key], frozen["defense"][key])
            self.assertFalse(ablated["defense"]["components"]["robust_bn_stats"])
            for d in (ablated, day18):
                d["defense"] = {k: v for k, v in d["defense"].items() if k not in ("components", "name")}
            self.assertEqual(ablated, day18)
        self.assertEqual(config["variants"]["full"]["components"],
                         {"clipping": True, "cosine_filter": True, "norm_filter": True, "robust_bn_stats": True})


if __name__ == "__main__":
    unittest.main()
