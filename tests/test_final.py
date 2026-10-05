"""Day 20 tests: confirmation design (fresh seeds, frozen thresholds), control equivalence, final artifacts."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.federated.final_validation import USED_SEEDS, final_overrides, load_final_config, planned_runs
from src.federated.proposed_study import load_proposed_config, run_overrides
from src.federated.run_fedavg import run_experiment
from src.federated.security import load_security_config
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final/final_validation_v1.json"


class FinalDesignTests(unittest.TestCase):
    def test_fresh_seeds_frozen_thresholds_and_variant5(self):
        config = load_final_config(CONFIG)
        self.assertFalse(set(config["seeds"]) & USED_SEEDS)
        self.assertEqual(len(planned_runs(config)), 2 * 4 * 3)
        frozen = load_proposed_config(ROOT / config["frozen_defense_config"])
        attacks = load_security_config(ROOT / config["attack_config"])
        for run_id, arm, condition, seed in planned_runs(config):
            ov = final_overrides(config, run_id, arm, condition, seed)
            for key in ("clipping_norm", "cosine_threshold", "norm_ratio_threshold", "bn_variance_floor"):
                self.assertEqual(ov["defense"][key], frozen["defense"][key])
            if arm == "variant5":
                self.assertEqual(ov["defense"]["mode"], "defend")
                self.assertEqual(ov["defense"]["components"], {"clipping": True, "cosine_filter": True,
                                                               "norm_filter": True, "robust_bn_stats": False})
            else:
                self.assertEqual(ov["defense"]["mode"], "observe")
                self.assertNotIn("components", ov["defense"])
            day18 = run_overrides(frozen, attacks, run_id, condition, seed)
            for d in (ov, day18):
                d["defense"] = {k: v for k, v in d["defense"].items() if k not in ("components", "name", "mode")}
            self.assertEqual(ov, day18)  # only mode / components / name differ from the Day 18 protocol

    def test_used_seeds_are_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["seeds"] = [42, 999, 1000]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_final_config(path)


class ObserveControlEquivalenceTests(unittest.TestCase):
    """The FedAvg control (defense in observe mode) must equal plain FedAvg through the real pipeline."""

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        _make_split(tmp / "split")
        base = {"experiment_id": "obs", "data": {"split_root": str(tmp / "split")},
                "partition": {"num_clients": 3, "min_client_segments": 1},
                "training": {"rounds": 2, "batch_size": 4}, "runtime": {"cpu_threads": 1}}
        observe = {**json.loads((ROOT / "configs/proposed/robust_defense_v1.json").read_text())["defense"],
                   "mode": "observe"}
        run_experiment(ROOT / "configs/federated/fedavg_v1.json", output_root=tmp / "plain", overrides=base,
                       log=lambda *a: None)
        run_experiment(ROOT / "configs/federated/fedavg_v1.json", output_root=tmp / "observe",
                       overrides={**base, "defense": observe}, log=lambda *a: None)
        cls.root = tmp

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls.tmp.cleanup()

    def test_observe_mode_is_bit_identical_to_plain_fedavg(self):
        a = torch.load(self.root / "plain/models/federated/obs_global.pt", weights_only=False)["model_state_dict"]
        b = torch.load(self.root / "observe/models/federated/obs_global.pt", weights_only=False)["model_state_dict"]
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        ha = pd.read_csv(self.root / "plain/metrics/federated/obs_round_history.csv").drop(columns="elapsed_seconds")
        hb = pd.read_csv(self.root / "observe/metrics/federated/obs_round_history.csv").drop(columns="elapsed_seconds")
        pd.testing.assert_frame_equal(ha, hb, check_exact=True)
        pa = np.load(self.root / "plain/metrics/federated/obs_test_predictions.npz")
        pb = np.load(self.root / "observe/metrics/federated/obs_test_predictions.npz")
        self.assertTrue(all(np.array_equal(pa[k], pb[k]) for k in pa.files))
        self.assertTrue((self.root / "observe/metrics/federated/obs_global_trajectory.npz").exists())


class FinalArtifactTests(unittest.TestCase):
    def test_consolidated_summary_structure_if_present(self):
        path = ROOT / "results/final/final_summary.json"
        if not path.exists():
            self.skipTest("final summary not generated yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        for section in ("utility", "privacy_confidentiality", "integrity_auditability", "robustness",
                        "communication", "computation", "research_questions"):
            self.assertIn(section, summary)
        for rq in ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5"):
            self.assertIn(rq, summary["research_questions"])
            self.assertIn(summary["research_questions"][rq]["status"], {"answered", "partially answered", "not answered"})

    def test_manifest_lists_existing_configs_if_present(self):
        path = ROOT / "results/final/experiment_manifest.json"
        if not path.exists():
            self.skipTest("manifest not generated yet")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest["experiments"]), 10)
        for experiment in manifest["experiments"]:
            for cfg in experiment.get("configs", []):
                self.assertTrue((ROOT / cfg).exists(), cfg)


if __name__ == "__main__":
    unittest.main()
