import hashlib
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from src.federated.config import load_federated_config
from src.federated.heterogeneity import client_heterogeneity, jensen_shannon
from src.federated.partition import assign_groups_coverage_greedy, build_client_partition
from src.federated.run_fedavg import apply_overrides
from src.federated.robustness import load_robustness_config, planned_runs, run_overrides
from test_federated import _make_split

ROOT = Path(__file__).resolve().parents[1]

# SHA-256 of the official Day 10 fedavg_v1 outputs (immutable reference run).
OFFICIAL_DAY10 = {
    "results/metrics/federated/fedavg_v1_test_predictions.npz": "5031c9465c219bd1dbfb",
    "results/metrics/federated/fedavg_v1_test_metrics.json": "045939aacce5a6927740",
    "results/metrics/federated/fedavg_v1_round_history.csv": "19b1b178931bf306557d",
    "results/models/federated/fedavg_v1_global.pt": "ff030c0755e4f5cdfe10",
}


def _groups(spec):
    return {g: {"records": [g], "segments": sum(c.values()), "class_counts": Counter(c)} for g, c in spec.items()}


class ControlledPartitionTests(unittest.TestCase):
    def test_rare_class_groups_go_to_different_clients(self):
        groups = _groups({"g1": {"N": 10, "X": 1}, "g2": {"N": 10, "X": 2}, "g3": {"N": 10},
                          "g4": {"N": 10}, "g5": {"N": 10, "Y": 1}, "g6": {"N": 10, "Y": 1}})
        assignment = assign_groups_coverage_greedy(groups, 2, seed=0)
        for rare in ("X", "Y"):
            holders = [i for i, client in enumerate(assignment) if any(rare in groups[g]["class_counts"] for g in client)]
            self.assertEqual(len(holders), 2, rare)
        self.assertEqual(sorted(len(c) for c in assignment), [3, 3])

    def test_whole_groups_assigned_once_and_deterministic(self):
        groups = _groups({f"g{i}": {"N": 5 + i, ("V" if i % 3 else "A"): 1} for i in range(11)})
        a = assign_groups_coverage_greedy(groups, 4, seed=7)
        self.assertEqual(a, assign_groups_coverage_greedy(groups, 4, seed=7))
        flat = [g for client in a for g in client]
        self.assertEqual(sorted(flat), sorted(groups))
        self.assertLessEqual(max(len(c) for c in a), 3)  # ceil(11 / 4)

    def test_controlled_partition_passes_leakage_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = _make_split(root)
            partition = build_client_partition(root, 3, 42, "controlled_group_partition")
            groups = [g for c in partition["clients"] for g in c["groups"]]
            self.assertEqual(sorted(groups), splits["train"]["groups"])
            self.assertEqual(partition, build_client_partition(root, 3, 42, "controlled_group_partition"))


class HeterogeneityTests(unittest.TestCase):
    def test_jensen_shannon_bounds(self):
        self.assertAlmostEqual(jensen_shannon([1, 2, 3], [2, 4, 6]), 0.0)
        self.assertAlmostEqual(jensen_shannon([1, 0], [0, 1]), 1.0)

    def test_client_absences_and_dominant_class(self):
        partition = {"clients": [{"client_id": "c0", "groups": ["a"], "class_counts": {"N": 8, "V": 2}},
                                 {"client_id": "c1", "groups": ["b"], "class_counts": {"N": 5}}]}
        rows, summary = client_heterogeneity(partition, ["N", "V", "Z"])
        self.assertEqual(summary["client_class_absences"], 1)  # Z absent from training is not counted
        self.assertEqual(rows[0]["dominant_class"], "N")
        self.assertAlmostEqual(rows[0]["dominant_share"], 0.8)
        self.assertEqual(rows[1]["missing_classes"], "V Z")
        self.assertEqual(summary["clients_per_class"], {"N": 2, "V": 1, "Z": 0})


class RobustnessDesignTests(unittest.TestCase):
    def test_predefined_seeds_and_namespaced_run_ids(self):
        config = load_robustness_config(ROOT / "configs/federated/fedavg_v1_robustness.json")
        self.assertEqual(config["seeds"], [42, 123, 2024])
        runs = planned_runs(config)
        self.assertEqual(len(runs), 6)
        self.assertEqual(runs[0][2], "natural")
        self.assertNotIn("fedavg_v1", {r[0] for r in runs})
        self.assertNotEqual(Path(config["output_root"]), Path("results"))

    def test_overrides_change_only_seed_partition_and_id(self):
        base = load_federated_config(ROOT / "configs/federated/fedavg_v1.json")
        changed = apply_overrides(base, run_overrides("fedavg_v1_natural_s123", "random_group_equal_count", 123))
        for section in ("model", "data", "training", "participation", "aggregation", "model_selection", "runtime"):
            self.assertEqual(changed[section], base[section], section)
        self.assertEqual(changed["partition"]["num_clients"], base["partition"]["num_clients"])
        self.assertEqual((changed["random_seed"], changed["partition"]["seed"]), (123, 123))


class OfficialArtifactTests(unittest.TestCase):
    def test_official_day10_outputs_unchanged(self):
        for relative, prefix in OFFICIAL_DAY10.items():
            path = ROOT / relative
            if not path.exists():
                self.skipTest(f"{relative} not present")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest()[:20], prefix, relative)


if __name__ == "__main__":
    unittest.main()
