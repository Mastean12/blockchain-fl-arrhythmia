import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.federated.data import ClientECGDataset
from src.federated.fedavg import (check_state_compatibility, client_update, fedavg_aggregate, get_model_state,
                                  select_clients, set_model_state)
from src.federated.partition import assign_groups_to_clients, build_client_partition, validate_client_partition
from src.models.cnn1d import ECG1DCNN

LABELS = ["N", "V"]
CLASS_TO_INDEX = {"N": 0, "V": 1}


def _write_shard(path, record_id, group, labels, rng):
    n = len(labels)
    np.savez_compressed(
        path, segments=rng.standard_normal((n, 216, 2)).astype(np.float32), labels=np.asarray(labels),
        record_ids=np.asarray([record_id] * n), patient_groups=np.asarray([group] * n),
        source_symbols=np.asarray(labels), annotation_samples=np.arange(n), starts=np.arange(n),
        ends=np.arange(n) + 216, sampling_frequency_hz=np.asarray(360), channel_names=np.asarray(["MLII", "V1"]))


def _make_split(root):
    """Synthetic split: 6 single-record training groups, a 2-record group in validation, 1 test record."""
    rng = np.random.default_rng(0)
    layout = {"train": {f"record_{r}": [r] for r in ("100", "101", "102", "103", "104", "105")},
              "validation": {"mitdb_group_201_202": ["201", "202"]}, "test": {"record_300": ["300"]}}
    sizes = {"100": 5, "101": 7, "102": 4, "103": 6, "104": 3, "105": 8, "201": 3, "202": 2, "300": 4}
    splits = {}
    for split, groups in layout.items():
        (root / split).mkdir(parents=True)
        seg = 0
        for group, records in groups.items():
            for record in records:
                labels = ["N"] * (sizes[record] - 1) + ["V"]
                _write_shard(root / split / f"{record}.npz", record, group, labels, rng)
                seg += sizes[record]
        splits[split] = {"groups": sorted(groups), "records": sorted(r for rs in groups.values() for r in rs),
                         "segment_count": seg}
    (root / "split_metadata.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return splits


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.splits = _make_split(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_partition_is_reproducible_and_seed_dependent(self):
        groups = [f"g{i}" for i in range(33)]
        a = assign_groups_to_clients(groups, 5, 42)
        self.assertEqual(a, assign_groups_to_clients(list(reversed(groups)), 5, 42))
        self.assertNotEqual(a, assign_groups_to_clients(groups, 5, 7))
        self.assertEqual(sorted(len(c) for c in a), [6, 6, 7, 7, 7])

    def test_no_validation_or_test_data_and_no_group_in_two_clients(self):
        partition = build_client_partition(self.root, 3, 42)
        groups = [g for c in partition["clients"] for g in c["groups"]]
        records = [r for c in partition["clients"] for r in c["records"]]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(set(groups), set(self.splits["train"]["groups"]))
        held_out = set(self.splits["validation"]["records"]) | set(self.splits["test"]["records"])
        self.assertFalse(set(records) & held_out)

    def test_client_sample_counts_sum_to_training_segments(self):
        partition = build_client_partition(self.root, 3, 42)
        for client in partition["clients"]:
            data = ClientECGDataset(self.root, client["records"], CLASS_TO_INDEX)
            self.assertEqual(len(data), client["segments"])
            self.assertEqual(sum(client["class_counts"].values()), client["segments"])
        self.assertEqual(sum(c["segments"] for c in partition["clients"]), self.splits["train"]["segment_count"])

    def test_validation_rejects_held_out_or_duplicated_assignment(self):
        partition = build_client_partition(self.root, 3, 42)
        leaked = json.loads(json.dumps(partition))
        leaked["clients"][0]["groups"].append("record_300")
        leaked["clients"][0]["records"].append("300")
        with self.assertRaises(ValueError):
            validate_client_partition(leaked, self.root)
        duplicated = json.loads(json.dumps(partition))
        duplicated["clients"][1]["groups"].append(duplicated["clients"][0]["groups"][0])
        with self.assertRaises(ValueError):
            validate_client_partition(duplicated, self.root)
        with self.assertRaises(ValueError):
            build_client_partition(self.root, 3, 42, min_client_segments=1000)

    def test_client_dataset_refuses_non_training_records(self):
        with self.assertRaises(FileNotFoundError):
            ClientECGDataset(self.root, ["300"], CLASS_TO_INDEX)
        with self.assertRaises(FileNotFoundError):
            ClientECGDataset(self.root, ["201"], CLASS_TO_INDEX)


class FedAvgTests(unittest.TestCase):
    def test_weighted_aggregation_matches_manual_average(self):
        a = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor(10, dtype=torch.long)}
        b = {"w": torch.tensor([4.0, 8.0]), "n": torch.tensor(20, dtype=torch.long)}
        out = fedavg_aggregate([a, b], [1, 3])
        torch.testing.assert_close(out["w"], torch.tensor([3.25, 6.5]))
        self.assertEqual(int(out["n"]), 18)  # round(0.25*10 + 0.75*20) = round(17.5)
        self.assertEqual(out["n"].dtype, torch.long)
        same = fedavg_aggregate([a, a], [5, 9])
        torch.testing.assert_close(same["w"], a["w"])

    def test_aggregation_rejects_bad_inputs(self):
        a = {"w": torch.zeros(2)}
        with self.assertRaises(ValueError):
            fedavg_aggregate([a, {"w": torch.zeros(3)}], [1, 1])
        with self.assertRaises(ValueError):
            fedavg_aggregate([a, a], [1, 0])
        with self.assertRaises(ValueError):
            fedavg_aggregate([a], [1, 2])

    def test_model_state_shapes_are_compatible_and_loadable(self):
        model = ECG1DCNN()
        state = get_model_state(model)
        check_state_compatibility(state, get_model_state(ECG1DCNN()))
        aggregated = fedavg_aggregate([state, get_model_state(ECG1DCNN())], [2, 3])
        set_model_state(ECG1DCNN(), aggregated)
        self.assertEqual(list(aggregated), list(state))
        with self.assertRaises(ValueError):
            check_state_compatibility(state, get_model_state(ECG1DCNN(num_classes=5)))

    def test_client_selection(self):
        self.assertEqual(select_clients(5, 1.0, 42, 1), [0, 1, 2, 3, 4])
        picked = select_clients(10, 0.3, 42, 4)
        self.assertEqual(len(picked), 3)
        self.assertEqual(picked, select_clients(10, 0.3, 42, 4))

    def test_client_update_is_reproducible_with_fixed_seed(self):
        torch.use_deterministic_algorithms(True)
        gen = torch.Generator().manual_seed(0)
        data = TensorDataset(torch.randn(40, 2, 216, generator=gen), torch.randint(0, 15, (40,), generator=gen))
        torch.manual_seed(1)
        global_state = get_model_state(ECG1DCNN())
        kwargs = dict(local_epochs=2, batch_size=8, learning_rate=1e-3, optimizer_name="Adam", seed=123)
        first = client_update(ECG1DCNN(), global_state, data, **kwargs)
        second = client_update(ECG1DCNN(), global_state, data, **kwargs)
        for key in global_state:
            torch.testing.assert_close(first["state"][key], second["state"][key], rtol=0, atol=0)
        self.assertEqual(first["num_samples"], 40)
        self.assertFalse(torch.equal(first["state"]["classifier.1.weight"], global_state["classifier.1.weight"]))


if __name__ == "__main__":
    unittest.main()
