import unittest

import numpy as np

from src.preprocessing.splitting import (
    SPLIT_NAMES,
    plan_group_splits,
    validate_split_membership,
)


def fake_record(record_id, group, value, label="N"):
    n = 1
    return {
        "group": group,
        "data": {
            "segments": np.full((n, 216, 2), value, dtype=np.float32),
            "labels": np.asarray([label]),
            "source_symbols": np.asarray([label]),
            "record_ids": np.asarray([record_id]),
            "patient_groups": np.asarray([group]),
            "annotation_samples": np.asarray([300]),
            "starts": np.asarray([228]),
            "ends": np.asarray([444]),
            "sampling_frequency_hz": np.asarray(360),
            "channel_names": np.asarray(["MLII", "V1"]),
        },
    }


class SplitTests(unittest.TestCase):
    def test_seeded_group_plan_is_reproducible_and_keeps_pair(self):
        groups = {f"record_{i}": [str(100+i)] for i in range(46)}
        groups["shared_201_202"] = ["201", "202"]
        proportions = {"train": .70, "validation": .15, "test": .15}
        first = plan_group_splits(groups, proportions, 42)
        second = plan_group_splits(groups, proportions, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 47)
        self.assertEqual({first[g] for g in first if g == "shared_201_202"}, {first["shared_201_202"]})
        counts = {s: list(first.values()).count(s) for s in SPLIT_NAMES}
        self.assertEqual(counts, {"train": 33, "validation": 7, "test": 7})

    def test_membership_validation_detects_overlap(self):
        records = {"100": fake_record("100", "group_1", 0), "101": fake_record("101", "group_1", 1)}
        split_records = {"train": ["100"], "validation": ["101"], "test": []}
        with self.assertRaisesRegex(ValueError, "Group overlap"):
            validate_split_membership(split_records, records, {"N"})

    def test_membership_validation_detects_identical_segment_across_splits(self):
        records = {"100": fake_record("100", "group_1", 0), "101": fake_record("101", "group_2", 0)}
        split_records = {"train": ["100"], "validation": ["101"], "test": []}
        with self.assertRaisesRegex(ValueError, "Duplicate segment content"):
            validate_split_membership(split_records, records, {"N"})

    def test_membership_validation_checks_labels_and_nonfinite_values(self):
        invalid_label = fake_record("100", "group_1", 1, "BAD")
        with self.assertRaisesRegex(ValueError, "invalid class label"):
            validate_split_membership({"train": ["100"], "validation": [], "test": []}, {"100": invalid_label}, {"N"})
        invalid_signal = fake_record("100", "group_1", 1)
        invalid_signal["data"]["segments"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_split_membership({"train": ["100"], "validation": [], "test": []}, {"100": invalid_signal}, {"N"})


if __name__ == "__main__":
    unittest.main()
