import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.preprocessing.config import load_config
from src.preprocessing.svdb import (build_svdb_splits, classify_symbols,
                                    process_svdb, resample_signal_and_annotations,
                                    validate_svdb_splits)
from src.preprocessing.storage import load_record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preprocessing/svdb_v1.json"


class SVDBIntegrationTests(unittest.TestCase):
    def test_annotation_mapping_and_explicit_exclusions(self):
        cfg = load_config(CONFIG)
        audit = classify_symbols(["N", "S", "V", "B", "+", "~"], cfg)
        self.assertEqual(audit["mapped_beat_counts"], {"N": 1, "S": 1, "V": 1})
        self.assertEqual(audit["excluded_beat_counts"], {"B": 1})
        self.assertEqual(audit["nonbeat_event_counts"], {"+": 1, "~": 1})
        with self.assertRaisesRegex(ValueError, "Unmapped SVDB"):
            classify_symbols(["Z"], cfg)

    def test_polyphase_resampling_maps_annotations_consistently(self):
        x = np.arange(128 * 2, dtype=float).reshape(128, 2)
        y, mapped, valid = resample_signal_and_annotations(x, [0, 64, 127], 128, 360, 45, 16)
        self.assertEqual(y.shape, (360, 2))
        self.assertEqual(mapped.tolist(), [0, 180, 357])
        self.assertTrue(valid.all())

    def test_preprocesses_annotations_to_shared_shape_and_builds_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            raw = tmp / "raw"
            raw.mkdir()
            ids = [str(800 + i) for i in range(6)]
            (raw / "RECORDS").write_text("\n".join(ids) + "\n", encoding="utf-8")
            for rid in ids:
                for ext in ("hea", "dat", "atr"):
                    (raw / f"{rid}.{ext}").write_bytes(b"fixture")

            # The last annotation models a legitimate source annotation exactly
            # at the exclusive record endpoint: count it, but make no window.
            samples = np.asarray([220, 390, 560, 730, 900, 1070, 1200], dtype=np.int64)
            symbols = ["N", "S", "V", "F", "B", "+", "N"]
            def fake_record(path, physical=True):
                rid = Path(path).name
                t = np.arange(1200, dtype=float) / 128.0
                phase = int(rid) / 100.0
                sig = np.column_stack((np.sin(2*np.pi*3*t + phase), np.cos(2*np.pi*2*t + phase)))
                return SimpleNamespace(p_signal=sig, fs=128, sig_name=["ECG1", "ECG2"])
            def fake_annotation(path, extension):
                return SimpleNamespace(sample=samples, symbol=symbols)

            processed, split = tmp / "processed", tmp / "splits"
            with patch("src.preprocessing.svdb.wfdb.rdrecord", side_effect=fake_record), \
                 patch("src.preprocessing.svdb.wfdb.rdann", side_effect=fake_annotation):
                manifest = process_svdb(raw, processed, CONFIG, checksums=False)
            self.assertEqual(manifest["record_count"], len(ids))
            self.assertEqual(manifest["source_annotation_symbol_counts"],
                             {"+": 6, "B": 6, "F": 6, "N": 12, "S": 6, "V": 6})
            self.assertEqual(manifest["terminal_boundary_excluded_count"], 6)
            first = load_record(processed / f"{ids[0]}.npz")
            self.assertEqual(first["segments"].shape[1:], (216, 2))
            self.assertEqual(set(first["labels"].tolist()), {"N", "S", "V", "F"})
            self.assertEqual(set(first["patient_groups"].tolist()), {f"record_{ids[0]}"})

            metadata = build_svdb_splits(processed, split)
            metadata_repeat = build_svdb_splits(processed, tmp / "splits_repeat")
            self.assertFalse(metadata["patient_disjoint_claim"])
            self.assertEqual(metadata["splits"], metadata_repeat["splits"])
            sets = [set(metadata["splits"][part]["records"]) for part in ("train", "validation", "test")]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertTrue((split / "split_metadata.json").is_file())
            verified = validate_svdb_splits(processed, split,
                                            ROOT / "configs/preprocessing/svdb_split_v1.json", CONFIG)
            self.assertTrue(verified["shard_copies_match"])
            self.assertFalse(verified["patient_disjoint_claim"])


if __name__ == "__main__":
    unittest.main()
