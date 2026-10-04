import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.preprocessing.config import load_config
from src.preprocessing.filtering import filter_signal
from src.preprocessing.normalization import normalize_segments
from src.preprocessing.segmentation import segment_record
from src.preprocessing.validation import validate_batch

ROOT=Path(__file__).resolve().parents[1]
CFG=load_config(ROOT/"configs/preprocessing/mitbih_v1.json")

class PreprocessingTests(unittest.TestCase):
    def test_filter_shape_finite_and_attenuates_baseline(self):
        t=np.arange(3600)/360
        x=(np.sin(2*np.pi*1*t)+2*np.sin(2*np.pi*80*t))[:,None]
        y=filter_signal(x,360,CFG)
        self.assertEqual(y.shape,x.shape); self.assertTrue(np.isfinite(y).all())
        self.assertLess(np.std(y[100:-100]),np.std(x[100:-100]))

    def test_segments_keep_label_alignment_and_drop_edges(self):
        x=np.random.default_rng(5).normal(size=(1000,2))
        batch=segment_record(x,[5,200,500,800],["N","+","V","A"],"100","record_100",CFG)
        self.assertEqual(batch.edge_dropped,1); self.assertEqual(batch.nonbeat_skipped,1)
        self.assertEqual(batch.labels.tolist(),["V","A"])
        self.assertEqual(batch.annotation_samples.tolist(),[500,800])
        batch.segments=normalize_segments(batch.segments,CFG)
        self.assertTrue(validate_batch(batch,CFG))

    def test_record_and_group_provenance_strings_are_not_truncated(self):
        x=np.random.default_rng(7).normal(size=(500,2))
        batch=segment_record(x,[250],["N"],"201","mitdb_group_201_202",CFG)
        self.assertEqual(batch.record_ids.tolist(),["201"])
        self.assertEqual(batch.patient_groups.tolist(),["mitdb_group_201_202"])

    def test_unknown_symbol_rejected(self):
        with self.assertRaises(ValueError): segment_record(np.ones((500,2)),[200],["?"],"100","g",CFG)

    def test_invalid_shape_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): filter_signal(np.ones(10),360,CFG)
        with self.assertRaises(ValueError): normalize_segments(np.full((1,216,2),np.nan),CFG)

    def test_constant_channel_and_provenance_mismatch_rejected(self):
        with self.assertRaises(ValueError): normalize_segments(np.ones((1,216,2)),CFG)
        batch=segment_record(np.random.default_rng(1).normal(size=(600,2)),[300],["N"],"100","g",CFG)
        batch.labels[0]="V"
        batch.segments=normalize_segments(np.random.default_rng(2).normal(size=(1,216,2)),CFG)
        with self.assertRaises(ValueError): validate_batch(batch,CFG)

if __name__=="__main__": unittest.main()
