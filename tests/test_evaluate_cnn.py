import unittest

import numpy as np

from src.models.evaluate_cnn import calculate_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_per_class_metrics_and_undefined_roc_for_absent_class(self):
        y_true = np.asarray([0, 0, 1, 1])
        y_pred = np.asarray([0, 1, 1, 2])
        probabilities = np.asarray([
            [0.80, 0.15, 0.05],
            [0.40, 0.50, 0.10],
            [0.10, 0.70, 0.20],
            [0.20, 0.40, 0.40],
        ])
        result = calculate_metrics(y_true, y_pred, probabilities, ["A", "B", "C"])
        by_label = {row["label"]: row for row in result["per_class"]}
        self.assertEqual(result["confusion_matrix"].tolist(), [[1, 1, 0], [0, 1, 1], [0, 0, 0]])
        self.assertEqual(by_label["A"]["recall"], 0.5)
        self.assertEqual(by_label["A"]["sensitivity"], 0.5)
        self.assertAlmostEqual(by_label["A"]["specificity"], 1.0)
        self.assertEqual(by_label["C"]["sensitivity"], None)
        self.assertEqual(by_label["C"]["auroc_ovr"], None)
        self.assertTrue(result["aggregate"]["accuracy"] < 1.0)
        self.assertIsNotNone(result["aggregate"]["macro_auroc_ovr_defined_classes"])

    def test_roc_auc_requires_probability_matrix_shape(self):
        with self.assertRaises((ValueError, IndexError)):
            calculate_metrics(np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0.8, 0.2]), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
