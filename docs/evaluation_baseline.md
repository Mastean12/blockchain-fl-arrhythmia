# Day 6 — Centralized CNN test evaluation

## Evaluation protocol

The final Day 5 checkpoint (`results/models/centralized_cnn_v1.pt`) was loaded in evaluation mode with no optimizer or gradient updates. The evaluator loads only NPZ shards from `data/splits/mitbih_v1/test/`; it does not load train or validation shards. The test set was not used for training, epoch selection, or hyperparameter tuning. The evaluated checkpoint had already been selected by minimum validation loss at epoch 1. Predictions were generated for all 15,559 test segments from seven records.

The target order is preserved from the checkpoint: `/`, `A`, `E`, `F`, `J`, `L`, `N`, `Q`, `R`, `S`, `V`, `a`, `e`, `f`, `j`. Softmax probabilities from the saved logits are used for ROC/AUROC. All reported values are generated from this test evaluation; no class was dropped from the reports.

## Metric definitions and averaging

For class `k` in a one-vs-rest confusion matrix:

- Sensitivity/recall: `TP_k / (TP_k + FN_k)`.
- Specificity: `TN_k / (TN_k + FP_k)`.
- Precision: `TP_k / (TP_k + FP_k)`.
- F1: `2 × precision_k × recall_k / (precision_k + recall_k)` (defined as zero when the denominator is zero for the standard report).

The confusion matrix rows are true labels and columns are predicted labels. Accuracy is correct predictions divided by all test segments.

Macro precision, recall, and F1 are unweighted averages over the full configured 15-label set; scikit-learn uses `zero_division=0` for unsupported/undefined classification-report entries. Consequently, macro recall/F1 include a zero entry for each label with no test support. The separate `macro_sensitivity_supported_classes` averages only classes with at least one positive test instance. Weighted precision, recall/sensitivity, and F1 weight each class by its true test support. In single-label multiclass evaluation, micro precision, recall, and F1 equal accuracy.

Specificity is mathematically defined for every label in this test set because each class has many one-vs-rest negatives, including labels with no positive examples. `macro_specificity_all_classes` averages all 15 specificities; `weighted_specificity` weights by true-class support. A class with zero support and no predictions has specificity 1, so specificity must be read alongside sensitivity and F1 and can look high when class detection is poor.

## ROC/AUROC method

Each one-vs-rest ROC compares the class's softmax probability to the binary truth indicator. Per-class ROC and AUROC require both positive and negative test examples. Only 8 of 15 labels have positive test support; AUROC is marked undefined for the other 7 rather than forcing a value.

- **Macro OVR AUROC:** arithmetic mean of the 8 defined per-class OVR AUROCs (not averaged over absent classes).
- **Weighted OVR AUROC:** mean of those same per-class AUROCs weighted by each class's test support.
- **Micro OVR AUROC:** flatten the one-hot truth and probability matrices over all sample/class decisions and compute one ROC/AUC.

Macro and micro AUROC summarize different aggregations. The ROC figure displays each defined per-class curve; the CSV records per-class and aggregate AUROCs.

## Observed results

| Metric | Value |
|---|---:|
| Test segments | 15,559 |
| Accuracy | 0.7688 |
| Macro precision (15 labels, zero division=0) | 0.1510 |
| Macro recall (15 labels, zero division=0) | 0.1431 |
| Macro sensitivity (supported labels only) | 0.2683 |
| Macro F1 (15 labels, zero division=0) | 0.1399 |
| Weighted precision | 0.7078 |
| Weighted recall/sensitivity | 0.7688 |
| Weighted F1 | 0.7208 |
| Macro specificity (15 labels) | 0.9664 |
| Support-weighted specificity | 0.7270 |
| Macro OVR AUROC (8 supported classes) | 0.6238 |
| Weighted OVR AUROC | 0.9004 |
| Micro OVR AUROC | 0.9624 |

The overall accuracy, weighted F1, and micro AUROC should not obscure weak class-level detection. The model has zero F1 for `A`, `F`, `L`, `Q`, and `a` despite positive test support; it detected no examples from `A`, `F`, `L`, `Q`, or `a`. The `L` class has 2,001 test examples and zero recall. `E`, `J`, `R`, `S`, `e`, `f`, and `j` have no test positives under the Day 4 split; their sensitivity and AUROC are undefined. See the class-level CSV for each precision, recall, sensitivity, specificity, F1, support, and AUC.

The test class support is strongly imbalanced and the validation-derived group split omitted several classes from test. These results are specific to the saved model and this split; they do not establish performance on a population.

**Test cohort and separation caveat (reconciled after the Day 8 audit):** the test set is 7 complete records (100, 101, 107, 113, 214, 219, 233), forming 7 of the 47 inferred subject-equivalence groups. No record or group in test also appears in train or validation, and the documented same-subject pair 201/202 is entirely in validation. The groups are record-derived, with the documented 201/202 subject equivalence incorporated; the 47-group structure is inferred from PhysioNet's 48-record/47-subject count. Because explicit patient identifiers are not available for all records, the test set should not be described as an independently verified patient-held-out cohort. See [`dataset_splits.md`](dataset_splits.md) and [`patient_mapping_audit.md`](patient_mapping_audit.md).

## Saved outputs

- `results/metrics/baseline_metrics.json` — aggregate metrics and test provenance.
- `results/metrics/baseline_classification_report.csv` — per-class metrics, including sensitivity, specificity, support, and AUROC availability.
- `results/metrics/baseline_classification_report.txt` — human-readable precision/recall/F1/support report and undefined-metric notes.
- `results/metrics/baseline_roc_metrics.csv` — per-class and aggregate ROC/AUROC results.
- `results/metrics/baseline_test_predictions.npz` — test truth, predicted classes, probabilities, and annotation provenance (no ECG signal arrays).
- `results/tables/baseline_confusion_matrix.csv` — labeled numerical confusion matrix.
- `results/tables/baseline_per_class_metrics.csv` — per-class metric table.
- `results/figures/baseline_confusion_matrix.png` — labeled confusion-matrix visualization.
- `results/figures/baseline_roc_curve.png` — one-vs-rest ROC curves for classes with test positives.
- `notebooks/05_model_evaluation.ipynb` — executable evaluation walkthrough.

No retraining or hyperparameter selection was performed after inspecting test results. No changes were made to the raw or split data.
