# Day 9 — Centralized baseline freeze

- **Freeze date:** 2026-10-05
- **Baseline identifier:** `centralized_cnn_v1` on `mitbih_v1` / `mitbih_split_v1`
- **Status:** frozen reference baseline for later experiments. A compact specification is in [`baseline_specification.md`](baseline_specification.md).

This document records the centralized baseline as it already exists. No dataset, preprocessing, label, split, model, prediction, or metric was changed to produce it. Every number below comes from an existing Day 2–8 artifact. The integrity check (section 9) confirmed that the artifacts are unchanged since the Day 8.5 verification.

The baseline is a fixed **reference point**, not a strong or clinically validated classifier. Its purpose is to let later federated, privacy-preserving, and blockchain-coordinated experiments be compared against the same data, labels, split, model, and evaluation protocol.

## 1. Dataset

| Item | Value |
|---|---|
| Source | MIT-BIH Arrhythmia Database, PhysioNet v1.0.0 (local copy, read only) |
| Records | 48 half-hour, two-channel ambulatory ECG excerpts |
| Subject information | PhysioNet reports 47 subjects (25 men, 22 women) and states that records 201 and 202 came from the same subject. No per-record patient identifier is provided. |
| Sampling frequency | 360 Hz |
| Channels | 2 per record, in mV. Lead configuration varies: MLII + V1 (40 records), MLII + V2 (2), MLII + V5 (2), V5 + V2 (2), MLII + V4 (1), V5 + MLII (1) |
| Raw annotations | 112,647 annotations across 23 symbols: 109,494 beat annotations (15 symbols) and 3,153 non-beat/event annotations (8 symbols) |
| Processed segments | 109,460 beat-centred windows. The 34 beat annotations without a complete window at a record boundary are dropped. |
| Segment shape | `[216 samples, 2 channels]`, float32 |

**Label taxonomy.** The 15 beat symbols `N`, `L`, `R`, `V`, `/`, `A`, `f`, `F`, `j`, `a`, `E`, `J`, `Q`, `e`, `S` are kept as 15 distinct model classes through an identity mapping. No class is merged, renamed, or dropped. The 8 non-beat/event symbols (`+`, `~`, `!`, `"`, `x`, `|`, `[`, `]`) create no beat windows. Model class index order is sorted-symbol order: `/`→0, `A`→1, `E`→2, `F`→3, `J`→4, `L`→5, `N`→6, `Q`→7, `R`→8, `S`→9, `V`→10, `a`→11, `e`→12, `f`→13, `j`→14. The taxonomy audit is in [`label_taxonomy_audit.md`](label_taxonomy_audit.md).

**Subject-equivalence grouping.** Records 201 and 202 share the group key `mitdb_group_201_202`. Every other record has a record-derived key (`record_<id>`). Combined with PhysioNet's 48-record/47-subject statement, this gives **47 inferred subject-equivalence groups**. The inference assumes the published subject count is complete and correct. Explicit patient identifiers are not available for every record, so the groups are not verified patient IDs. See [`patient_mapping_audit.md`](patient_mapping_audit.md).

## 2. Preprocessing (`configs/preprocessing/mitbih_v1.json`)

1. **Loading and validation:** WFDB reads physical units (mV). The loader checks that `.hea`, `.dat`, and `.atr` files exist, that the signal is 360 Hz with 2 channels and finite values, and that annotation locations are in range. Raw files are never written.
2. **Filtering:** Butterworth band-pass, 0.5–40 Hz, order-4 prototype, designed as second-order sections and applied zero-phase with `scipy.signal.sosfiltfilt` (`padtype="odd"`). Each complete record is filtered before segmentation. The filter is non-causal.
3. **Segmentation:** one window per beat annotation, from 72 samples (0.2 s) before to 144 samples (0.4 s) after the annotated sample. That is 216 samples (0.6 s), with the annotation at index 72. Incomplete boundary windows are dropped (`drop_incomplete`) and never padded. Neighbouring windows within a record may overlap.
4. **Annotation handling:** each of the 15 beat symbols maps to itself. The 8 non-beat/event symbols are counted and skipped. Any unknown symbol raises an error.
5. **Normalization:** z-score normalization per segment and per channel over the 216 samples (`ddof=0`, epsilon `1e-8`, float32). No statistics are shared across the cohort.
6. **Metadata preservation:** each per-record NPZ shard stores `segments`, `labels`, `source_symbols`, `record_ids`, `patient_groups`, annotation sample positions, `[start, end)` offsets, sampling frequency, and channel names. `manifest.json` records software versions (Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, WFDB 4.3.1) and counts.
7. **Validation:** the pipeline checks dimensions, finite values, annotation/window offset alignment, label-to-source-symbol identity, normalized mean and standard deviation, and configured labels. Flat windows are rejected rather than altered. Identifier strings are regression-tested against truncation (the Day 4 repair).

## 3. Dataset split (`configs/preprocessing/mitbih_split_v1.json`)

- **Method:** complete groups are assigned to splits by seeded random permutation (NumPy `default_rng`). Segments are never assigned individually. The assignment is not stratified by label.
- **Random seed:** 42. Target proportions are 70/15/15, and groups are apportioned by largest remainder.
- **Sizes:**

| Split | Groups | Records | Segments | Segment share |
|---|---:|---:|---:|---:|
| Train | 33 | 33 | 77,550 | 70.85% |
| Validation | 7 | 8 | 16,351 | 14.94% |
| Test | 7 | 7 | 15,559 | 14.21% |
| **Total** | **47** | **48** | **109,460** | **100%** |

- **Validation records:** 111, 112, 114, 119, 123, 201, 202, 217. The 201/202 pair is kept together here.
- **Test records/groups:** 100, 101, 107, 113, 214, 219, 233 (`record_100`, `record_101`, `record_107`, `record_113`, `record_214`, `record_219`, `record_233`).
- **Overlap results:** `split_metadata.json` reports empty record and group overlap lists for train/validation, train/test, and validation/test. No exact duplicate of segment bytes plus label appears across splits. `validate_saved_splits` passes.
- **Interpretation:** the split uses record-derived grouping with the documented 201/202 subject equivalence included. This gives 47 inferred subject-equivalence groups, and no group appears in more than one split. Because explicit patient identifiers are not available for every record, this is **not** independently verified patient-level separation.

## 4. Centralized CNN (`src/models/cnn1d.py`, `ECG1DCNN`)

| Stage | Configuration | Output length |
|---|---|---:|
| Input | `[batch, 2, 216]` (channels first) | 216 |
| Block 1 | Conv1d 2→16, kernel 7, stride 1, padding 3, no bias → BatchNorm1d(16) → ReLU → MaxPool1d(2) | 108 |
| Block 2 | Conv1d 16→32, kernel 5, stride 1, padding 2, no bias → BatchNorm1d(32) → ReLU → MaxPool1d(2) | 54 |
| Block 3 | Conv1d 32→64, kernel 3, stride 1, padding 1, no bias → BatchNorm1d(64) → ReLU | 54 |
| Head | AdaptiveAvgPool1d(1) → Dropout(0.3) → Linear 64→15 | 15 logits |

- **Trainable parameters:** 10,127 (rechecked by loading the checkpoint).
- **Training configuration** (`configs/training/centralized_cnn_v1.json`, identical to the copy stored in the checkpoint):

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Loss | `CrossEntropyLoss`, unweighted |
| Epochs | 5 |
| Random seed | 42, with `torch.use_deterministic_algorithms(True)` |
| Training data | Shuffled each epoch with a seeded generator |
| Hardware | CPU, 8 threads, 0 DataLoader workers |
| Balancing | No class weighting, resampling, or augmentation |

- **Software:** PyTorch 2.14.1+cpu, scikit-learn 1.9.1.
- **Checkpoint used for final evaluation:** `results/models/centralized_cnn_v1.pt`. This is the epoch with the lowest validation loss: **epoch 1**, validation loss 1.2786, validation accuracy 0.6853. Validation loss rose in later epochs while training loss fell, which is an overfitting warning.

## 5. Evaluation protocol

- **Validation set:** used only for monitoring each epoch and selecting the checkpoint by minimum validation loss.
- **Held-out test set:** evaluated once with the selected checkpoint in eval mode with no gradient updates (`src/models/evaluate_cnn.py`). The evaluator opens only `data/splits/mitbih_v1/test/`.
- **No test-set tuning:** the test set was not used for training, epoch selection, architecture choice, or hyperparameter selection. `baseline_metrics.json` records `test_data_used_for_training_or_model_selection: false`.
- **Multiclass metrics:** accuracy, plus per-class precision, recall/sensitivity, and F1 from the 15×15 confusion matrix (rows are true labels, columns are predictions). In single-label multiclass evaluation, micro precision, recall, and F1 equal accuracy.
- **One-vs-rest specificity:** `TN_k / (TN_k + FP_k)` for each class. It is defined for all 15 classes, including classes with no positive test examples, for which it is 1.0 if the class is never predicted.
- **Averaging:** macro precision, recall, and F1 are unweighted means over all 15 configured classes with `zero_division=0`, so each class absent from test adds 0. A separate macro sensitivity averages only the 8 classes with test support. Weighted averages weight each class by its true test support.
- **AUROC:** one-vs-rest on softmax probabilities. Per-class AUROC is computed only for the 8 classes that have both positive and negative test examples; the other 7 are marked undefined. Macro AUROC is the mean of the 8 defined values. Weighted AUROC weights those 8 values by support. Micro AUROC flattens all sample/class decisions into a single ROC.

## 6. Final baseline performance (Day 6 artifacts, not recomputed)

Test set: 15,559 segments from 7 records. Values are from `results/metrics/baseline_metrics.json` and `results/tables/baseline_per_class_metrics.csv`, rounded to four decimals.

### Aggregate metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7688 |
| Macro precision (15 classes) | 0.1510 |
| Macro recall (15 classes) | 0.1431 |
| Macro sensitivity (8 supported classes) | 0.2683 |
| Macro F1 (15 classes) | 0.1399 |
| Weighted precision | 0.7078 |
| Weighted recall | 0.7688 |
| Weighted F1 | 0.7208 |
| Macro specificity (15 classes) | 0.9664 |
| Weighted specificity | 0.7270 |
| Macro OvR AUROC (8 defined classes) | 0.6238 |
| Weighted OvR AUROC (8 defined classes) | 0.9004 |
| Micro OvR AUROC | 0.9624 |

### Per-class metrics

"Undefined" means the class has no positive test examples. The artifact stores recall as 0 for those classes under `zero_division=0`, and that 0 enters the 15-class macro recall above.

| Class | Test support | Precision | Recall / sensitivity | Specificity | F1 | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `/` | 2,078 | 0.9971 | 0.5038 | 0.9998 | 0.6694 | 0.9977 |
| `A` | 50 | 0.0000 | 0.0000 | 0.9999 | 0.0000 | 0.7595 |
| `E` | 0 | 0.0000 | undefined | 1.0000 | 0.0000 | undefined |
| `F` | 13 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.4548 |
| `J` | 0 | 0.0000 | undefined | 1.0000 | 0.0000 | undefined |
| `L` | 2,001 | 0.0000 | 0.0000 | 0.9757 | 0.0000 | 0.6593 |
| `N` | 10,197 | 0.8240 | 0.9934 | 0.5964 | 0.9008 | 0.9333 |
| `Q` | 4 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.1529 |
| `R` | 0 | 0.0000 | undefined | 0.9924 | 0.0000 | undefined |
| `S` | 0 | 0.0000 | undefined | 1.0000 | 0.0000 | undefined |
| `V` | 1,210 | 0.4445 | 0.6488 | 0.9316 | 0.5276 | 0.8716 |
| `a` | 6 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.1611 |
| `e` | 0 | 0.0000 | undefined | 1.0000 | 0.0000 | undefined |
| `f` | 0 | 0.0000 | undefined | 1.0000 | 0.0000 | undefined |
| `j` | 0 | 0.0000 | undefined | 1.0000 | 0.0000 | undefined |

The confusion matrix is in `results/tables/baseline_confusion_matrix.csv`. Notable errors: all 2,001 `L` beats are predicted as `N` (1,369) or `V` (632); 49 of 50 `A` beats are predicted as `N`; and 57 `N` beats are predicted as `R`, a class with no test support.

## 7. Baseline limitations

1. **Severe class imbalance.** `N` makes up 75,028 of 109,460 processed segments (68.5%) and 10,197 of 15,559 test segments (65.5%). `S` has 2 segments in the whole dataset, `e` has 16, and `Q` has 33.
2. **Low macro-F1.** It is 0.1399 over 15 classes, against an accuracy of 0.7688.
3. **Zero-F1 classes with test support.** `A`, `F`, `L`, `Q`, and `a` are never correctly detected. This includes `L`, which has 2,001 test examples.
4. **Only 8 of 15 classes have positive test support.** `E`, `J`, `R`, `S`, `e`, `f`, and `j` have no test examples, so their sensitivity and AUROC cannot be estimated. Validation also lacks `E`, `Q`, `R`, `S`, and `e`. `S` occurs in only 1 group, `e` in 1, and `E` in 2, so no group-disjoint split can give these classes support in all three partitions.
5. **Very small test support for some classes.** `Q` (4), `a` (6), and `F` (13) have per-class estimates that are highly unstable, including AUROC values below 0.5.
6. **Inferred subject-equivalence grouping.** The 47 groups depend on PhysioNet's published subject count being complete and correct. If an authoritative source showed another shared-subject pair, the split would have to be revised.
7. **No complete explicit patient identifiers.** The test set is group-disjoint but is not an independently verified patient-held-out cohort.
8. **Small fixed test cohort.** The test set is 7 records from one seeded split. No cross-validation, repeated splits, or confidence intervals are available, so the variance of every metric is unknown.
9. **Overfitting warning and minimal training.** The checkpoint comes from epoch 1 of 5, and validation loss worsened afterwards. There was no class weighting, balancing, augmentation, or hyperparameter search.
10. **Provisional task definition.** The 15-symbol beat taxonomy has not been justified by literature as the final research endpoint. Rhythm and event annotations are out of scope.
11. **Lead heterogeneity.** Channel position is not always the same lead (for example, V5 + MLII and V5 + V2 records). The model treats input channels by position only.
12. **Preprocessing choices are not sensitivity-tested.** The window length, filter band, and per-window z-scoring are starting parameters. Whole-record zero-phase filtering is non-causal.
13. **Single-database scope.** All results are from MIT-BIH, a single historical cohort in which 25 of 48 records were selected to include uncommon but clinically significant arrhythmias. Its class frequencies are not population prevalence, and there is no external or multi-institution validation.

## 8. Scientific interpretation

Accuracy alone is not enough here. About 65.5% of test segments are `N`, so a classifier that always predicted `N` would reach about 0.655 accuracy while detecting no arrhythmia. The baseline's 0.7688 accuracy and 0.7208 weighted F1 come mostly from `N`, `/`, and `V`. The macro-F1 of 0.1399 and the five zero-F1 supported classes show that most beat types are not detected. Weighted metrics and micro AUROC (0.9624) are dominated by the majority class. The macro AUROC of 0.6238 over defined classes includes values below chance for `Q` and `a`. Specificity is high for rare classes partly because those classes are rarely or never predicted. For arrhythmia detection, minority-class sensitivity matters, so later experiments must report macro and per-class metrics, support, and undefined classes alongside accuracy.

The baseline shows that the pipeline can train and evaluate a 15-output beat classifier reproducibly, and that it separates several well-supported beat types in this particular split. It does **not** show clinical usefulness, readiness for clinical deployment, diagnostic validity, or generalization beyond this dataset and split.

## 9. Integrity verification (Day 9)

- SHA-256 hashes of all 111 files under `results/`, `data/splits/`, `configs/`, and `src/` match the Day 8.5 verification exactly. Key hashes are listed in [`baseline_specification.md`](baseline_specification.md).
- The checkpoint loads strictly into `ECG1DCNN` with 10,127 trainable parameters. Its stored training configuration equals `configs/training/centralized_cnn_v1.json`.
- Running the frozen checkpoint again on the 15,559 test segments, without saving anything, reproduces the stored `baseline_test_predictions.npz` labels exactly (100% agreement, same label order). Stored predictions equal the argmax of the stored probabilities.
- Accuracy and macro-F1 computed from the stored predictions equal `baseline_metrics.json`, and the confusion matrix equals `baseline_confusion_matrix.csv`.
- `split_metadata.json` reports seed 42, 47 groups, 48 records, 109,460 segments, the split sizes above, and empty overlap lists. `validate_saved_splits` passes.
- The processed manifest reports 48 records, 109,460 segments, shape `[216, 2]`, and 360 Hz. `preprocessing_config.json` is byte-identical to `configs/preprocessing/mitbih_v1.json`.
- All 15 existing `unittest` tests pass.

## 10. Rules for future experiments

Every later experiment (federated, privacy-preserving, blockchain-coordinated, or proposed method) compares against this baseline unless it deliberately changes **one** documented factor and reports that change. Changing the taxonomy, preprocessing, split, model, or training configuration creates a new versioned baseline. It must not overwrite `centralized_cnn_v1`.
