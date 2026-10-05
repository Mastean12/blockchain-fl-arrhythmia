# Day 7 — Research checkpoint

**Checkpoint date:** 2026-10-05
**Project:** *A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection*

## 1. Executive Summary

Days 1–6 established a Python research environment, explored the MIT-BIH Arrhythmia Database, built a reproducible ECG beat preprocessing pipeline and record-group split, trained a centralized 1D CNN, and evaluated the frozen checkpoint on the held-out test partition. No federated learning, privacy mechanism, blockchain, or proposed algorithm has been implemented.

The project is at the **centralized-baseline audit and experimental-design checkpoint**. The baseline artifacts are reproducible and the test evaluation is complete, but this is an exploratory baseline rather than a sufficiently supported final comparator: the 15-symbol target is provisional, the random group split leaves multiple labels unsupported in validation/test, person identities are incomplete, and minority-class test performance is poor. **Decision: NOT READY to proceed to Federated Learning experiments** until the label taxonomy, subject/record separation, and evaluation protocol are reviewed and made adequate. This does not invalidate the completed baseline; it limits what can be inferred from it.

## 2. What Worked

- **Environment/setup:** Python 3.12.10 virtual environment, requested scientific dependencies, JupyterLab, and VS Code configuration were established. Day 1 import checks and `pip check` were recorded as passing; PyTorch was CPU-only.
- **Dataset acquisition/loading:** PhysioNet MIT-BIH v1.0.0 was downloaded to the ignored raw-data directory. The Day 2 notebook read the 48 official records through WFDB and verified all 704 entries in the supplied checksum manifest without mismatches.
- **ECG exploration:** Record/header, channel, annotation, and event-symbol counts and figures were generated and documented. The analysis distinguishes source counts from population prevalence.
- **Preprocessing:** A configured modular pipeline processed all 48 records. The five-record preflight and full processing completed; output shape, finite-value, annotation/label alignment, and provenance checks were recorded as passing. Raw input was preserved.
- **Dataset splitting:** A seeded (42), unstratified group assignment created mutually exclusive train/validation/test partitions. Saved-shard validation passed for available record/group overlap, duplicate content, dimensions, labels, finite values, metadata, and repeatability.
- **1D CNN/training:** The configurable PyTorch CNN initialized and passed forward-shape tests. Its five-epoch CPU training run completed, saving the checkpoint selected by minimum validation loss (epoch 1) and history.
- **Evaluation:** The frozen checkpoint was evaluated on the test partition only. Reports, predictions/provenance, confusion matrix, and ROC artifacts were saved. The Day 6 notebook was reported to execute without errors; no retraining or test-driven selection occurred.
- **Reproducibility/tests:** All 15 repository `unittest` cases passed during this checkpoint. The saved split validator completed successfully. Recomputed test accuracy, macro-F1, and weighted-F1 from the saved prediction artifact exactly matched `baseline_metrics.json`.

## 3. What Failed or Needs Improvement

| Category | Actual issue or status |
|---|---|
| Failed approach / data handling | Day 4 preflight caught that Day 3 identifier arrays had been stored as one-character Unicode, truncating record/group IDs. Split generation was stopped; the code was fixed, tested, and all 48 derived shards regenerated from unchanged raw data before splitting. |
| Training behavior | Training loss fell from 0.7918 to 0.1595, while validation loss rose from 1.2786 at epoch 1 to 1.8939 at epoch 5. This is an overfitting warning in the short run. Validation metrics also vary by epoch. |
| Test performance | Test accuracy is 0.7688, but macro-F1 is 0.1399. `A`, `F`, `L`, `Q`, and `a` have zero F1 despite positive test support; `L` has 2,001 examples and zero recall. |
| Split support | The non-stratified split has no test examples for seven labels and no validation examples for five labels. Some supported classes have very small test counts (`F`: 13, `Q`: 4, `a`: 6). These class estimates are consequently incomplete or unstable. |
| Patient identity | Available metadata guarantees record and known-group disjointness; it does not provide a complete record-to-subject mapping. A fully patient-held-out evaluation cannot be claimed. |
| Preprocessing | The documented filter, fixed window, and per-window normalization are implemented consistently, but their sensitivity and task-specific suitability have not been evaluated. Record channel configurations vary. |
| Dataset/labels | The 15-way identity mapping preserves source symbols but is still provisional; very rare symbols include `S` (2 processed examples), `e` (16), and `Q` (33). The dataset's selected records and annotation distribution do not establish population prevalence. |
| Implementation/runtime | Training and the recorded PyTorch environment were CPU-only. `pytest` is not installed in the environment; the existing suite is `unittest`-compatible and all 15 tests were run successfully with `python -m unittest`. No other failure or warning is recorded in the Day 1–6 logs. |

No undocumented training crash, raw-data corruption, fabricated metric, or failed Day 6 evaluation is reported in the existing record.

## 4. Dataset Statistics

### Source and signal

| Statistic | Recorded value |
|---|---:|
| Database | MIT-BIH Arrhythmia Database, PhysioNet v1.0.0 |
| Records | 48 |
| Subjects | 47, as described by PhysioNet; project metadata does not map all records to subjects |
| Sampling frequency | 360 Hz for all records |
| Channels | 2 per record; six ordered channel-name configurations |
| Samples / record / channel | 650,000 |
| Signal shape / record | 650,000 × 2 |
| Header-derived duration | 1,805.56 seconds (about 30.09 minutes) per record |
| Raw annotations | 112,647 instances: 109,494 beat symbols and 3,153 non-beat/event annotations |
| Processed beat segments | 109,460 from 48 records |
| Processed segment shape | 216 samples × 2 channels |
| Configured classification labels | 15 distinct original beat symbols |

### Processed label distribution

Labels retain their original MIT-BIH annotation symbols; the descriptions below are source annotation names, not a merged target taxonomy or new clinical interpretation.

| Symbol | Source annotation name | Segments |
|---|---|---:|
| `/` | Paced beat | 7,027 |
| `A` | Atrial premature beat | 2,546 |
| `E` | Ventricular escape beat | 106 |
| `F` | Fusion of ventricular and normal beat | 802 |
| `J` | Nodal (junctional) premature beat | 83 |
| `L` | Left bundle branch block beat | 8,072 |
| `N` | Normal beat | 75,028 |
| `Q` | Unclassifiable beat | 33 |
| `R` | Right bundle branch block beat | 7,255 |
| `S` | Supraventricular premature or ectopic beat | 2 |
| `V` | Premature ventricular contraction | 7,129 |
| `a` | Aberrated atrial premature beat | 150 |
| `e` | Atrial escape beat | 16 |
| `f` | Fusion of paced and normal beat | 982 |
| `j` | Nodal (junctional) escape beat | 229 |
| **Total** | **15 source symbols** | **109,460** |

| Symbol | Segments | Symbol | Segments | Symbol | Segments |
|---|---:|---|---:|---|---:|
| `/` | 7,027 | `A` | 2,546 | `E` | 106 |
| `F` | 802 | `J` | 83 | `L` | 8,072 |
| `N` | 75,028 | `Q` | 33 | `R` | 7,255 |
| `S` | 2 | `V` | 7,129 | `a` | 150 |
| `e` | 16 | `f` | 982 | `j` | 229 |
| **Total** | **109,460** | | | | |

### Split sizes and label counts

The separation unit is the available group key: 47 keys across 48 records, with records 201/202 together. Train contains 33 keys/records; validation contains 7 keys/8 records; test contains 7 keys/records. Counts and proportions are based on segments.

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| `/` | 3,407 | 1,542 | 2,078 |
| `A` | 2,418 | 78 | 50 |
| `E` | 106 | 0 | 0 |
| `F` | 782 | 7 | 13 |
| `J` | 80 | 3 | 0 |
| `L` | 3,948 | 2,123 | 2,001 |
| `N` | 53,489 | 11,342 | 10,197 |
| `Q` | 29 | 0 | 4 |
| `R` | 7,255 | 0 | 0 |
| `S` | 2 | 0 | 0 |
| `V` | 5,049 | 870 | 1,210 |
| `a` | 28 | 116 | 6 |
| `e` | 16 | 0 | 0 |
| `f` | 722 | 260 | 0 |
| `j` | 219 | 10 | 0 |
| **Total** | **77,550** | **16,351** | **15,559** |
| **Segment share** | **70.85%** | **14.94%** | **14.21%** |
| **Records** | **33** | **8** | **7** |

The saved split metadata reports no record/group overlap between any pair of partitions. The validator also checks for exact segment-content-plus-label duplicates across partitions; checkpoint validation passed. This is not proof of complete patient independence: record-derived keys are not verified patient IDs. Test lacks `E`, `J`, `R`, `S`, `e`, `f`, and `j`; validation lacks `E`, `Q`, `R`, `S`, and `e`.

## 5. Preprocessing Decisions

| Stage | Implemented decision and rationale recorded in project docs |
|---|---|
| Sampling frequency | Require 360 Hz as reported by all MIT-BIH headers; no resampling is performed. Fixed sample-based filter/window parameters are therefore tied to this dataset rate. |
| Filtering | Fourth-order Butterworth bandpass prototype, 0.5–40 Hz, designed as second-order sections and applied using forward-backward `sosfiltfilt` with odd padding. The recorded rationale is to band-limit ECG while avoiding phase shift from a one-direction pass. |
| Segmentation | Annotation-centered windows use 72 samples before and 144 after the annotation (216 samples total, 0.6 s at 360 Hz). Incomplete boundary windows are dropped and counted; 34 beat annotations were dropped at record edges. |
| Annotation handling | Fifteen observed WFDB beat symbols map to themselves. No beat classes are merged. Eight known event/non-beat symbols (including rhythm and signal-quality markers) are counted and excluded from heartbeat windows; unknown symbols fail validation. The 3,153 known non-beat/event annotations were skipped. |
| Normalization | Per-segment, per-channel z-score, population standard deviation (`ddof=0`), epsilon `1e-8`, stored as `float32`. Constant/near-constant windows are rejected rather than silently altered. This local normalization is documented as the selected reproducible method; it has not been compared against alternatives. |
| Metadata | Segment shards retain source record/group, symbols, annotation sample, start/end offsets, sampling rate, and channel names. |
| Invalid/missing data | Source checksum, record dimensions, finite values, annotation positions/counts, segment length, label validity/alignment, and normalization checks are enforced. All 48 records produced nonempty segments; no record exclusions are documented. Raw files remain untouched. |

The complete parameters/configuration are recorded in `configs/preprocessing/mitbih_v1.json` and `docs/preprocessing.md`. No preprocessing parameter is changed by this checkpoint.

## 6. Dataset Split Decisions

- **Targets:** 70% / 15% / 15% of group keys for train/validation/test; actual segment shares are 70.85% / 14.94% / 14.21%.
- **Method:** NumPy seeded random assignment of whole available group keys, seed **42**; labels are not used to stratify, rebalance, or remove samples.
- **Leakage prevention:** complete record shards are assigned once; records 201 and 202 share the documented same-tape group. Saved metadata and `validate_saved_splits` report no record or group overlap, no duplicate segment-content-plus-label samples across splits, and reproducibility at the same seed.
- **Limitations:** only one known shared record group is patient-linked; remaining identifiers are record-derived pseudonyms. No verified full subject-held-out guarantee is possible. Random allocation gives sparse/absent minority-label support in validation and test, so some metrics cannot be estimated on those splits.

## 7. Centralized Baseline Model

- **Model:** `ECG1DCNN`, Conv1D blocks 2?16 (kernel 7), 16?32 (kernel 5), and 32?64 (kernel 3), each with batch normalization and ReLU; max pooling by 2 after the first two blocks; adaptive average pooling; dropout 0.3; linear 64?15 output head.
- **Input:** `(batch, 2 channels, 216 samples)`; stored segments are `(216, 2)`.
- **Classes:** 15 source symbols in the order `/`, `A`, `E`, `F`, `J`, `L`, `N`, `Q`, `R`, `S`, `V`, `a`, `e`, `f`, `j`.
- **Trainable parameters:** 10,127.
- **Training:** Adam, learning rate 0.001, cross-entropy loss, batch size 256, five epochs, seed 42, CPU with eight threads, zero data-loader workers; training samples shuffled. No class weights, augmentation, or balancing.
- **Checkpoint selection:** minimum validation loss, epoch 1. Test data was not loaded for training or selection.

Architecture details and the run record are in `docs/centralized_baseline.md`; settings are in `configs/training/centralized_cnn_v1.json`.

## 8. Baseline Performance

All aggregate figures below come from `results/metrics/baseline_metrics.json`. Per-class figures are from the saved Day 6 classification report; sensitivity is recall. Specificity and ROC follow the one-vs-rest definitions in `docs/evaluation_baseline.md`. Values are shown to four decimals (the underlying files retain full precision).

| Test metric | Value |
|---|---:|
| Test segments | 15,559 |
| Accuracy | 0.7688 |
| Macro precision (15 labels, zero for undefined) | 0.1510 |
| Macro recall (15 labels, zero for no support) | 0.1431 |
| Macro F1 (15 labels) | 0.1399 |
| Weighted precision | 0.7078 |
| Weighted recall/sensitivity | 0.7688 |
| Weighted F1 | 0.7208 |
| Macro specificity (15 one-vs-rest classes) | 0.9664 |
| Support-weighted specificity | 0.7270 |
| Macro OVR AUROC (8 classes with positives) | 0.6238 |
| Weighted OVR AUROC (8 defined classes) | 0.9004 |
| Micro OVR AUROC | 0.9624 |

### Per-class test results

For labels with zero test support, sensitivity and AUROC are undefined (`—`). Their classification-report precision/recall/F1 use the documented `zero_division=0` convention. Specificity is defined one-vs-rest and can be high even when a class is never detected.

| Label | Support | Precision | Recall | Sensitivity | Specificity | F1 | OVR AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/` | 2,078 | 0.9971 | 0.5038 | 0.5038 | 0.9998 | 0.6694 | 0.9977 |
| `A` | 50 | 0.0000 | 0.0000 | 0.0000 | 0.9999 | 0.0000 | 0.7595 |
| `E` | 0 | 0.0000 | 0.0000 | — | 1.0000 | 0.0000 | — |
| `F` | 13 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.4548 |
| `J` | 0 | 0.0000 | 0.0000 | — | 1.0000 | 0.0000 | — |
| `L` | 2,001 | 0.0000 | 0.0000 | 0.0000 | 0.9757 | 0.0000 | 0.6593 |
| `N` | 10,197 | 0.8240 | 0.9934 | 0.9934 | 0.5964 | 0.9008 | 0.9333 |
| `Q` | 4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.1529 |
| `R` | 0 | 0.0000 | 0.0000 | — | 0.9924 | 0.0000 | — |
| `S` | 0 | 0.0000 | 0.0000 | — | 1.0000 | 0.0000 | — |
| `V` | 1,210 | 0.4445 | 0.6488 | 0.6488 | 0.9316 | 0.5276 | 0.8716 |
| `a` | 6 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.1611 |
| `e` | 0 | 0.0000 | 0.0000 | — | 1.0000 | 0.0000 | — |
| `f` | 0 | 0.0000 | 0.0000 | — | 1.0000 | 0.0000 | — |
| `j` | 0 | 0.0000 | 0.0000 | — | 1.0000 | 0.0000 | — |

The strongest F1 values are `N` (0.9008), `/` (0.6694), and `V` (0.5276). `A`, `F`, `L`, `Q`, and `a` have positive support but zero precision/recall/F1; `L` is particularly important because it has 2,001 test examples. Accuracy is potentially misleading: the test set is dominated by `N` (10,197/15,559), and weighted F1 is much higher than macro F1. Seven classes have no positive test support, preventing test sensitivity/AUROC estimation for them. These are results for this split/model, not population estimates.

## 9. Interpretation

The run shows that the pipeline and model can optimize a 15-output ECG beat classifier and discriminate several well-supported symbols in this particular held-out record split, especially `N`, `/`, and `V`. It does **not** establish reliable performance across all source symbols: five supported test labels receive no correct predictions, and seven labels cannot be evaluated for sensitivity at all. The gap between training and validation behavior, strong class imbalance, absent split classes, and incomplete patient identifiers constrain interpretation.

Plausible contributors to poor minority-label results include the small training support for rare symbols, the non-stratified group assignment, and a short unweighted training procedure; these are hypotheses, not established causes. No clinical usefulness, deployment readiness, privacy guarantee, or generalized patient-level performance is demonstrated.

## 10. Unresolved Research Questions

These questions have no answers in the current results and must be investigated before making design claims:

1. What literature-supported target taxonomy and beat inclusion/exclusion policy best match the research question while retaining auditable source labels?
2. Can verified subject identifiers or an authoritative record-to-subject mapping be obtained, and what split strategy then ensures subject independence?
3. How should non-IID client partitions be defined without conflating record-specific acquisition effects with institution-level heterogeneity?
4. How does federated performance compare with this centralized baseline and with local-only training under a fair, fixed evaluation protocol?
5. Which aggregation baselines are necessary, and what limitation would any candidate aggregation/coordination method address?
6. What are the communication volume and round/convergence costs under realistic client participation?
7. What threat model and privacy objective should govern privacy mechanisms? What utility, privacy budget, and computational/communication cost result from differential privacy?
8. Is homomorphic encryption technically and operationally justified for the chosen aggregation operation, and what latency, memory, and communication overhead does it add?
9. Which permissioned blockchain role is necessary for update integrity, verification, coordination, or auditability, and what measurable overhead does it introduce compared with a non-blockchain coordinator?
10. Which poisoning or malicious-client behaviors are relevant, how will attacks and defenses be specified, and how will robustness be measured?
11. Does any proposed method yield a measurable improvement over appropriate baselines without unacceptable predictive, privacy, compute, communication, or blockchain costs?
12. How sensitive are conclusions to filtering/window/normalization choices, model selection, class support, and random seeds?

## 11. Current Research Baseline

The current reference point is the **15-class, source-symbol identity MIT-BIH beat classifier** using 360 Hz two-channel recordings, the documented Butterworth 0.5–40 Hz filter, annotation-centered 216 × 2 windows, per-window/per-channel z-score normalization, and seed-42 unstratified record-group split (77,550 train / 16,351 validation / 15,559 test). It is the 10,127-parameter `ECG1DCNN` trained for five epochs with Adam (0.001), batch size 256, and cross-entropy, with the epoch-1 minimum-validation-loss checkpoint. Its fixed Day 6 test metrics are accuracy 0.7688, macro-F1 0.1399, weighted-F1 0.7208, and macro OVR AUROC 0.6238 across eight supported classes. Any future comparison must disclose this baseline's provisional label mapping, class-support gaps, and incomplete subject identity; a revised protocol must be versioned and must not overwrite these results.

## 12. Readiness for Federated Learning

# NOT READY

Evidence: the target label taxonomy remains provisional; the split has no test positives for seven labels and no validation positives for five; full subject-level separation cannot be demonstrated; validation behavior warns of overfitting; and important supported classes have zero test F1. Starting FL now would add client-partition and optimization effects before the centralized comparator and evaluation unit are adequately settled. First review the taxonomy and subject mapping, define an evaluation plan with adequate support, and establish a defensible centralized reference under that agreed plan. No Day 7 change modifies the existing baseline or starts FL implementation.

## 13. Day 7 Research Log

Day 7's research-log entry is recorded in [`research_log.md`](research_log.md), including checkpoint creation, metric and split validation, the main limitations, and the NOT READY decision.

## 14. README Status

No README change was necessary. It already describes the project aim and high-level architecture; this checkpoint records the detailed status and does not change the research scope.

## 15. Tests and Validation

- Ran `python -m unittest discover -s tests -v`: **15 tests passed** (preprocessing, splitting, CNN, and evaluator tests).
- Ran `validate_saved_splits` on `data/splits/mitbih_v1/`: completed successfully. It reopened all 48 split shards and found 77,550 / 16,351 / 15,559 segments; its record/group disjointness, exact duplicate, schema, finite-value, and label checks raised no errors.
- Recomputed accuracy, macro-F1, and weighted-F1 from `baseline_test_predictions.npz`; each exactly matched `baseline_metrics.json` at stored precision. Probability rows were finite and summed to one within tolerance; dimensions were `(15,559,)` for truth/prediction and `(15,559, 15)` for probabilities.
- Checked that referenced configs, docs, checkpoint, reports, plots, prediction file, notebooks, source modules, and split metadata exist in the checkout. Raw MIT-BIH files and ignored derived shards are locally present; raw/processed/split signals are not tracked by Git according to `.gitignore`.
- Confirmed from saved split metadata that all three pairwise record/group-overlap lists are empty. This confirms record/available-key isolation, not full patient isolation.
- `pytest` itself is unavailable (`No module named pytest`); the suite was run through `unittest` instead. Existing Day 2–6 logs report successful notebook executions; notebooks were not re-executed during Day 7 because their execution is not needed for these checkpoint integrity checks.

## 16. Git Commit

Day 7 checkpoint and research-log update are committed together as one local commit (`research: document baseline checkpoint`). No push was performed. The commit includes only `docs/research_checkpoint_day7.md` and `docs/research_log.md`; existing unrelated working-tree items were left untouched.
