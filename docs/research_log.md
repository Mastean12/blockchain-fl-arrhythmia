# Day 1 — Project setup

**Date:** 2026-10-02

## Completed

- Confirmed the existing GitHub checkout, clean starting status, and `origin` fetch/push remote.
- Established the approved research directory structure.
- Created a Python 3.12 virtual environment at `.venv` (Python 3.12.10).
- Configured VS Code to use `.venv` and recommended the Python and Jupyter extensions.
- Installed and pinned the initial dependencies in `requirements.txt`.
- Added the working research aim, objectives, and high-level architecture to the README.
- Created a literature-review matrix template.

## Scope and decisions

This is environment and documentation setup only. No dataset was downloaded, and no model, federated-learning, privacy, blockchain, or proposed algorithm implementation was added. The research objectives and architecture remain subject to refinement through the literature review. No results or novelty claims are asserted.

## Verification

- Import check passed for NumPy, Pandas, SciPy, Matplotlib, Seaborn, Scikit-learn, WFDB, JupyterLab, PyTorch, Torchvision, Torchaudio, and tqdm.
- `pip check` reported no broken requirements.
- JupyterLab reported version 4.6.4.
- Installed PyTorch build is CPU-only (`2.14.1+cpu`). GPU support was not set up or tested.

# Day 2 — MIT-BIH dataset exploration

**Date:** 2026-10-02

## Completed

- Read `AGENTS.md` and inspected the existing Git checkout, Day 1 setup files, and current working-tree changes before editing.
- Retrieved PhysioNet MIT-BIH Arrhythmia Database v1.0.0 into `data/raw/mitbih/` from PhysioNet's public S3 mirror. The source directory contains 706 files (109,343,870 bytes); the included SHA-256 manifest verified 704 files with no mismatches.
- Added `notebooks/01_dataset_exploration.ipynb`, executed it successfully, and saved the generated record, channel, annotation, and auxiliary-note tables plus two figures under `results/`.
- Added `docs/dataset_mitbih.md` with provenance, license/citation, header statistics, source-symbol meanings, observed counts, imbalance observations, assumptions, and unresolved preprocessing decisions.
- Added `data/raw/mitbih/` to `.gitignore` so the downloaded raw data is not added to Git.

## Observed facts from local files

- 48 records are listed; all headers report 360 Hz, two channels, and 650,000 samples per channel (650,000 × 2 signal shape; 1,805.56 seconds per record).
- 48 `.atr` files listed by `RECORDS` contain 112,647 annotation instances: 109,494 standard beat-symbol instances across 15 observed symbols and 3,153 other/event-symbol instances across 8 observed symbols.
- `N` is 68.5444% of counted beat symbols; `S` has 2 occurrences (0.0018%). These are dataset counts, not population prevalence or clinical conclusions.

## Scope and reproducibility

No model was trained. No final preprocessing, relabeling, or train/validation/test split was performed. The notebook reads original signal and annotation files through WFDB and validates source files using the PhysioNet checksum manifest; derived CSVs and figures are written only under `results/`. No commit or push was made.

# Day 3 — ECG preprocessing pipeline

**Date:** 2026-10-02

## Completed

- Reviewed `AGENTS.md`, the Day 2 dataset notes, and the dataset-exploration notebook before implementation. Day 2 had counted 15 beat symbols but intentionally did not define merged target classes.
- Added a versioned configuration and modular code for WFDB loading/checksums, filtering, beat-window segmentation, normalization, source-symbol labels, validation, and NPZ/CSV/JSON storage under `src/preprocessing/`.
- Processed a five-record representative subset (100, 104, 200, 201, 202) first: 11,198 windows, with no output failures. Then processed all 48 records into `data/processed/mitbih_v1/`.
- Executed `notebooks/02_preprocessing_validation.ipynb` successfully. It displays raw/filtered ECG, annotated normalized windows, examples for available classes, and dataset-wide metadata/count checks; it saves a processed class-distribution figure.
- Added `docs/preprocessing.md`, five unit tests, and a `.gitignore` rule for the reproducible processed dataset. Raw MIT-BIH data remains untouched.

## Recorded preprocessing decisions

- Butterworth bandpass, 0.5–40 Hz, order-4 prototype, designed as SOS and applied using forward-backward `sosfiltfilt`, at 360 Hz with odd padding.
- Beat-centered fixed windows: 72 samples before and 144 after the annotation (216 samples total); incomplete edge windows are counted and dropped.
- Per-window, per-channel z-score normalization (`ddof=0`, epsilon `1e-8`), stored as float32. Constant/near-constant windows fail rather than being silently changed.
- Identity mapping for the 15 Day 2 beat symbols. No symbols were merged. The eight known event symbols are counted and excluded from heartbeat windows. Unknown symbols cause an error.
- Records 201 and 202 share a pseudonymous grouping key because the source record directory documents the same analog tape; other group keys are record-derived and must not be interpreted as confirmed patient identifiers.

## Actual output and checks

- 48 records produced **109,460 segments**. The processed distribution is in `data/processed/mitbih_v1/class_distribution.csv` and `manifest.json`.
- The output contains **34 boundary-dropped beat annotations** and **3,153 known non-beat/event annotations skipped**, consistent with the Day 2 event count.
- Per-class segment counts: `/` 7,027; `A` 2,546; `E` 106; `F` 802; `J` 83; `L` 8,072; `N` 75,028; `Q` 33; `R` 7,255; `S` 2; `V` 7,129; `a` 150; `e` 16; `f` 982; `j` 229.
- All 48 records generated nonempty segments. Pipeline source checksums, dimensions, finite values, labels, segment lengths, annotations, normalization and label/segment alignment checks passed. Five unit tests passed; notebook execution returned no cell errors; post-run output counts and metadata alignment were checked.

## Open research decisions

The source-label identity mapping is only an initial, auditable representation, not the final classification target. Literature review and study objectives must justify which symbols the research task evaluates and how paced, fusion, escape, unclassifiable, rare and event annotations are treated. The filter band, window duration, normalization and subject-level dependency mapping also require sensitivity analysis and evaluation-design decisions before training. No model, split, augmentation, class balancing, federated learning, privacy, blockchain, or proposed algorithm was implemented. No commit or push was made.

# Day 4 — Dataset splitting and preparation

**Date:** 2026-10-04

## Preflight finding and repair

- The first split audit found Day 3's `np.full(..., dtype=str)` had silently truncated persisted record IDs and group keys to one character. Split generation stopped before producing any final split.
- Corrected string-width allocation in `src/preprocessing/segmentation.py` and added a regression test. Regenerated all 48 **derived** processed record shards from the unchanged raw MIT-BIH data. Verified 48 distinct full record IDs, 47 group keys, and the same 109,460 segment count. No raw files were changed or records omitted.

## Split method and output

- Split at the available record-group level. Seeded random group assignment (seed **42**, configurable) apportioned 47 groups by largest remainder as 33 train, 7 validation, and 7 test groups. The known same-tape pair (records 201/202) remains in one partition.
- This yields 33/8/7 records and 77,550/16,351/15,559 segments, or 70.85%/14.94%/14.21% of segments. No class stratification, balancing, or record exclusion was used.
- Saved per-record NPZ shards under `data/splits/mitbih_v1/{train,validation,test}/` plus split metadata and class-distribution CSVs.
- A complete temporary preflight over the existing processed dataset passed before final split creation. The final saved splits were reopened and revalidated. Pairwise record/group overlap and exact segment-content-plus-label duplicates across splits were absent. All arrays have expected dimensions and finite values, metadata fields are present, labels are valid, and same-seed assignment is reproducible.
- Executed `notebooks/03_dataset_splitting_validation.ipynb` without cell errors and saved `results/figures/mitbih_split_class_distribution.png`. Ten preprocessing/splitting unit tests pass.

## Observed class-support limits

- Validation has no `E`, `Q`, `R`, `S`, or `e`; test has no `E`, `J`, `R`, `S`, `e`, `f`, or `j`. Validation has 7 `F` and 3 `J`; test has 4 `Q` and 6 `a`. `S` has two segments overall, both in train.
- See `docs/dataset_splits.md` for full per-class counts and proportions. These omissions are reported, not corrected through resampling or class removal.

## Limitations and pending decisions

- `patient_groups` contains record-derived pseudonyms for most records, not verified patient identities. MIT-BIH is described as 48 records from 47 subjects, and the 201/202 shared-tape relationship is documented; complete record-to-subject mapping remains unavailable. The result guarantees no overlap by record and known group keys but cannot claim verified patient-held-out evaluation for all records.
- Review whether this random grouped split's sparse/absent evaluation classes are appropriate before Day 5 model evaluation. Also resolve the task's target label taxonomy before training.
- No model, class balancing, augmentation, federated learning, privacy, blockchain, or proposed algorithm was implemented. No commit or push was made.

# Day 5 — First centralized 1D CNN

**Date:** 2026-10-04

## Completed

- Reviewed `AGENTS.md`, Day 2 dataset exploration, Day 3 preprocessing, and Day 4 split documentation/code before model implementation.
- Confirmed train and validation segments have stored shape `(216, 2)` and are mapped to 15 distinct source-symbol labels. The model receives channels-first tensors `(batch, 2, 216)`. Training has all 15 classes; validation lacks `E`, `Q`, `R`, `S`, and `e` under the fixed group split.
- Added `src/models/cnn1d.py` (three Conv1D blocks, batch normalization, ReLU, max pooling, adaptive global average pooling, dropout, and 15-logit linear head), with 10,127 trainable parameters.
- Added a configurable training runner that opens only the explicit train and validation partitions. No test partition was opened. No reweighting, balancing, augmentation, or test evaluation was performed.
- Executed all five configured epochs from seed 42, batch size 256, Adam at learning rate 0.001, cross-entropy loss, CPU, and eight CPU threads. Training completed in about 30.5 seconds; model, history, and figure were saved under `results/`.
- Added and executed `notebooks/04_centralized_baseline.ipynb`, architecture/results documentation, and three model initialization/forward-pass tests. The three model tests passed.

## Actual run metrics

- Epoch 1: train loss 0.7918, train accuracy 0.8195; validation loss 1.2786, accuracy 0.6853, macro-F1 0.1737.
- Epoch 5: train loss 0.1595, train accuracy 0.9598; validation loss 1.8939, accuracy 0.7004, macro-F1 0.1351.
- Best checkpoint by validation loss: epoch 1. Validation macro-F1 is computed over all 15 configured labels with zero for unsupported labels; five labels are absent in validation.
- Training loss reduction alongside worsening validation loss after epoch 1 is an overfitting warning. No test metric or generalization conclusion is reported.

## Open decisions

The source-symbol identity mapping is still provisional, and validation/test support is inadequate for several rare labels under the current split. Review the target label definition, class support, and any revised split protocol before treating this as a definitive centralized baseline. The current checkpoint only represents a short exploratory run. No federated learning, differential privacy, homomorphic encryption, blockchain, or proposed algorithm was implemented. No commit or push was made.

# Day 6 — Centralized CNN test evaluation

**Date:** 2026-10-04

## Completed

- Evaluated the frozen Day 5 checkpoint on the Day 4 held-out test partition only. The reusable evaluation module loads only `data/splits/mitbih_v1/test/`; it does not open training or validation samples or update model weights.
- Generated predictions and softmax probabilities for all 15,559 test segments from seven record shards.
- Saved the labeled numerical confusion matrix, publication-quality confusion and ROC plots, machine-readable and human-readable class reports, per-class and aggregate metrics, and prediction/provenance arrays under `results/`.
- Created and successfully executed `notebooks/05_model_evaluation.ipynb`. No test-set result was used to retrain or tune the model. Raw and split datasets were not modified.

## Actual test metrics

- Accuracy 0.7688; macro precision 0.1510; macro recall (all 15 configured labels with zero for absent labels) 0.1431; macro F1 0.1399; weighted F1 0.7208.
- Macro sensitivity over the eight supported classes 0.2683; weighted sensitivity 0.7688. Macro specificity over all 15 one-vs-rest classes 0.9664; support-weighted specificity 0.7270.
- Macro OVR AUROC 0.6238 over eight classes with test positives; weighted OVR AUROC 0.9004; micro OVR AUROC 0.9624.
- Classes `A`, `F`, `L`, `Q`, and `a` had zero F1 despite positive support; `L` had support 2,001 and zero recall. Sensitivity/AUROC are undefined for `E`, `J`, `R`, `S`, `e`, `f`, and `j`, which have no test positives under the Day 4 split.

## Limitations

The macro/weighted/micro summaries differ substantially under class imbalance. Validation/test split support is inadequate or absent for several labels, and person-level identity is not fully available beyond the known 201/202 grouping. Specificity is reported with its one-vs-rest formula and must not be interpreted without sensitivity/F1. The test metrics are for this model and this split only; they do not establish population performance. No federated learning, privacy mechanism, blockchain, or proposed algorithm was implemented. No commit or push was made.

# Day 7 — Research checkpoint

**Date:** 2026-10-05

## Completed

- Audited the Day 1–6 documentation, source/configuration, saved split metadata, centralized model artifacts, Day 6 metrics, and Git history.
- Created `docs/research_checkpoint_day7.md` with the current baseline, actual dataset/split/test statistics, preprocessing and model decisions, limitations, unresolved questions, and readiness assessment.
- Revalidated all saved split shards: no record/group overlap or cross-split duplicate segment content was found; expected shapes, labels, and finite values passed validation.
- Ran all 15 existing `unittest` tests successfully and recomputed aggregate and per-class metrics from saved test predictions; all matched the Day 6 metric files.

## Major findings and limitations

The project has a reproducible centralized 1D CNN reference point, but its 15-symbol mapping is provisional. Train/validation/test contain 77,550/16,351/15,559 segments (70.85%/14.94%/14.21%); record and available group keys are disjoint, but complete patient-held-out status cannot be verified. Test accuracy is 0.7688, macro-F1 0.1399, weighted-F1 0.7208, and macro OVR AUROC 0.6238 over eight supported labels. `A`, `F`, `L`, `Q`, and `a` have zero test F1 despite support; seven labels have no test positives. Training/validation loss behavior raises an overfitting warning. These results do not establish population performance or clinical utility.

## Decision before Federated Learning

**NOT READY.** Review the label taxonomy, available patient mapping, and evaluation/split protocol before using this under-supported baseline as the comparator for FL experiments. No federated learning, privacy, blockchain, or proposed algorithm was implemented during this checkpoint. The README required no change. `pytest` is not installed; all tests were run successfully with `unittest`.
