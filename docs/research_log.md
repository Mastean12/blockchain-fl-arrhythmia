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
