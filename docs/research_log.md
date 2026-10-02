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
