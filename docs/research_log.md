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


# Day 8 — Dataset, label taxonomy and leakage audit

**Date:** 2026-10-05

## Completed

- Recounted all local MIT-BIH `.atr` annotation symbols with WFDB and confirmed the Day 2 table: 112,647 annotations across 23 symbols, comprising 109,494 beat symbols and 3,153 non-beat/event symbols.
- Traced each of the 15 configured beat symbols through preprocessing, saved segment labels/source symbols, and sorted CNN class indices. The mapping is identity; eight non-beat/event symbols are explicitly excluded from beat windows; 34 incomplete edge windows are dropped by the documented boundary policy. No beat class is merged or silently discarded.
- Created `docs/label_taxonomy_audit.md`, `docs/patient_mapping_audit.md`, and executed `notebooks/06_label_and_leakage_audit.ipynb`. The notebook produced per-symbol, per-class, split, test-support, and group-support tables plus two new distribution figures.
- Rechecked all 15 output classes, class totals, train/validation/test counts, source-label alignment, split group/record disjointness, and existing split integrity.
- Reviewed and corrected `docs/dataset_splits.md` to incorporate the subject-equivalence evidence from the official local dataset documentation.

## Findings and decisions

The current model has 15 output classes, not a merged eight-class taxonomy. Eight original labels have positive support in the test split; seven have no positive test instances. `N` has 75,028 processed segments, versus `S` 2, `e` 16, and `Q` 33. `S` appears in one group, `e` in one, and `E` in two; three-way class coverage cannot be created for those labels without breaking group separation. In the current test results, accuracy is 0.7688 and macro-F1 is 0.1399; `A`, `F`, `L`, `Q`, and `a` have zero F1 despite positive test support. No class merge or retraining was made to improve scores.

The label map remains a transparent exploratory source-symbol baseline, but its suitability as the final research task and the exclusion of rhythm/event symbols need a literature- and question-based decision. Preserve the existing baseline; do not revise its mapping or retrain until the target definition and an adequately supported evaluation protocol are agreed.

The local PhysioNet introduction reports 48 records from 47 subjects and identifies records 201/202 as the same subject. That pair accounts for the one-record excess; thus the remaining 46 records are singleton subject-equivalence groups, assuming the published cohort count is complete. The current 47-group split keeps 201/202 together and has no pairwise record/group overlap or duplicate segment-plus-label samples. This supports separation by 47 inferred subject-equivalence groups. Because explicit patient identifiers are not available for all records, it is not independently verified patient-level separation. Test class coverage remains limited even though no known cross-split subject-equivalence leakage was found.

**Documentation reconciliation (Day 8.5):** before freezing the baseline, `dataset_splits.md`, `evaluation_baseline.md`, `patient_mapping_audit.md`, and `label_taxonomy_audit.md` were reconciled. They now consistently separate record-level grouping, inferred subject-equivalence grouping, and verified patient identifiers (which are not available). Only documentation changed; no data, split, model, or metric was modified.

## Validation and issues

- All existing 15 `unittest` tests passed; `validate_saved_splits` completed successfully; notebook assertions confirmed all raw symbols are categorized and split distributions reproduce the saved metadata.
- Day 6 evaluation artifact SHA-256 hashes were captured before the audit and compared afterward; the model, test predictions, confusion matrix, reports, ROC files, and figures remain unchanged.
- The first notebook execution found an audit-code error that summed `Counter` keys rather than their values; the assertion was corrected and the notebook then executed successfully. Jupyter emitted a Windows Proactor/ZeroMQ fallback runtime warning and an unencrypted TCP kernel warning during local execution; no notebook cell error remained.
- No raw dataset, processed split, model, taxonomy, or Day 6 artifact was modified. No commit or push was made.

## Unresolved questions

- Which literature-supported task taxonomy best matches arrhythmia detection, and should rhythm/event symbols or blocked atrial events be within scope?
- Should evaluation retain all 15 labels and explicitly report undefined metrics for unsupported test labels, or define a task-specific subset before another model run?
- What evaluation claims are supportable for `S`, `e`, and `E`, given their occurrence in only one, one, and two subject groups respectively?
- After the target taxonomy and evaluation objective are fixed, does the split need a new version, and what class coverage is feasible without subject leakage?
- How much of the weak class performance reflects the mapping/task definition, data support, model behavior, or split composition? No causal conclusion is established by this audit.


# Day 9 — Centralized baseline freeze

**Date:** 2026-10-05

## Completed

- Froze the centralized baseline `centralized_cnn_v1` and documented it in `docs/baseline_freeze.md` (full record) and `docs/baseline_specification.md` (compact specification with artifact paths and SHA-256 hashes).
- Verified the freeze against the current artifacts. All 111 hashed files under `results/`, `data/splits/`, `configs/`, and `src/` match the Day 8.5 verification. The checkpoint loads strictly with 10,127 parameters. Running it again on the test set without saving anything reproduced the stored test predictions exactly. The stored predictions reproduce the saved accuracy, macro-F1, and confusion matrix.
- Updated an outdated status sentence in `README.md`.

## Baseline definition

MIT-BIH v1.0.0, 48 records and 109,460 beat windows of `[216, 2]` samples. Processing is a 0.5–40 Hz zero-phase Butterworth band-pass, windows from 0.2 s before to 0.4 s after each beat, and per-segment per-channel z-scoring. The 15 beat symbols map to themselves (no merges); 8 event symbols are excluded. The split is `mitbih_split_v1` (seed 42) over 47 inferred subject-equivalence groups with 201/202 grouped: 77,550 train, 16,351 validation, and 15,559 test segments (test records 100, 101, 107, 113, 214, 219, 233). No group appears in more than one split. The model is a 3-block Conv1D CNN with 10,127 parameters, trained with Adam (learning rate 0.001), batch size 256, unweighted cross-entropy, 5 epochs, and seed 42. The checkpoint is from epoch 1, the epoch with minimum validation loss. Test results: accuracy 0.7688, macro-F1 0.1399, weighted-F1 0.7208, macro OvR AUROC 0.6238 over 8 defined classes.

## Important limitations

Severe imbalance (`N` is 65.5% of the test set). `A`, `F`, `L`, `Q`, and `a` have zero F1 despite test support, and only 8 of 15 classes have any test support. The 47 groups are inferred from the published subject count; explicit patient IDs are not available for every record, so this is not independently verified patient-level separation. There is one small fixed test set without confidence intervals, an overfitting warning, and a provisional taxonomy. Accuracy alone overstates performance, and no clinical claim is supported.

## Decision before Federated Learning

**READY, with conditions.** The Day 7 blockers were that the taxonomy, subject separation, and evaluation protocol had not been reviewed. Day 8 and Day 8.5 completed those reviews, and the baseline is now fixed and reproducible as a comparator. Federated experiments may begin, provided that they:

1. use the same taxonomy, split, preprocessing, model, and evaluation protocol unless one factor is deliberately changed;
2. report macro, per-class, and undefined metrics alongside accuracy;
3. derive client partitions only from training groups, keeping the validation and test groups fixed;
4. do not claim a federated gain on classes that this test set cannot evaluate.

The Day 8 questions about taxonomy and target definition remain open. If they are resolved later, the result must be a new versioned baseline. No federated learning, privacy mechanism, blockchain, or proposed algorithm was implemented in this step.


# Day 10 — Federated learning baseline (FedAvg)

**Date:** 2026-10-05

## Completed

- Implemented simulated FedAvg in `src/federated/`: configuration, group-level client partitioning, train-only client datasets, client update, sample-weighted FedAvg aggregation, round orchestration, per-round validation, one-time test evaluation, communication-cost calculation, and figures. The `centralized_cnn_v1` architecture and `src/models/` are unchanged.
- Added `configs/federated/fedavg_v1.json`, `tests/test_federated.py` (10 tests), the executed notebook `notebooks/07_federated_fedavg_baseline.ipynb` (the single official run), and `docs/federated_learning_baseline.md`.

## Configuration

- **Clients:** 5, built from the 33 training groups only. Strategy `random_group_equal_count`: seeded (42) permutation of whole groups into 7/7/7/6/6 groups, holding 16,406 / 18,698 / 16,563 / 13,946 / 11,937 segments. No validation or test data is assigned, and there is no balancing.
- **Training:** 20 rounds, 1 local epoch, full participation (5 clients per round), Adam with lr 0.001 reset each round, batch size 256, unweighted cross-entropy, seed 42, same initial weights as the centralized model.
- **Selection:** the round with minimum validation loss, the same rule as the baseline.

## Results

- **Selected round:** 3 (validation loss 1.1894).
- **Test (evaluated once):** accuracy 0.7069, macro-F1 0.0848, weighted-F1 0.6574, macro OvR AUROC 0.6733.
- **Compared with `centralized_cnn_v1`**, FedAvg **degrades** performance: accuracy −0.062, macro-F1 −0.055, weighted-F1 −0.063.
  - The global model predicts only `N` and `V` on test. Paced `/` F1 falls from 0.6694 to 0 (1,900 of 2,078 predicted as `V`), and 1,961 of 2,001 `L` beats are predicted as `V`.
  - `V` F1 falls from 0.5276 to 0.3058, and `N` F1 rises from 0.9008 to 0.9669.
  - `A`, `F`, `L`, `Q`, and `a` have zero F1 under both models, and seven classes remain unevaluable.
  - AUROC rises (macro 0.6238 → 0.6733; `L` 0.6593 → 0.9913) and weighted specificity rises. This reflects score ranking and the absence of rare-class predictions, not better decisions.
- **Communication:** a 41,428-byte payload per model transfer, 414,280 bytes per round, and 8.29 MB in total over 20 rounds. This is analytical, not network-measured.

## Convergence observations

Rounds 0–1 predict all `N`. Validation loss is lowest at round 3 and then rises while client training loss keeps falling (0.4455 → 0.1922), the same overfitting pattern as the centralized run. Validation macro-F1 peaks at 0.0913 (round 8), below the centralized validation macro-F1 of 0.1737 in every round. Validation accuracy collapses in rounds 18–20 (to 0.5907). client_04 has the highest local loss throughout. A re-run of rounds 1–3 reproduced the saved global model bit-identically.

## Limitations

This is a simulation on one public database; the clients are not hospitals. There is no privacy guarantee and no blockchain, and clients are assumed honest. It is a single seed and partition with no confidence intervals. Training budgets differ (3 local epochs at the selected round against 1 centralized epoch). The label skew is extreme: `/` and `L` are each present on only 2 of 5 clients. All `centralized_cnn_v1` limitations still apply, including 8 of 15 classes with test support and inferred subject-equivalence groups rather than verified patient IDs.

## Unresolved questions

- How much of the degradation comes from label non-IID partitioning versus client drift, optimizer reset, or BatchNorm averaging? Candidate one-factor ablations: an IID-like stratified group partition, other local-epoch counts, and keeping BatchNorm statistics local.
- Is the collapse to `N`/`V` stable across seeds and partitions? Repeated runs are needed before any FL-versus-centralized conclusion.
- Should later comparisons match the training budget (sample passes) between centralized and federated training?
- Given that per-class AUROC is high while argmax recall is zero (for example `L`), should the evaluation protocol add threshold-independent or calibrated decision analyses? Any such change must be versioned and applied to both baselines.
- The taxonomy questions from Day 8 remain open.

No privacy, encryption, blockchain, attack, or proposed-algorithm code was implemented. The work was committed locally after validation; nothing was pushed.


# Day 11 — FedAvg robustness and experimental control

**Date:** 2026-10-05

## Purpose

Test whether the Day 10 FedAvg degradation and collapse to `N`/`V` are reproducible across predefined seeds and group partitions, with the `fedavg_v1` configuration held fixed. The goal is reproducibility, not optimizing FedAvg. The official Day 10 run is preserved unchanged as `fedavg_v1_official`. All robustness outputs are under `results/robustness/`. Full report: `docs/fedavg_robustness.md`.

## Design

- **Seeds:** 42, 123, and 2024, fixed in advance. Each seed sets both the training seed and the partition seed.
- **Partitions:** 5 clients from whole training groups.
  - Primary: the natural `random_group_equal_count` partition.
  - Diagnostic: `controlled_group_partition`, which visits groups rarest content first and assigns each to the client lacking most of its classes. It uses training labels only and does no balancing.
- **Runs:** 6 in total. Each selects its round on validation loss and evaluates the test set once. Training settings are unchanged: 20 rounds, 1 local epoch, Adam with lr 0.001, batch size 256, full participation, and sample-weighted FedAvg.
- **Pre-training finding:** `/` and `L` each occur in only 2 training groups, so no whole-group partition can place them on more than 2 of 5 clients. The controlled partition reaches the minimum feasible number of client-class absences (25–26, against 29–30 for natural partitions) but cannot change the skew of `/` and `L`.

## Results

- `fedavg_v1_natural_s42` reproduced the official model and test predictions **bit-identically**.
- **Test macro-F1:** natural 0.0851 ± 0.0134 (range 0.0717–0.0986); controlled 0.0816 ± 0.0031. Centralized: 0.1399.
- **Test accuracy:** natural 0.6567 ± 0.0762; controlled 0.6600 ± 0.0358. Centralized: 0.7688.
- **Test weighted-F1:** natural 0.6021 ± 0.0489; controlled 0.6060 ± 0.0155. Centralized: 0.7208.
- These are means ± sample SD over 3 runs each. They are descriptive only, with no significance test.
- **Degradation is stable:** all 6 runs are below the centralized baseline on accuracy, macro-F1, weighted-F1, and macro recall.
- **N/V collapse is effectively reproducible:** at least 99.0% of test predictions are `N` or `V` in every run. The strictly two-class form occurs only in the official seed-42 run; other runs add a few spurious `R` predictions (no `R` in test). One run (natural s2024) predicts `/` once, correctly, and `L` twice, both wrongly. Across all runs, 1 of 2,078 paced beats and 0 of 2,001 `L` beats are detected. Paced `/` F1 is about 0 in every run, against 0.669 centralized.
- Macro AUROC is above centralized in 4 of 6 runs, which again reflects score ranking rather than decisions.
- The selected round varies widely (2, 2, 3, 15, 17, 20), and validation macro-F1 never exceeds 0.10.
- The controlled partition narrowed the spread of test macro-F1 but did not change its level or prevent the collapse.

## Unresolved questions

- Is losing `/` and `L` caused by their 2-record confinement (label skew), or by FedAvg mechanics such as client drift, Adam reset, or BatchNorm averaging? The partition control could not separate these. Candidate one-factor experiments: keep BatchNorm statistics local (FedBN-style), use SGD or keep optimizer state across rounds, vary local epochs, and as a diagnostic only, an oracle partition that splits records within the training set.
- How variable is the centralized baseline across seeds? Only FedAvg variability has been measured.
- Is minimum validation loss an appropriate selection criterion when it diverges from validation macro-F1? Any change must be predeclared and applied to both baselines.
- Day 8 taxonomy questions remain open.

## Finalization

In `docs/fedavg_robustness.md`, three over-broad statements were corrected against the saved tables: the gap-versus-spread comparison, the `/`/`L` detection counts, and the description of the convergence shapes. The robustness runs had overwritten each other's centralized-vs-FedAvg comparison tables because they shared one unprefixed file name. These were rebuilt per run by post-processing the saved test outputs, and the misleading leftover pair was removed; no experiment was re-run. Per-run robustness checkpoints and `.npz` prediction files were added to `.gitignore` as regenerable. The configuration, code, tests, summary tables, and figures stay tracked.

No privacy, encryption, blockchain, attack, or proposed-algorithm code was implemented. Nothing was committed or pushed.


# Day 12 — BatchNorm diagnostic (FedBN-style)

**Date:** 2026-10-05

## Purpose

This is a one-factor diagnostic of the Day 10/11 FedAvg degradation: does keeping BatchNorm local to each client, instead of averaging it, change the minority-class collapse? Full report: `docs/fedbn_diagnostic.md`. Outputs: `results/diagnostics/fedbn/`.

## Design

- **Intervention:** `aggregation.batchnorm = "local"`. Every client keeps its own BatchNorm weight, bias, running statistics, and counter across rounds. Conv1d and Linear parameters still go through sample-weighted FedAvg.
- **Fixed:** the natural 5-client partition, seeds 42, 123, and 2024, and every Day 11 setting (20 rounds, 1 local epoch, Adam lr 0.001, batch size 256, full participation, minimum-validation-loss selection, one test evaluation, frozen metrics).
- **Pairing:** each run is paired by seed with the Day 11 natural FedAvg run, so partition, initialisation, and training order are identical.
- **Evaluation model:** the shared weights plus the sample-weighted average of client BatchNorm states. It is used only for evaluation and never broadcast. Under standard FedAvg the same construction is exactly the broadcast model.
- **Extra diagnostic:** each client's personalised model (own BatchNorm) is scored on validation only.

## Results

- **Selected rounds:** 3, 2, 12 (FedAvg: 3, 2, 17).
- **Test macro-F1:** 0.0844 / 0.0974 / 0.0706 against FedAvg 0.0848 / 0.0986 / 0.0717. Mean paired Δ −0.0009, range −0.0012 to −0.0004.
- **Test accuracy:** 0.7031 / 0.6928 / 0.5579 (Δ −0.0055). **Weighted-F1:** 0.6542 / 0.5822 / 0.5519 (Δ −0.0060). **Macro recall:** Δ −0.0007.
- **Macro AUROC:** 0.6694 / 0.6596 / 0.6451 (Δ +0.0107; higher in 2 of 3 seeds).
- **Predicted classes:** 2 / 3 / 3. Between 99.77% and 100% of test predictions are `N` or `V`, and `/` and `L` recall is 0 in every run.
- **Personalised models:** client models reach validation macro-F1 of 0.037–0.092, never near the centralized 0.1737.
- **Size of the effect:** all paired differences are an order of magnitude smaller than seed-to-seed variability (0.0269) and the gap to centralized (at least 0.041).
- **Mechanism check:** with momentum 0.1 and 47–74 batches per local epoch, at most about 0.7% of the starting running statistics survive one epoch. Running statistics are therefore re-estimated locally in both arms. The effective manipulated factor is mainly BatchNorm affine locality.
- **Integrity:** the default code path reproduced the official Day 10 model bit-identically. All 257 pre-existing files under `results/`, `data/splits/`, `configs/`, and `tests/` are byte-identical.

## Conclusion

The BatchNorm-only intervention does **not** materially change the minority-class collapse in this setup. This argues against BatchNorm averaging being the main driver for these seeds and this configuration. It does not establish what the cause is, and it does not rule out BatchNorm effects in other regimes, such as fewer local steps or frozen statistics.

## Unresolved questions

- Remaining one-factor candidates:
  - optimizer state (SGD, or keeping Adam state across rounds);
  - local epochs and client drift;
  - training budget matched to the centralized run;
  - the selection criterion (validation loss against macro-F1).
- Label skew of `/` and `L`, which are confined to 2 training records each. Only a diagnostic that splits records within the training set could vary this.
- How variable is the centralized baseline across seeds?
- Day 8 taxonomy questions remain open.

No privacy, encryption, blockchain, attack, or proposed-algorithm code was implemented. Nothing was committed or pushed.


# Day 13 — Client-level differential privacy (DP-FedAvg)

**Date:** 2026-10-05

## Purpose

This is a privacy-only intervention on the frozen standard FedAvg setup: client-level DP via per-client update clipping and Gaussian noise added at server aggregation. Full report: `docs/differential_privacy.md`. Outputs: `results/privacy/`.

## Implementation

- `src/federated/privacy.py`:
  - L2 clipping of each client's full floating-point state update, covering weights, biases, BatchNorm affine parameters, and running statistics.
  - Sample-weighted aggregation with fixed public weights.
  - Gaussian noise N(0, (z · C · max p_k)²), giving sensitivity C · max p_k under add/remove-one-client adjacency.
  - A seeded per-round noise stream.
  - The accountant.
- **Accountant:** no DP library is installed, so one is implemented and documented. With full participation, T Gaussian rounds compose exactly to μ-GDP (μ = √T / z; Dong, Roth & Su 2019), and ε comes from the analytic Gaussian conversion (Balle & Wang 2018). RDP (Mironov 2017) is reported as an upper-bound cross-check. q < 1 is rejected.
- **Integration:** an optional `privacy` section in `run_fedavg.py`. It cannot be combined with local BatchNorm, and the default path still reproduces the official Day 10 model bit-identically. The paired comparison module was generalised to any one-factor arm; it reproduces the FedBN tables exactly.
- **Tests:** 11 new (`tests/test_federated_privacy.py`), 50 in total.

## Declared configuration (fixed before running)

| Setting | Value |
|---|---|
| Privacy unit | One client |
| Clipping norm C | 1.0 (a priori) |
| Noise multiplier z | 2.69 (minimum for ε ≤ 8 is 2.6843) |
| δ | 10⁻⁵ |
| Client sampling rate q | 1.0 |
| Rounds T | 20 |
| Seeds | 42, 123, 2024; noise stream seed · 1000003 + 7919 + round |
| Resulting ε | **7.98** (GDP; RDP bound 9.36) |

Partition, training, round selection, and the single test evaluation are identical to Day 11, and runs are paired by seed.

## Results

- **Selected round:** 0 in all 3 seeds. No DP-trained round beat the untrained initial model on validation loss.
- **Test results** therefore describe the initial networks:

| Seed | Accuracy | Macro-F1 | Weighted-F1 | Macro AUROC | Predictions |
|---|---:|---:|---:|---:|---|
| 42 | 0.6554 | 0.0528 | 0.5189 | 0.4244 | all `N` |
| 123 | 0.0778 | 0.0096 | 0.0112 | 0.5823 | all `V` |
| 2024 | 0.0071 | 0.0081 | 0.0094 | 0.6320 | mostly `Q`/`e` |

- **Paired degradation against FedAvg:** macro-F1 −0.032 / −0.089 / −0.064 (−38% / −90% / −89%), accuracy −0.05 / −0.62 / −0.56.
- **3-seed means:** accuracy 0.247 ± 0.356 against FedAvg 0.657 ± 0.076; macro-F1 0.024 ± 0.025 against 0.085 ± 0.013.
- **Noise dominated the signal:** the noise-to-signal ratio was 63–69 per round, matching the a priori estimate of about 65. All client updates were clipped in every round. The per-coordinate noise (about 0.63) is 4–11× the initial weight scale. Noisy BatchNorm variances were clamped to 0. Validation loss was 1.3·10⁶ to 3·10¹² after round 1.
- **Communication:** DP adds 0 bytes (8.29 MB over 20 rounds, the same as FedAvg). No material wall-clock increase was observed (timing not controlled).

## Interpretation

At client-level ε = 7.98 with only 5 clients, DP-FedAvg destroys utility. This is consistent with the noise scale required to hide one of 5 clients (sensitivity about C/4 against a signal of at most C). It does not show that DP is unusable for this task in general: other privacy units, more clients, other ε values, and DP-compatible normalisation were not tested. The guarantee is central DP with a trusted server, it covers all 20 released models, and the logged unclipped-norm diagnostics are outside it.

## Fixes made during Day 13

- The `title` parameter of `plot_robustness_convergence` was shadowed by a loop variable, so the Day 12 FedBN convergence figure had the wrong suptitle. It was fixed and that one figure regenerated from saved histories.
- Validation-loss panels now switch to a log axis when values span more than 50×.

## Unresolved questions

- Record-level DP-SGD within clients as an alternative privacy unit, with its own accounting and the caveat that segments from one record are correlated.
- An ε sweep to map the utility-privacy curve, and the effect of more (simulated) clients.
- DP-compatible normalisation would need a deliberate, versioned architecture change, which is out of scope under the frozen model.
- Earlier open questions remain: the cause of the `N`/`V` collapse, the variability of the centralized baseline, and taxonomy.

No homomorphic encryption, blockchain, attack, or proposed-algorithm code was implemented. Nothing was committed or pushed.
