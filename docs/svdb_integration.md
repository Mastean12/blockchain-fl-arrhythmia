# SVDB dataset integration and external evaluation v1

## Scope and current status

This provides a versioned preprocessing and record-split path for the MIT-BIH Supraventricular Arrhythmia Database (SVDB). No model training was performed. A frozen MIT-BIH checkpoint was evaluated once on the held-out SVDB test records as documented below. The integration does not change MIT-BIH data, splits, configurations, results, the HE+DP protocol, or blockchain code. The importer and tests are `src/preprocessing/svdb.py` and `tests/test_svdb_integration.py`; configuration is in `configs/preprocessing/svdb_v1.json` and `configs/preprocessing/svdb_split_v1.json`.

SVDB uses the same two-channel beat-window input dimensions after resampling. Its annotation mapping is limited to exact identity symbols in the frozen MIT-BIH class vocabulary; unsupported beat symbols are explicitly excluded and counted, known event symbols are counted but not segmented, and unknown symbols stop processing. The real release has now been audited and processed, and a frozen-checkpoint external evaluation is available. The scope is limited by the annotation and identity caveats below.

## Verified source facts

- The official PhysioNet release is SVDB version 1.0.0, DOI [10.13026/C2V30W](https://doi.org/10.13026/C2V30W), with 78 half-hour ECG recordings selected to supplement supraventricular-arrhythmia examples in MIT-BIH. The official page states the [Open Data Commons Attribution License v1.0](https://physionet.org/content/svdb/1.0.0/).
- PhysioNet provides a `RECORDS` list, WFDB `.hea`/`.dat` records, `.atr` annotations, and a SHA-256 manifest. Its `ANNOTATORS` file describes `atr` as reference beat and signal-quality annotations.
- The official [800.hea header](https://physionet.org/files/svdb/1.0.0/800.hea) declares two signals, 128 Hz, 230,400 samples, and generic `ECG1`/`ECG2` channel names. The existing MIT-BIH preprocessing expects two channels at 360 Hz, so the importer applies polyphase resampling by 45/16, maps annotation indices to the target grid by nearest-sample rounding, then reuses the existing 0.5–40 Hz filter, 0.2/0.4-second beat windows, and per-window/channel z-score normalization.
- PhysioNet's [SVDB description](https://archive.physionet.org/physiobank/database/html/about.htm) says annotations were made semi-automatically and were not audited to the same extent as MIT-BIH Arrhythmia Database annotations. Treat annotation labels as reference labels with this limitation, not as error-free clinical ground truth.
- The PhysioNet [standard annotation table](https://physionet.org/static/lightwave/doc/annotations.html) defines the beat-symbol vocabulary and cautions that each database's documentation must be checked. This project uses only identical source symbols already present in the MIT-BIH target vocabulary; it does not merge or infer classes.

## Label handling

`svdb_v1.json` maps the existing 15 beat symbols to themselves: `N`, `L`, `R`, `V`, `/`, `A`, `f`, `F`, `j`, `a`, `E`, `J`, `Q`, `e`, and `S`. Identity mapping is a source-code mapping, not a clinical taxonomy claim. The interpretation of each code must follow the source annotation convention.

The configured out-of-vocabulary beat symbols `B`, `r`, `n`, and `?` are explicitly excluded and counted. Known non-beat/event symbols are also counted and omitted from beat windows. Any symbol outside the supported and exclusion lists raises an error, so processing cannot silently drop or relabel a newly encountered annotation. Counts are emitted per record in `record_summary.csv` and all source-symbol counts are included in the in-memory audit result; update the versioned mapping only after reviewing the source definition.

The complete-release symbol inventory was measured during the 2026-10-16 run; counts are recorded below. No unknown symbol was observed. The identity mapping establishes code-level compatibility with the checkpoint vocabulary, not equivalence of annotation quality or clinical interpretation across databases.

## Identity, splits, and leakage limits

Splits are made at the complete-record level with a seeded 70/15/15 allocation (seed 42), using available record-derived keys. Segment windows are never independently divided across splits. The saved `split_metadata.json` records record membership, seed, assignment unit, and validation results. It explicitly does not claim patient-disjoint evaluation.

The dataset-level documentation does not provide a complete record-to-patient map, and record numbers are not patient identifiers. Consequently the implementation can prevent record leakage but cannot prove that different SVDB records belong to different people or detect possible person overlap with MIT-BIH. This limitation must accompany any external-evaluation result.

## Reproduction

Place the unmodified official release under `data/raw/svdb/` (the acquired copy is in `data/raw/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0/`), including `RECORDS` and `SHA256SUMS.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -c "from src.preprocessing.svdb import process_svdb, build_svdb_splits; raw='data/raw/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0'; process_svdb(raw, 'data/processed/svdb_v1'); build_svdb_splits('data/processed/svdb_v1', 'data/splits/svdb_v1')"
```

Both output functions refuse overwrite by default. Preserve raw files as downloaded, retain the generated manifest/config copies and split metadata, and review annotation counts and exclusions before using the outputs.

## Tests and next experiment

The new automated tests cover symbol parsing and exclusions, unknown-symbol rejection, signal and annotation resampling, 216×2 preprocessing output, record provenance, deterministic split generation, and split disjointness. They use synthetic signals and do not validate SVDB annotations or model utility.

The external evaluation is complete for the frozen checkpoint and seeded held-out records, as detailed below. Do not retrain, tune thresholds, revise preprocessing, or change mappings based on the SVDB test results.

## Source citations

Use the SVDB DOI and cite Greenwald, *Improved detection and classification of arrhythmias in noise-corrupted electrocardiograms using contextual information*, Ph.D. thesis, Harvard-MIT Division of Health Sciences and Technology, 1990, as requested by PhysioNet.

## Initial environment verification (2026-10-15)

At the initial v1 integration stage, the SVDB release was not locally present and direct workspace access to the official endpoint failed. This is historical status; the dataset was subsequently supplied and verified as documented in the following section.

At that stage, the frozen centralized checkpoint was present at `results/models/centralized_cnn_v1.pt`, with SHA-256 `95dce4cc3334a616aa46229fcc350606a60d8bbb5e56262866cc95fb4a6e499f`, input shape `[2, 216]`, and the configured 15-symbol MIT-BIH output mapping. That established structural compatibility only; SVDB annotations and external metrics had not yet been validated.

At that stage, the existing targeted regression command

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_preprocessing tests.test_splitting tests.test_evaluate_cnn tests.test_svdb_integration -v
```

passed 15 tests. These synthetic/mocked checks did not replace real-release validation. The real release audit and evaluation below supersede the then-open acquisition item. Keep outputs versioned and do not tune on SVDB test results.

## Complete release audit, preprocessing, splits, and evaluation (2026-10-16)

### Source audit and annotation handling

The release at `data/raw/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0/` contains 78 unique IDs in `RECORDS`. All 78 IDs have non-empty `.hea`, `.dat`, and `.atr` files; there are no extra `.hea` records. All 416 entries in `SHA256SUMS.txt` were present and matched their SHA-256 values. WFDB inspection found 78 recordings with two channels (`ECG1`, `ECG2`), 128 Hz, and 230,400 samples each. There were 187,877 annotation entries, with these source-symbol frequencies:

| Symbol | Count | Handling |
|---|---:|---|
| `N` | 162,339 | Identity mapped |
| `S` | 12,188 | Identity mapped |
| `V` | 9,943 | Identity mapped |
| `|` | 2,211 | Known non-beat event; counted, not segmented |
| `~` | 1,082 | Known non-beat event; counted, not segmented |
| `Q` | 79 | Identity mapped |
| `F` | 23 | Identity mapped |
| `J` | 9 | Identity mapped |
| `a` | 1 | Identity mapped |
| `+` | 1 | Known non-beat event; counted, not segmented |
| `B` | 1 | Configured excluded beat; counted, not segmented |

No unknown symbols occurred. Configured excluded beat symbols `r`, `n`, and `?` had zero occurrences. PhysioNet’s standard annotation definitions support the shared meanings of the mapped codes (including `N`, `S`, `V`, `F`, `J`, `Q`, and `a`); the source SVDB annotations remain semi-automatically generated and less extensively reviewed than MIT-BIH annotations. This evaluation therefore measures agreement with the released reference annotations, not independently adjudicated clinical truth.

Record `860` has one final `N` annotation at sample 230,400, exactly the exclusive signal endpoint. It is included in source-symbol counts but excluded from segmentation because no signal samples exist there to form a beat window. The importer now records this case explicitly as `terminal_boundary_excluded_count=1` and still rejects annotations beyond the endpoint. In addition, 51 in-range beat annotations were dropped by the configured incomplete-window edge rule. The one `B` beat and 3,294 event annotations were explicitly counted and omitted. These exclusions account for the difference between source annotations and 184,530 output segments.

### Preprocessing and split validation

Configuration `configs/preprocessing/svdb_v1.json` resampled the two 128 Hz channels to 360 Hz using polyphase ratio 45/16, mapped annotations to the nearest target sample, applied the existing 0.5–40 Hz fourth-order Butterworth zero-phase filter, extracted 0.2 s before and 0.4 s after each beat, dropped incomplete edge windows, and normalized each segment/channel with z-score normalization. All 78 output records passed finite-value, label, provenance, and `[216, 2]` shape validation. The processed corpus contains **184,530 segments** at 360 Hz, with mapped class counts: `N` 162,290; `S` 12,187; `V` 9,941; `Q` 79; `F` 23; `J` 9; `a` 1.

Split configuration `configs/preprocessing/svdb_split_v1.json` uses seed 42 and 70/15/15 record-held-out allocation. Resulting partitions are train 54 records / 130,544 segments, validation 12 / 26,742, and test 12 / 27,244. Record IDs and available group keys are disjoint, split membership passed content-duplicate validation, and a second build under the same seed reproduced identical membership. SVDB does not supply record-to-patient identifiers: these are record-disjoint splits only, not verified patient-disjoint splits. Cross-dataset patient overlap with MIT-BIH cannot be assessed.

### Frozen-checkpoint external evaluation

Checkpoint: `results/models/centralized_cnn_v1.pt`; SHA-256 `95dce4cc3334a616aa46229fcc350606a60d8bbb5e56262866cc95fb4a6e499f`; input `[2,216]`; 15 output symbols matching the exact configured identity mapping. It was selected on MIT-BIH validation (best epoch 1) and was neither retrained nor tuned on SVDB. Evaluation used only the held-out test records `808, 812, 820, 821, 826, 852, 853, 864, 870, 881, 882, 883` (27,244 segments).

Measured accuracy was **0.66756** and macro-F1 was **0.07199**. Macro-F1 follows the evaluator’s fixed 15-class convention, assigning zero to classes with no test support; the low score and class imbalance must be considered alongside accuracy. Weighted-F1 was 0.71642. Support and per-class precision/recall/F1:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `N` | 23,683 | 0.88344 | 0.75113 | 0.81193 |
| `Q` | 17 | 0.00000 | 0.00000 | 0.00000 |
| `S` | 2,464 | 0.00000 | 0.00000 | 0.00000 |
| `V` | 1,079 | 0.21036 | 0.36886 | 0.26792 |
| `a` | 1 | 0.00000 | 0.00000 | 0.00000 |
| `/`, `A`, `E`, `F`, `J`, `L`, `R`, `e`, `f`, `j` | 0 each | 0 | 0 | 0 |

The full 15×15 confusion matrix, per-class report, predictions, metrics JSON, and figures are in `results/svdb_external_v1/`. The confusion-matrix CSV is `results/svdb_external_v1/tables/baseline_confusion_matrix.csv`; class metrics are in `results/svdb_external_v1/metrics/baseline_classification_report.csv`. For the five supported test rows, nonzero confusion counts are: `N→/ 3, N→L 1,024, N→N 17,789, N→R 3,590, N→V 1,277`; `Q→/ 1, Q→N 7, Q→R 1, Q→V 8`; `S→/ 5, S→L 2, S→N 1,821, S→R 427, S→V 209`; `V→/ 14, V→A 3, V→L 9, V→N 518, V→R 137, V→V 398`; `a→N 1`. All other matrix cells are zero.

This is a single external evaluation on one fixed record split, not evidence of broad generalization. The test split has only 17 `Q` and one `a` example, and ten model classes have no test support. The SVDB annotations have lower review depth than MIT-BIH, and patient-level independence is unverified. No SVDB training, test-driven threshold choice, or test-driven preprocessing/mapping change was performed.

### Reproduction and verification

The processed manifest and record summaries are in `data/processed/svdb_v1/`; split membership and integrity results are in `data/splits/svdb_v1/split_metadata.json`. External evaluation outputs are versioned under `results/svdb_external_v1/`. The checkpoint evaluator now records the supplied checkpoint path and SHA-256 in its metrics JSON.

#### Current worktree provenance recheck (2026-10-16)

The read-only audit rechecked all 78 raw SVDB record triplets with `verify_record_files(..., checksums=True)` against `SHA256SUMS.txt` (416 checksum entries); every listed source record passed. It independently revalidated the recovered processed shards, deterministic split membership, record/group separation, content/provenance duplicates, split metadata counts, and SHA-256 equality between each processed shard and its copied split shard. The recovered split has 54/12/12 records and 130,544/26,742/27,244 segments for train/validation/test. `patient_disjoint_claim` remains false.

The MIT-BIH source list has 48 records; its saved 33/8/7 split covers all 48, with the validator reporting no record/group overlap or content/provenance duplicate. Saved split metadata SHA-256 is `229b0db300778b0e5203b7d13537840410d52ce86752279f4e419195596831c2`. These checks read the current raw and recovered artifacts; they did not regenerate preprocessing, splits, or evaluation outputs.

Read-only commands from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\tests;$PWD"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from src.preprocessing.ecg_io import listed_records,verify_record_files; raw=Path('data/raw/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0'); records=listed_records(raw); [verify_record_files(raw,r,checksums=True) for r in records]; print(len(records), 'SVDB source records verified')"
.\.venv\Scripts\python.exe -c "from src.preprocessing.svdb import validate_svdb_splits; print(validate_svdb_splits('data/processed/svdb_v1_recovered_20261016','data/splits/svdb_v1_recovered_20261016','configs/preprocessing/svdb_split_v1.json','configs/preprocessing/svdb_v1.json'))"
.\.venv\Scripts\python.exe -c "from src.preprocessing.splitting import validate_saved_splits; result=validate_saved_splits('data/splits/mitbih_v1','configs/preprocessing/mitbih_v1.json'); print({name:len(records) for name,records in result['record_sets'].items()})"
```

The relevant regression command is:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_preprocessing tests.test_splitting tests.test_evaluate_cnn tests.test_svdb_integration -v
```

It passed 15 tests after adding explicit coverage for the exact-endpoint annotation rule. To reproduce the evaluation without a GUI backend, use:

```powershell
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -c "from src.models.evaluate_cnn import evaluate_checkpoint; evaluate_checkpoint('results/models/centralized_cnn_v1.pt', 'data/splits/svdb_v1/test', 'configs/preprocessing/svdb_v1.json', 'results/svdb_external_v1')"
```

The evaluation output directory already exists; preserve it and do not rerun into it without an explicit overwrite/version decision.

## Recovered provenance verification (2026-10-16)

The original generated locations `data/processed/svdb_v1/` and `data/splits/svdb_v1/` were present but inaccessible to the current Windows identity. They were not ACL-modified or overwritten. A separate deterministic recovery was generated from the locally readable raw release, using the unchanged v1 preprocessing and split configurations, under:

- `data/processed/svdb_v1_recovered_20261016/`
- `data/splits/svdb_v1_recovered_20261016/`

The recovery processed 78 records and 184,530 segments after checking each record against the source SHA-256 manifest. The recovered manifest SHA-256 is `b6968cc886278f6f698e535174b97208271984d1a0360ecf36ea2774f27321cf`. The recovered test split contains the same 12 record IDs and 27,244 segments recorded in the historical evaluation summary. Its `split_metadata.json` SHA-256 is `8113dd48408cf485130dedc99c43b587b019326f17ef6a8ab74e4389210e6cb2`. The copied preprocessing and split configurations match the repository's v1 configurations. The split builder reported no record or available-group overlap and passed its content-duplicate check. This independently establishes reproducible membership and split validity; it does not recover the inaccessible original metadata bytes.

To keep provenance and evaluation outputs separate, the frozen checkpoint was re-evaluated once on the recovered test split into `results/svdb_external_v1_recovered_20261016/`. The 12 test record IDs, test size (27,244), checkpoint SHA-256, and every saved metric exactly match the historical `results/svdb_external_v1/metrics/baseline_metrics.json`. The recovered `baseline_metrics.json` SHA-256 is `d376c8658013791bb5af625b033278d44b6aa9a470f47d88ea8a588b1f17fcf1`. This is a reproduction of the frozen evaluation, not a second test-driven model-selection or tuning run. The recovered per-beat prediction array remains ignored by Git.

The exact recovery/evaluation commands, from the repository root, are:

```powershell
.\.venv\Scripts\python.exe -c "from src.preprocessing.svdb import process_svdb, build_svdb_splits; raw='data/raw/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0'; process_svdb(raw, 'data/processed/svdb_v1_recovered_20261016'); build_svdb_splits('data/processed/svdb_v1_recovered_20261016', 'data/splits/svdb_v1_recovered_20261016')"
.\.venv\Scripts\python.exe -c "from src.models.evaluate_cnn import evaluate_checkpoint; evaluate_checkpoint('results/models/centralized_cnn_v1.pt', 'data/splits/svdb_v1_recovered_20261016/test', 'configs/preprocessing/mitbih_v1.json', 'results/svdb_external_v1_recovered_20261016', batch_size=512, device='cpu')"
```

These versioned data roots are specifically Git-ignored; the new small evaluation summary remains available as evidence. Do not replace or delete the historical paths if access is restored.
