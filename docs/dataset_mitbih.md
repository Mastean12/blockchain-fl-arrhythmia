# MIT-BIH Arrhythmia Database — Dataset Exploration

**Exploration date:** 2026-10-02  
**PhysioNet version:** 1.0.0  
**Local raw directory:** `data/raw/mitbih/`

## Source and provenance

The dataset was downloaded from the official [PhysioNet MIT-BIH Arrhythmia Database v1.0.0 page](https://physionet.org/content/mitdb/1.0.0/) via its public PhysioNet S3 mirror. The download contains 706 files (109,343,870 bytes on disk, about 109.3 MB). The supplied `SHA256SUMS.txt` manifest covers 704 files; all 704 were checked by the executed notebook and matched. The checksum manifest itself and `RECORDS`/`ANNOTATORS` are not entries in that manifest.

The source is licensed under [Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/1.0.0/#files-panel). The PhysioNet record page asks users to cite Moody and Mark, “The impact of the MIT-BIH Arrhythmia Database,” *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50 (2001), DOI [10.1109/51.932724](https://doi.org/10.1109/51.932724), as well as the dataset DOI [10.13026/C2F305](https://doi.org/10.13026/C2F305).

PhysioNet describes 48 half-hour, two-channel ambulatory ECG excerpts from 47 subjects. It notes that 23 recordings were selected at random from a larger collection and 25 were selected to include less common but clinically significant arrhythmias. Therefore, the observed annotation frequencies below describe this database, not population prevalence.

## Raw directory structure

The dataset files are kept under `data/raw/mitbih/` and are excluded from Git by `.gitignore`. No signal or annotation source file was rewritten by the analysis. The notebook only reads raw files through WFDB and validates them against the supplied checksums.

- 48 `.hea` headers, 48 `.dat` signal files, and 48 `.xws` display files for records listed in `RECORDS`.
- 48 current `.atr` annotation files corresponding to `RECORDS`, plus the separately supplied original `102-0.atr` file (49 `.atr` files in the top-level directory).
- `RECORDS`, `ANNOTATORS`, and `SHA256SUMS.txt` metadata files.
- The download also preserves the source `mitdbdir/` documentation and `x_mitdb/` auxiliary files.

The notebook uses the 48 identifiers in the official `RECORDS` file. In particular, record 102 uses `102.atr`; `102-0.atr` is retained as supplied and is not substituted.

## Record and signal characteristics

| Characteristic | Observed value |
|---|---:|
| Records listed in `RECORDS` | 48 |
| Sampling frequency | 360 Hz for all 48 records |
| Signal channels per record | 2 for all 48 records |
| Signal length per channel | 650,000 samples for all 48 records |
| Full record array shape | 650,000 samples × 2 channels |
| Header-derived duration | 1,805.56 seconds (30.09 minutes) for each record |
| Channel units | mV |
| Channel configurations | 6 distinct ordered channel-name pairs |
| Reference annotation files analyzed | 48 `.atr` files |

Channel-name configurations from headers:

| Ordered channel names | Records |
|---|---:|
| MLII; V1 | 40 |
| MLII; V2 | 2 |
| MLII; V5 | 2 |
| V5; V2 | 2 |
| MLII; V4 | 1 |
| V5; MLII | 1 |

The PhysioNet description gives the recording digitization as 360 samples per second per channel, 11-bit resolution, over a 10 mV range. Header configurations vary by record; channel order is retained as recorded. WFDB physical values were used only to display short signal excerpts, not to transform or save signal data.

## Annotation information and counting convention

Each `.atr` file is read using `wfdb.rdann(record, "atr")`. Across the 48 records the notebook counted **112,647 annotation instances**: **109,494 beat annotations** and **3,153 non-beat/event annotations**.

The beat distribution below counts the standard WFDB beat symbols individually. This is a source-symbol count, not an AAMI class remapping. Descriptions follow the [MIT-BIH directory symbol table](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm) and [PhysioNet annotation-code reference](https://archive.physionet.org/physiobank/annotations.shtml). Symbols such as `+` (rhythm change), `~` (signal-quality change), `!` (ventricular flutter wave), `x` (non-conducted P-wave), and `|` (QRS-like artifact) are retained in the all-annotation inventory but excluded from heartbeat-class counts.

### Beat-symbol distribution

| Symbol | Source description | Count | Share of beat annotations |
|---|---|---:|---:|
| `N` | Normal beat | 75,052 | 68.5444% |
| `L` | Left bundle branch block beat | 8,075 | 7.3748% |
| `R` | Right bundle branch block beat | 7,259 | 6.6296% |
| `V` | Premature ventricular contraction | 7,130 | 6.5118% |
| `/` | Paced beat | 7,028 | 6.4186% |
| `A` | Atrial premature beat | 2,546 | 2.3252% |
| `f` | Fusion of paced and normal beat | 982 | 0.8969% |
| `F` | Fusion of ventricular and normal beat | 803 | 0.7334% |
| `j` | Nodal (junctional) escape beat | 229 | 0.2091% |
| `a` | Aberrated atrial premature beat | 150 | 0.1370% |
| `E` | Ventricular escape beat | 106 | 0.0968% |
| `J` | Nodal (junctional) premature beat | 83 | 0.0758% |
| `Q` | Unclassifiable beat | 33 | 0.0301% |
| `e` | Atrial escape beat | 16 | 0.0146% |
| `S` | Supraventricular premature or ectopic beat (atrial or nodal) | 2 | 0.0018% |
| **Total** | **15 observed beat symbols** | **109,494** | **100%** |

### Other annotation symbols

| Symbol | Source description | Count |
|---|---|---:|
| `+` | Rhythm change marker; rhythm label is carried in auxiliary note | 1,291 |
| `~` | Signal-quality change marker | 616 |
| `!` | Ventricular flutter wave | 472 |
| `"` | Comment annotation | 437 |
| `x` | Non-conducted P-wave (blocked atrial premature beat) | 193 |
| `|` | Isolated QRS-like artifact | 132 |
| `[` | Start of ventricular flutter/fibrillation | 6 |
| `]` | End of ventricular flutter/fibrillation | 6 |
| **Total** | **8 observed non-beat/event symbols** | **3,153** |

Rhythm labels are stored as auxiliary notes attached to annotation events, rather than as beat symbols. The notebook summarizes these separately. The most frequent observed rhythm-note strings include `(N` (530), `(B` (221), `(AFIB` (107), `(PREX` (103), `(T` (83), `(VT` (61), `(P` (60), and `(AFL` (45). The notebook's full auxiliary-note counts are saved in `results/tables/mitbih_aux_note_counts.csv`.

## Descriptive class-imbalance observations

- `N` is the most frequent beat symbol: 75,052 annotations (68.5444% of the counted beat annotations).
- `S` is the least frequent observed beat symbol: 2 annotations (0.0018%). The largest-to-smallest observed symbol count ratio is **37,526:1**.
- Five symbols (`N`, `L`, `R`, `V`, `/`) account for **95.4792%** of beat annotations.
- Nine of the 15 observed beat symbols each account for less than 1% of beat annotations.
- These are descriptive counts under the raw WFDB symbol convention. They do not establish clinical prevalence, utility, model difficulty, or which labels should be merged for a later experiment.

The distribution figure uses a logarithmic count axis so that low-frequency symbols are visible alongside `N`. Record-level counts are also supplied; per-record totals range from 1,519 to 3,400 annotation instances (mean 2,346.81), including both beat and non-beat annotations.

## Representative recordings and artifacts

Records 100 and 200 are displayed for the first 10 seconds of each of their two channels, with the annotation symbols at their WFDB sample positions. They are illustrative examples selected for exploration, not a sampling strategy or clinical interpretation.

- Executable notebook: `notebooks/01_dataset_exploration.ipynb`
- Record metadata: `results/tables/mitbih_record_metadata.csv`
- Channel configurations: `results/tables/mitbih_channel_configurations.csv`
- Beat-symbol counts: `results/tables/mitbih_beat_annotation_distribution.csv`
- All annotation symbols: `results/tables/mitbih_all_annotation_symbol_counts.csv`
- Auxiliary notes: `results/tables/mitbih_aux_note_counts.csv`
- Record-level annotation counts: `results/tables/mitbih_record_annotation_counts.csv`
- Beat distribution plot: `results/figures/mitbih_beat_annotation_distribution.png`
- Representative annotated ECG plots: `results/figures/mitbih_representative_annotations.png`

## Assumptions, limitations, and questions before preprocessing

1. The notebook counts each symbol present in the database's `atr` annotation files. It does not infer new events or convert annotations into a target label scheme.
2. `!` is treated as a ventricular-flutter-wave event, not as a heartbeat class; rhythm markers and other event symbols remain visible in the full-symbol table.
3. Auxiliary-note NUL padding is stripped in memory for readable summary tables; raw annotation files remain unchanged.
4. All 48 records are included in the descriptive summary, with no record exclusions or test-period-only restriction.
5. PhysioNet reports 48 records from 47 subjects. The record-to-subject mapping and any dependence between records must be established before designing a subject-aware validation scheme; a record must not automatically be assumed to be an independent subject.
6. Before preprocessing, decide and document the target label definition and whether rare source symbols remain separate, are excluded, or are grouped. Any such mapping needs a literature-based rationale and a clear report of the resulting counts.
7. Decide how to handle paced beats, escape beats, fusion beats, unclassifiable beats, rhythm-only annotations, and signal-quality/artifact events for the eventual research task.
8. Establish leakage controls and an evaluation design before segmentation or split creation. No filtering, resampling, beat-window extraction, or train/validation/test split was performed in this exploration.

### Day 3 follow-up

For the initial reproducible preprocessing pass, the 15 observed beat symbols were retained as 15 distinct source labels (identity mapping); the eight known event symbols were counted but not converted into heartbeat segments. This is an operational preprocessing choice, not a final task taxonomy or clinical decision. No source beat classes were merged. Boundary beats without a complete 216-sample window were excluded and counted. See [`docs/preprocessing.md`](preprocessing.md) for the complete parameters and the open decisions that remain before model evaluation.
