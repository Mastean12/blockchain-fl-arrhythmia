# Day 8 ? Label taxonomy audit

**Audit date:** 2026-10-05  
**Dataset:** MIT-BIH Arrhythmia Database, PhysioNet v1.0.0  
**Scope:** trace source annotation symbols through configured preprocessing, saved split labels, and the Day 5/6 CNN class indices. No label, model, split, or Day 6 evaluation artifact was changed.

## Summary and central finding

The implemented label pipeline is an identity mapping for all 15 configured beat annotation symbols. Each source symbol becomes the same preprocessing label and the same processed label; the model maps those labels to indices 0?14 in sorted-symbol order. No beat class is merged. Eight labels have positive support in the held-out test partition. **There is no eight-class merged taxonomy in the current config, processed data, checkpoint, or evaluation report.** ?Eight supported classes? means eight of the 15 model outputs have at least one positive test example.

Raw annotations contain 23 distinct symbols: 15 beat symbols and 8 non-beat/event symbols. The raw beat total is 109,494. Preprocessing produced 109,460 windows and recorded 34 incomplete boundary windows as dropped; the 3,153 non-beat/event annotations do not receive beat windows. Every encountered symbol is covered by either the beat map or non-beat list; unknown symbols raise an error rather than being silently skipped.

## 1. Raw annotation-symbol distribution

Counts below are the actual Day 2 WFDB annotation counts across the 48 records and were independently recounted from the local `.atr` files in `notebooks/06_label_and_leakage_audit.ipynb`. Auxiliary rhythm notes are separate metadata attached to annotations; they are not additional annotation symbols or model classes.

| Source symbol | PhysioNet annotation description | Source type | Raw count |
| --- | --- | --- | --- |
| `!` | Ventricular flutter wave | non-beat/event | 472 |
| `"` | Comment annotation | non-beat/event | 437 |
| `+` | Rhythm change marker; rhythm label is carried in the auxiliary note | non-beat/event | 1,291 |
| `/` | Paced beat | beat | 7,028 |
| `A` | Atrial premature beat | beat | 2,546 |
| `E` | Ventricular escape beat | beat | 106 |
| `F` | Fusion of ventricular and normal beat | beat | 803 |
| `J` | Nodal (junctional) premature beat | beat | 83 |
| `L` | Left bundle branch block beat | beat | 8,075 |
| `N` | Normal beat | beat | 75,052 |
| `Q` | Unclassifiable beat | beat | 33 |
| `R` | Right bundle branch block beat | beat | 7,259 |
| `S` | Supraventricular premature or ectopic beat (atrial or nodal) | beat | 2 |
| `V` | Premature ventricular contraction | beat | 7,130 |
| `[` | Start of ventricular flutter/fibrillation | non-beat/event | 6 |
| `]` | End of ventricular flutter/fibrillation | non-beat/event | 6 |
| `a` | Aberrated atrial premature beat | beat | 150 |
| `e` | Atrial escape beat | beat | 16 |
| `f` | Fusion of paced and normal beat | beat | 982 |
| `j` | Nodal (junctional) escape beat | beat | 229 |
| `x` | Non-conducted P-wave (blocked atrial premature beat) | non-beat/event | 193 |
| `\|` | Isolated QRS-like artifact | non-beat/event | 132 |
| `~` | Signal-quality change marker | non-beat/event | 616 |

**Totals:** 112,647 annotation instances; 109,494 beat symbols; 3,153 non-beat/event symbols. The persisted, machine-readable table is [`day8_raw_annotation_symbols.csv`](../results/tables/day8_raw_annotation_symbols.csv).

## 2. Complete source-to-model mapping

The current config is [`mitbih_v1.json`](../configs/preprocessing/mitbih_v1.json). `segment_record` maps each configured beat symbol to its configured value, which is the same symbol. The saved shards preserve `labels` and `source_symbols` identically. The training/evaluation loaders create the class order by sorting the configured label values; the Day 6 prediction artifact records that exact order.

| Source symbol | Source annotation name | Raw count | Preprocessing label | Processed segments | Boundary drops | Final model class / index | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `!` | Ventricular flutter wave | 472 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `"` | Comment annotation | 437 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `+` | Rhythm change marker; rhythm label is carried in the auxiliary note | 1,291 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `/` | Paced beat | 7,028 | `/` | 7,027 | 1 | `/` / 0 | Identity mapping; retained as one output class |
| `A` | Atrial premature beat | 2,546 | `A` | 2,546 | 0 | `A` / 1 | Identity mapping; retained as one output class |
| `E` | Ventricular escape beat | 106 | `E` | 106 | 0 | `E` / 2 | Identity mapping; retained as one output class |
| `F` | Fusion of ventricular and normal beat | 803 | `F` | 802 | 1 | `F` / 3 | Identity mapping; retained as one output class |
| `J` | Nodal (junctional) premature beat | 83 | `J` | 83 | 0 | `J` / 4 | Identity mapping; retained as one output class |
| `L` | Left bundle branch block beat | 8,075 | `L` | 8,072 | 3 | `L` / 5 | Identity mapping; retained as one output class |
| `N` | Normal beat | 75,052 | `N` | 75,028 | 24 | `N` / 6 | Identity mapping; retained as one output class |
| `Q` | Unclassifiable beat | 33 | `Q` | 33 | 0 | `Q` / 7 | Identity mapping; retained as one output class |
| `R` | Right bundle branch block beat | 7,259 | `R` | 7,255 | 4 | `R` / 8 | Identity mapping; retained as one output class |
| `S` | Supraventricular premature or ectopic beat (atrial or nodal) | 2 | `S` | 2 | 0 | `S` / 9 | Identity mapping; retained as one output class |
| `V` | Premature ventricular contraction | 7,130 | `V` | 7,129 | 1 | `V` / 10 | Identity mapping; retained as one output class |
| `[` | Start of ventricular flutter/fibrillation | 6 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `]` | End of ventricular flutter/fibrillation | 6 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `a` | Aberrated atrial premature beat | 150 | `a` | 150 | 0 | `a` / 11 | Identity mapping; retained as one output class |
| `e` | Atrial escape beat | 16 | `e` | 16 | 0 | `e` / 12 | Identity mapping; retained as one output class |
| `f` | Fusion of paced and normal beat | 982 | `f` | 982 | 0 | `f` / 13 | Identity mapping; retained as one output class |
| `j` | Nodal (junctional) escape beat | 229 | `j` | 229 | 0 | `j` / 14 | Identity mapping; retained as one output class |
| `x` | Non-conducted P-wave (blocked atrial premature beat) | 193 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `\|` | Isolated QRS-like artifact | 132 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |
| `~` | Signal-quality change marker | 616 | Excluded (non-beat/event) | 0 | 0 | ? | Non-beat/event symbol; not a heartbeat class, so no beat window/model label |

The exact crosswalk, including all source symbols and counts, is also saved as [`day8_label_pipeline_counts.csv`](../results/tables/day8_label_pipeline_counts.csv).

### Exclusion and boundary rationale

- The eight event/marker symbols `!`, `"`, `+`, `[`, `]`, `x`, `|`, and `~` are explicitly listed as non-beat annotations. They mark rhythm/event boundaries, signal quality, comments, or other non-beat events; the configured task extracts heartbeat windows and therefore does not assign them beat classes. `+` auxiliary rhythm text is counted separately in the Day 2 auxiliary-note table and is not incorporated into this beat classifier.
- Of 109,494 beat annotations, 34 fall too close to a record boundary to construct the configured 72-sample pre / 144-sample post window. The configured `drop_incomplete` policy excludes these windows. By symbol, the observed drops are `/`: 1, `F`: 1, `L`: 3, `N`: 24, `R`: 4, and `V`: 1. All other mapped beat symbols have zero boundary drops.
- No other beat symbol is excluded, merged, or renamed. The full annotation and label count equality checks passed in the notebook. The exclusion of `!`, `[`, `]`, and `x` may matter to the project?s broader arrhythmia-detection aim because these source symbols describe ventricular flutter/fibrillation boundaries, flutter waves, or blocked atrial events; the beat-classification pipeline?s exclusion is internally explicit, but its fit to the final research question still needs literature/task justification.

## 3. Processed classes and split distribution

The CNN has **15 outputs**, with this explicit class-index map:

| Model index | Class name (source symbol) |
| --- | --- |
| 0 | `/` |
| 1 | `A` |
| 2 | `E` |
| 3 | `F` |
| 4 | `J` |
| 5 | `L` |
| 6 | `N` |
| 7 | `Q` |
| 8 | `R` |
| 9 | `S` |
| 10 | `V` |
| 11 | `a` |
| 12 | `e` |
| 13 | `f` |
| 14 | `j` |

These processed counts equal the configured 15-class task. `N` comprises 75,028/109,460 processed windows (68.55%); `S` has 2 windows, `e` 16, and `Q` 33. Those are severe support differences, not grounds to merge labels after inspecting scores.

| Class | Index | Processed total | Train | Validation | Test | Records with class | Groups with class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/` | 0 | 7,027 | 3,407 | 1,542 | 2,078 | 4 | 4 |
| `A` | 1 | 2,546 | 2,418 | 78 | 50 | 27 | 26 |
| `E` | 2 | 106 | 106 | 0 | 0 | 2 | 2 |
| `F` | 3 | 802 | 782 | 7 | 13 | 17 | 16 |
| `J` | 4 | 83 | 80 | 3 | 0 | 5 | 5 |
| `L` | 5 | 8,072 | 3,948 | 2,123 | 2,001 | 4 | 4 |
| `N` | 6 | 75,028 | 53,489 | 11,342 | 10,197 | 40 | 39 |
| `Q` | 7 | 33 | 29 | 0 | 4 | 6 | 6 |
| `R` | 8 | 7,255 | 7,255 | 0 | 0 | 6 | 6 |
| `S` | 9 | 2 | 2 | 0 | 0 | 1 | 1 |
| `V` | 10 | 7,129 | 5,049 | 870 | 1,210 | 37 | 36 |
| `a` | 11 | 150 | 28 | 116 | 6 | 7 | 6 |
| `e` | 12 | 16 | 16 | 0 | 0 | 1 | 1 |
| `f` | 13 | 982 | 722 | 260 | 0 | 3 | 3 |
| `j` | 14 | 229 | 219 | 10 | 0 | 5 | 5 |

### Eight labels with positive test support

This is a **test-support subset**, not an eight-class remapping. The remaining seven model classes stay in the model and reports but have no positive test examples.

| Original class | Processed total | Train | Validation | Test | Records with class | Groups with class |
| --- | --- | --- | --- | --- | --- | --- |
| `/` | 7,027 | 3,407 | 1,542 | 2,078 | 4 | 4 |
| `A` | 2,546 | 2,418 | 78 | 50 | 27 | 26 |
| `F` | 802 | 782 | 7 | 13 | 17 | 16 |
| `L` | 8,072 | 3,948 | 2,123 | 2,001 | 4 | 4 |
| `N` | 75,028 | 53,489 | 11,342 | 10,197 | 40 | 39 |
| `Q` | 33 | 29 | 0 | 4 | 6 | 6 |
| `V` | 7,129 | 5,049 | 870 | 1,210 | 37 | 36 |
| `a` | 150 | 28 | 116 | 6 | 7 | 6 |

Test-unsupported labels are `E` (106 processed), `J` (83), `R` (7,255), `S` (2), `e` (16), `f` (982), and `j` (229). No test sensitivity/recall or AUROC can be estimated for a class with no positive test cases. Counts by all splits are in [`day8_split_class_distribution.csv`](../results/tables/day8_split_class_distribution.csv); the eight-row support subset is [`day8_test_supported_8_class_distribution.csv`](../results/tables/day8_test_supported_8_class_distribution.csv).

## 4. Baseline review

The Day 6 baseline remains a valid **exploratory 15-way source-symbol classification run** under the recorded seed-42 split and training protocol. It does not represent an eight-class model or support performance claims for the seven test-absent classes. Its test accuracy is 0.7688, while macro-F1 is 0.1399 and weighted-F1 is 0.7208. `A`, `F`, `L`, `Q`, and `a` have zero F1 despite positive test support; `L` has 2,001 positive test segments and zero recall. Accuracy is strongly influenced by `N` (10,197 test examples), so it is not an adequate standalone summary.

The current taxonomy is transparent and traceable to source annotations, but its choice as the final research endpoint has not been justified by a documented task definition and literature review. The direct identity crosswalk itself is not an undocumented merge; rather, the unresolved issue is whether this 15-symbol beat task and exclusion of event/rhythm annotations answer the intended research question. **No taxonomy revision or retraining is justified solely to improve scores.** Keep the existing model/results as a fixed exploratory baseline. Before a future training run, decide whether the research task retains all 15 symbols or uses a literature/task-justified crosswalk. Any new mapping must version the config, state every merge/exclusion, and be evaluated with a protocol chosen before model fitting.

The current split need not be replaced to address patient leakage: the Day 8 mapping audit supports separation by 47 inferred subject-equivalence groups, although this is not independently verified patient-level separation (details in [`patient_mapping_audit.md`](patient_mapping_audit.md)). Coverage remains limited. In particular, `S` occurs in only one group (2 samples), and `e` in one group (16 samples), so neither can appear in all three mutually exclusive group-disjoint splits without leakage. `E` occurs in only two groups. A revised split cannot create independent subjects for these labels. Revisit split design only if the predeclared evaluation objective requires support beyond what this dataset can provide.

## 5. Alternatives for future review (not decisions)

1. Retain the 15 source-symbol classes for a transparent exploratory task and qualify results by per-class support.
2. Define a literature-supported target taxonomy tied to a precise research question, with a complete published source-symbol crosswalk and explicit handling of every event/rare label.
3. Define a narrower task-specific label scope in advance, stating eligibility, exclusions, data retained, and intended claims before rebuilding versioned data/splits and retraining.

No alternative is selected here. A taxonomy must not be selected by whichever mapping produces the best score.

## 6. Reproducibility and audit files

- Notebook: [`06_label_and_leakage_audit.ipynb`](../notebooks/06_label_and_leakage_audit.ipynb)
- Full 23-symbol source table: [`day8_raw_annotation_symbols.csv`](../results/tables/day8_raw_annotation_symbols.csv)
- Full crosswalk and per-symbol edge drops: [`day8_label_pipeline_counts.csv`](../results/tables/day8_label_pipeline_counts.csv)
- 15-class train/validation/test table: [`day8_split_class_distribution.csv`](../results/tables/day8_split_class_distribution.csv)
- Eight test-supported original labels: [`day8_test_supported_8_class_distribution.csv`](../results/tables/day8_test_supported_8_class_distribution.csv)
- Label support by record/group: [`day8_class_group_support.csv`](../results/tables/day8_class_group_support.csv)
- New Day 8 figures: `results/figures/mitbih_label_leakage_audit.png` and `results/figures/mitbih_split_class_support_audit.png`

The Day 6 model, predictions, confusion matrix, reports, and ROC artifacts were read only. Their pre/post SHA-256 hashes are checked as part of Day 8 validation.
