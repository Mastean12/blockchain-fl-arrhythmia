# MIT-BIH Train/Validation/Test Splits (Day 4)

## Source and identifiers

The source is the Day 3 processed dataset, 48 records and 109,460 beat windows. The split unit is `patient_groups` from the processed NPZ shards. Day 3 assigned `mitdb_group_201_202` to records 201 and 202 because the PhysioNet record notes state that record 202 was taken from the same analog tape as record 201. The remaining keys (`record_100`, etc.) are record-derived pseudonyms; they do **not** establish the underlying patient identity or prove that those records belong to distinct people. PhysioNet reports 48 records from 47 subjects, but a complete record-to-subject mapping is not present in this dataset’s processed metadata. [PhysioNet MIT-BIH record notes](https://physionet.org/physiobank/database/html/mitdbdir/records.htm)

The split therefore provides record separation and keeps the one documented shared group together. It does not support an unqualified claim of complete patient-level separation for all records. This limitation must be resolved if verified subject IDs become available.

## Strategy and reproducibility

- **Strategy:** random permutation of complete group keys using NumPy `default_rng`; segments and labels are never independently shuffled or distributed.
- **Seed:** 42, configurable in [`configs/preprocessing/mitbih_split_v1.json`](../configs/preprocessing/mitbih_split_v1.json).
- **Targets:** 70% train, 15% validation, 15% test.
- **Group allocation:** 47 available keys are apportioned by largest remainder to 33/7/7 groups (70.21%/14.89%/14.89%). This puts 201 and 202 in a common single partition. Random group assignment is not stratified by labels.
- **Result format:** each split contains per-record NPZ shards preserving `segments`, `labels`, `source_symbols`, `record_ids`, `patient_groups`, annotation sample positions, segment start/end offsets, sampling frequency, and channel names. `split_metadata.json` records membership, seed, class counts, proportions, overlap checks, and limitations; per-split class counts are also written as CSV.
- **Regeneration:** run `build_splits` from `src.preprocessing.splitting` with the processed dataset, split config, and label config. Existing output is not overwritten by default.

## Actual allocation

| Split | Groups | Records | Record share | Segments | Segment share |
|---|---:|---:|---:|---:|---:|
| Train | 33 | 33 | 68.75% | 77,550 | 70.85% |
| Validation | 7 | 8 | 16.67% | 16,351 | 14.94% |
| Test | 7 | 7 | 14.58% | 15,559 | 14.21% |
| **Total** | **47** | **48** | **100%** | **109,460** | **100%** |

“Groups” are the available separation keys and are not a verified patient count. Validation has eight records because the grouped 201/202 pair is assigned together. The actual segment proportions are close to the requested proportions; record and group shares differ because one group contains two records and segment counts differ by record.

## Class counts by source label

Counts retain the Day 2 identity mapping. Zeroes are explicit; no class was balanced, merged, or removed.

| Source label | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `/` | 3,407 | 1,542 | 2,078 | 7,027 |
| `A` | 2,418 | 78 | 50 | 2,546 |
| `E` | 106 | 0 | 0 | 106 |
| `F` | 782 | 7 | 13 | 802 |
| `J` | 80 | 3 | 0 | 83 |
| `L` | 3,948 | 2,123 | 2,001 | 8,072 |
| `N` | 53,489 | 11,342 | 10,197 | 75,028 |
| `Q` | 29 | 0 | 4 | 33 |
| `R` | 7,255 | 0 | 0 | 7,255 |
| `S` | 2 | 0 | 0 | 2 |
| `V` | 5,049 | 870 | 1,210 | 7,129 |
| `a` | 28 | 116 | 6 | 150 |
| `e` | 16 | 0 | 0 | 16 |
| `f` | 722 | 260 | 0 | 982 |
| `j` | 219 | 10 | 0 | 229 |
| **Total** | **77,550** | **16,351** | **15,559** | **109,460** |

Validation lacks `E`, `Q`, `R`, `S`, and `e`; test lacks `E`, `J`, `R`, `S`, `e`, `f`, and `j`. Validation also has single-digit counts for `F` (7) and `J` (3); test has `Q` (4) and `a` (6). `S` has only two segments overall and both are in train. These are observed limitations of this seeded record-group split, not evidence that any source class should be merged or excluded. Revisit split suitability and label scope before Day 5 evaluation; do not interpret rare-class metrics without accounting for missing classes.

## Leakage and integrity validation

The splitting module checks pairwise disjoint record IDs and group keys, hashes segment content together with its label to detect exact duplicates across partitions, and validates NPZ metadata fields, labels, finite values, `[segments, 216, 2]` dimensions, and metadata lengths. It also checks that a repeated seed produces the same group assignment. The saved output was reopened and these checks passed. No records were excluded; event symbols had already been excluded from heartbeat windows during Day 3, and all 109,460 processed beat segments are assigned exactly once.

## Day 3 metadata repair discovered during split preflight

The first split preflight found that the Day 3 segmentation code created NumPy identifier arrays using `dtype=str`, which selected one-character Unicode storage. That truncated record IDs (for example, `201` to `2`) and group IDs (for example, `mitdb_group_201_202` to `m`) in the original processed NPZ shards. A split based on those values would not have been leakage-safe. The source was not changed. The identifier-width bug was corrected in `src/preprocessing/segmentation.py`, a regression test was added, and all 48 derived records were regenerated from the unchanged raw files. The rebuilt dataset retained 109,460 segments and was checked for 48 distinct record IDs and 47 group keys before splitting. The defective generated shards were replaced after the corrected rebuild passed validation.

## Limitations and Day 5 review

1. A full verified patient-to-record table is not available; record-level separation is guaranteed, while person-level separation is guaranteed only for the known 201/202 pair. Obtain/confirm a complete subject mapping before describing this as patient-held-out evaluation.
2. The random group assignment is not class-stratified, and several labels are absent in validation or test. The seed is fixed for reproducibility but does not solve this support limitation.
3. The Day 3 source-symbol identity mapping remains a provisional label representation, not a final classification target.
4. Records 201 and 202 being from the same analog tape is the documented grouping basis; it is not used to infer any other patient identities.

No model training, balancing, augmentation, federated learning, privacy mechanism, or blockchain implementation was added. The notebook [`03_dataset_splitting_validation.ipynb`](../notebooks/03_dataset_splitting_validation.ipynb) reproduces the split integrity and distribution inspection.
