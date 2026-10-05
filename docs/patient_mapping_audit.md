# Day 8 — Patient and record mapping audit

**Audit date:** 2026-10-05  
**Dataset:** MIT-BIH Arrhythmia Database, PhysioNet v1.0.0

## Executive finding

The local dataset contains no unique subject identifier field in the signal headers, record metadata CSV, annotation files, or saved split shards. However, the official PhysioNet documentation shipped with the local dataset (`data/raw/mitbih/mitdbdir/intro.htm`) says the database has **48 records from 47 subjects**, and explicitly says records **201 and 202 came from the same male subject**. The known pair accounts for the single-record excess over the subject count. Assuming these published counts cover the full 48-record set, the other 46 records each represent one distinct subject. This supports 47 inferred subject-equivalence groups without inventing patient names or IDs.

The Day 4 split uses 47 pseudonymous groups and keeps 201/202 together. Its saved memberships have no group or record overlap across train, validation, and test. On the official metadata evidence above, the current split is **disjoint by inferred subject-equivalence group**. Explicit patient identifiers are not available for all records. The conclusion is a grouping inference from the source's total and explicitly identified pair, not a record-to-person identifier table, so it should not be described as independently verified patient-level separation.

## Sources inspected and available fields

| Source | Available subject/record information | Finding |
|---|---|---|
| Local PhysioNet `mitdbdir/intro.htm` | Cohort-level subject count/demographics; named relation for one record pair | Describes 48 records from 47 subjects; states records 201 and 202 came from the same male subject; gives 25 men/22 women. It does not supply subject IDs for all records. |
| Local `mitdbdir/records.htm` | Record-level summaries, including record number, channel names, sex, and age | Has an entry for each record. Demographic attributes are not unique identifiers and cannot be used to infer identity by matching age/sex. The 201/202 entries agree on male, age 68; their same-subject relation is explicitly stated in the documentation. |
| `results/tables/mitbih_record_metadata.csv` | Record ID, sampling rate, length, channel count/names, units, duration | Contains 48 record rows; no patient/subject ID column. |
| WFDB headers and annotations | Record names and signal/annotation structure | No subject ID field used by this project. |
| Day 3 processed/split NPZ metadata | `record_ids`, `patient_groups`, annotation and signal provenance | `patient_groups` are generated pseudonyms: `mitdb_group_201_202` for the known pair and `record_<id>` for all other records. These strings are not source patient IDs. |
| `data/splits/mitbih_v1/split_metadata.json` | Split record/group membership and overlap validation | Reports 48 records, 47 groups, seed 42, group-level assignment, and empty pairwise overlap lists. |

No identity is created from demographic similarity. The raw metadata does not provide names or a complete named subject table.

## Current split membership

| Split | Records | Groups | Segments | Record identifiers |
|---|---:|---:|---:|---|
| Train | 33 | 33 | 77,550 | 102, 103, 104, 105, 106, 108, 109, 115, 116, 117, 118, 121, 122, 124, 200, 203, 205, 207, 208, 209, 210, 212, 213, 215, 220, 221, 222, 223, 228, 230, 231, 232, 234 |
| Validation | 8 | 7 | 16,351 | 111, 112, 114, 119, 123, 201, 202, 217 |
| Test | 7 | 7 | 15,559 | 100, 101, 107, 113, 214, 219, 233 |

The 201/202 shared-subject group is assigned wholly to validation. The other 46 singleton groups are assigned once each. Group labels are pseudonyms only.

## Leakage checks and risks

- Saved split metadata has empty `record_ids` and `group_ids` overlap lists for train/validation, train/test, and validation/test.
- `validate_saved_splits` reopened the saved NPZ shards during Day 8 validation and completed without errors. It rechecked record/group membership, exact segment-content-plus-label duplicates across splits, dimensions, metadata lengths, finite values, valid labels, and reproducibility-related saved metadata.
- No record crosses partitions. Records 201 and 202, the only documented same-subject pair, are together; therefore no known subject-equivalence group crosses a partition.
- No exact ECG-segment-plus-label duplicate was found across partitions. Neighboring beat windows may overlap within a record by design, but each complete record and subject-equivalence group belongs to one split, so those windows do not cross splits.
- The split is not randomized by true subject IDs, because none are present; it uses the supported 47-group equivalence relation. The claim depends on the completeness/correctness of the official 48-record/47-subject count and the documented pair. If an authoritative source later identifies another shared subject across records, the grouping and split must be revised.
- Subject-equivalence group disjointness does not make the test set representative: it remains a small fixed selection of records with several absent classes and highly uneven per-class support. Record/device/channel differences may also make held-out-record performance different from held-out-institution performance.

## Assessment

**Can patient-level separation be established?** Separation can be established at the level of 47 inferred subject-equivalence groups, based on the local official cohort count plus the explicitly identified 201/202 pair. It cannot be established as independently verified patient-level separation, because explicit patient identifiers are not available for all records and the project cannot provide named or numeric patient identities. This refines Day 7's more cautious conclusion, which considered the processed record-derived keys without fully using the source cohort-count constraint.

**Does the current split have patient/record leakage?** No known cross-split overlap of records or inferred subject-equivalence groups is present. The known same-subject pair is grouped together and assigned to validation. The split audit also found no exact duplicated segment-plus-label samples across partitions.

**Remaining limitations:** This establishes independence within this dataset's documented subject grouping, not broader external validity. Keep group provenance and this inference in all future experiments; do not describe pseudonymous group keys as actual patient IDs.

## Reproducibility references

- Local source overview: [PhysioNet MIT-BIH directory introduction](https://physionet.org/content/mitdb/1.0.0/mitdbdir/intro.htm); local copy is `data/raw/mitbih/mitdbdir/intro.htm` (raw data intentionally not tracked)
- Per-record notes: [PhysioNet MIT-BIH record notes](https://physionet.org/content/mitdb/1.0.0/mitdbdir/records.htm); local copy is `data/raw/mitbih/mitdbdir/records.htm` (raw data intentionally not tracked)
- Processed record metadata: [`mitbih_record_metadata.csv`](../results/tables/mitbih_record_metadata.csv)
- Split definition and membership: [`dataset_splits.md`](dataset_splits.md) and `data/splits/mitbih_v1/split_metadata.json`
- Generated per-label group support: [`day8_class_group_support.csv`](../results/tables/day8_class_group_support.csv)
