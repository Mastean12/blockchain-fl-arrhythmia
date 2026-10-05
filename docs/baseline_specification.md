# Frozen centralized baseline specification

Compact reference for the baseline frozen on Day 9 (2026-10-05). The full rationale is in [`baseline_freeze.md`](baseline_freeze.md). Every future experiment uses exactly this baseline unless it deliberately changes one stated factor. Any change creates a new versioned baseline and does not overwrite this one.

```yaml
baseline_id: centralized_cnn_v1
frozen: 2026-10-05
status: frozen exploratory reference (not clinically validated)

dataset:
  name: MIT-BIH Arrhythmia Database
  source_version: PhysioNet 1.0.0
  records: 48
  subjects_reported_by_source: 47
  explicit_patient_ids_available: false
  sampling_frequency_hz: 360
  channels: 2                     # lead pair varies by record; MLII+V1 in 40 records
  processed_version: mitbih_v1
  processed_segments: 109460
  segment_shape: [216, 2]

preprocessing:
  config: configs/preprocessing/mitbih_v1.json
  filter: {type: butterworth_bandpass, low_hz: 0.5, high_hz: 40.0, prototype_order: 4, method: sosfiltfilt, padtype: odd}
  segmentation: {pre_samples: 72, post_samples: 144, window_samples: 216, boundary_policy: drop_incomplete, dropped_edge_windows: 34}
  normalization: {method: per_segment_per_channel_zscore, ddof: 0, epsilon: 1.0e-8, dtype: float32}
  software: {python: 3.12.10, numpy: 2.5.3, scipy: 1.18.1, wfdb: 4.3.1}

labels:
  raw_annotation_symbols: 23
  model_classes: 15               # identity mapping, no merges
  class_order: ["/", "A", "E", "F", "J", "L", "N", "Q", "R", "S", "V", "a", "e", "f", "j"]
  excluded_event_symbols: ["+", "~", "!", "\"", "x", "|", "[", "]"]

split:
  version: mitbih_split_v1
  config: configs/preprocessing/mitbih_split_v1.json
  metadata: data/splits/mitbih_v1/split_metadata.json
  unit: inferred subject-equivalence group (record-derived; records 201+202 grouped)
  groups: 47                      # 47 inferred subject-equivalence groups, not verified patient IDs
  method: seeded random permutation of whole groups, largest-remainder apportionment, not label-stratified
  seed: 42
  target_proportions: {train: 0.70, validation: 0.15, test: 0.15}
  train:      {groups: 33, records: 33, segments: 77550}
  validation: {groups: 7, records: 8, segments: 16351, records_list: ["111", "112", "114", "119", "123", "201", "202", "217"]}
  test:       {groups: 7, records: 7, segments: 15559, records_list: ["100", "101", "107", "113", "214", "219", "233"]}
  cross_split_record_overlap: none
  cross_split_group_overlap: none
  cross_split_exact_duplicate_segments: none
  separation_claim: group-disjoint by inferred subject equivalence; NOT independently verified patient-level separation

model:
  identifier: centralized_cnn_v1
  class: src.models.cnn1d.ECG1DCNN
  input: [batch, 2, 216]
  layers:
    - Conv1d(2->16, k=7, p=3, bias=False), BatchNorm1d, ReLU, MaxPool1d(2)
    - Conv1d(16->32, k=5, p=2, bias=False), BatchNorm1d, ReLU, MaxPool1d(2)
    - Conv1d(32->64, k=3, p=1, bias=False), BatchNorm1d, ReLU
    - AdaptiveAvgPool1d(1), Dropout(0.3), Linear(64->15)
  trainable_parameters: 10127
  checkpoint: results/models/centralized_cnn_v1.pt
  checkpoint_selection: minimum validation loss (epoch 1 of 5; val loss 1.2786)

training:
  config: configs/training/centralized_cnn_v1.json
  optimizer: Adam
  learning_rate: 0.001
  batch_size: 256
  loss: CrossEntropyLoss (unweighted)
  epochs: 5
  seed: 42
  deterministic_algorithms: true
  device: cpu
  class_balancing: none
  augmentation: none
  software: {pytorch: 2.14.1+cpu, scikit_learn: 1.9.1}

evaluation:
  module: src.models.evaluate_cnn
  validation_use: per-epoch monitoring and checkpoint selection only
  test_use: single evaluation of the frozen checkpoint; no tuning on test
  metrics: accuracy; per-class precision, recall/sensitivity, one-vs-rest specificity, F1; confusion matrix
  macro_average: unweighted over all 15 classes, zero_division=0
  weighted_average: by true test support
  auroc: one-vs-rest on softmax; per-class only where defined (8 classes); macro and weighted over defined classes; micro over flattened decisions
  test_classes_with_support: 8   # "/", A, F, L, N, Q, V, a
  test_classes_without_support: 7  # E, J, R, S, e, f, j

metrics_test:   # from results/metrics/baseline_metrics.json (Day 6, not recomputed)
  accuracy: 0.7688
  macro_precision: 0.1510
  macro_recall: 0.1431
  macro_sensitivity_supported: 0.2683
  macro_f1: 0.1399
  weighted_precision: 0.7078
  weighted_recall: 0.7688
  weighted_f1: 0.7208
  macro_specificity: 0.9664
  weighted_specificity: 0.7270
  macro_auroc_ovr_defined: 0.6238
  weighted_auroc_ovr_defined: 0.9004
  micro_auroc_ovr: 0.9624
  zero_f1_supported_classes: ["A", "F", "L", "Q", "a"]

artifacts:
  metrics_json: results/metrics/baseline_metrics.json
  predictions: results/metrics/baseline_test_predictions.npz
  classification_report: [results/metrics/baseline_classification_report.csv, results/metrics/baseline_classification_report.txt]
  roc_metrics: results/metrics/baseline_roc_metrics.csv
  per_class_metrics: results/tables/baseline_per_class_metrics.csv
  confusion_matrix: results/tables/baseline_confusion_matrix.csv
  figures: [results/figures/baseline_confusion_matrix.png, results/figures/baseline_roc_curve.png, results/figures/centralized_cnn_v1_training.png]
  training_history: [results/metrics/centralized_cnn_v1_history.json, results/metrics/centralized_cnn_v1_history.csv]

sha256:   # verified on Day 9; identical to the Day 8.5 verification
  results/models/centralized_cnn_v1.pt: 95dce4cc3334a616aa46229fcc350606a60d8bbb5e56262866cc95fb4a6e499f
  results/metrics/baseline_metrics.json: f5367cee6971870f5e592c0eae943eedba5ad55bbe9a4a79cede6ba55630fb34
  results/metrics/baseline_test_predictions.npz: 5ed17e1e57c6eebe84de33154b37cfcab33f0c4df6ee3ca0c4455b7af85a8c45
  results/metrics/baseline_classification_report.csv: 75c9570d436bb11f53ad8a5ad5c517d6361fb9a4b232fb947ffac406b89a498a
  results/metrics/baseline_roc_metrics.csv: 764a349d687764aae97d05fed08fa973e247eaeb102845334a4ac56c3b08b4de
  results/tables/baseline_confusion_matrix.csv: e07f4ed25d48687aae68fa6d4904a223be1847e1fa03776bae4549e2ce721923
  results/tables/baseline_per_class_metrics.csv: 75c9570d436bb11f53ad8a5ad5c517d6361fb9a4b232fb947ffac406b89a498a
  data/splits/mitbih_v1/split_metadata.json: 229b0db300778b0e5203b7d13537840410d52ce86752279f4e419195596831c2
  configs/preprocessing/mitbih_v1.json: 4b1b9d4fec033641277e3249ef8fe8dad74207349270f6e9457ddaffc0a4444f
  configs/preprocessing/mitbih_split_v1.json: dee4584c04a1c8106b2a83d5f3e56f549c484afdc8561b2b862e8a616d277eb3
  configs/training/centralized_cnn_v1.json: af5dcc1b1d6d4108c3c72d5a57addd93e3a75c0fd4335a162d3e03d7ba7c59bd
  data/processed/mitbih_v1/manifest.json: 4655b2afb5feb5b33726d84b5b40ce1b62c28b8e13d66d4b30ef22361fbfc117
  src/models/cnn1d.py: 606f24de4778a6e58462a37ac72b76a45ff0e21ac1337d7113518b46a2f861ce

known_limitations:
  - severe class imbalance (N = 68.5% of segments, 65.5% of test)
  - low macro-F1 (0.1399) despite 0.7688 accuracy
  - zero F1 for supported classes A, F, L, Q, a
  - only 8 of 15 classes have positive test support; validation lacks E, Q, R, S, e
  - 47 groups are inferred from the source subject count; explicit patient IDs unavailable
  - single fixed 7-record test set; no repeated splits or confidence intervals
  - checkpoint from epoch 1; overfitting warning; no balancing or tuning
  - provisional 15-symbol taxonomy; rhythm/event annotations out of scope
  - lead configuration varies across records
  - single database; no external validation; no clinical claims
```

Hash note: values are SHA-256 of the raw file bytes (`sha256sum`). The split NPZ shards are not tracked in Git but are included in the 111-file comparison against Day 8.5 recorded in [`baseline_freeze.md`](baseline_freeze.md#9-integrity-verification-day-9). The 48 processed NPZ shards were not part of the Day 8.5 hash set. They are checked indirectly through the manifest hash and the split validation.
