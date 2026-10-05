# Day 11 — FedAvg robustness and experimental control

- **Date:** 2026-10-05
- **Configuration:** [`configs/federated/fedavg_v1_robustness.json`](../configs/federated/fedavg_v1_robustness.json), which uses the unchanged base [`fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Code:** `src/federated/robustness.py`, `src/federated/heterogeneity.py`, and the `controlled_group_partition` strategy in `src/federated/partition.py`
- **Outputs:** `results/robustness/` (separate from the official Day 10 outputs)

> This robustness experiment tests whether the Day 10 FedAvg degradation is reproducible. It is not intended to optimize FedAvg performance.

## 1. Research question

On Day 10, the official FedAvg run (`fedavg_v1`) performed worse than the frozen `centralized_cnn_v1` baseline. Its test macro-F1 was 0.0848 against 0.1399, accuracy 0.7069 against 0.7688, and weighted-F1 0.6574 against 0.7208. Its global model predicted only `N` and `V`. This experiment asks two questions:

1. **Reproducibility.** Does the degradation, and the collapse to `N`/`V`, recur under other predefined seeds and other natural group partitions, with the FedAvg configuration held fixed?
2. **Diagnostic.** Does a partition that reduces client class absence (while still assigning whole groups) avoid the collapse?

## 2. Official Day 10 experiment (unchanged)

`fedavg_v1` and its outputs are treated as **`fedavg_v1_official`** and were not modified:

- `configs/federated/fedavg_v1.json`
- `results/*/federated/fedavg_v1_*`
- `results/models/federated/fedavg_v1_global.pt`

Their SHA-256 hashes were recorded before Day 11 and match after it (§9). The robustness run `fedavg_v1_natural_s42` uses the same seed and partition. It reproduced the official global model **bit-identically**, and its test predictions are identical array by array. Its test evaluation therefore adds no new test information; it confirms the pipeline is deterministic.

## 3. Training budget (documented, not altered)

The Day 10 comparison has a known budget difference:

- The centralized checkpoint was selected at epoch 1, after one pass over the training data with continuous Adam state.
- The official FedAvg model was selected at round 3, after 3 local epochs per client (about 3 passes) with Adam reset each round.

The frozen centralized baseline was not changed. For robustness, the FedAvg configuration is held **fixed** at the `fedavg_v1` values: same CNN, Adam, learning rate 0.001, batch size 256, 1 local epoch, 20 rounds, full participation of 5 clients, sample-weighted FedAvg, and selection by minimum validation loss. Nothing was tuned on validation or test results. Because selection is by validation loss, the effective training budget differs from run to run: the selected rounds were 2, 2, 3, 15, 17, and 20.

## 4. Design

### Seeds

The seeds were fixed in advance and none were added after seeing results: **42, 123, 2024**. Each seed sets both `random_seed` (model initialisation, client shuffling, dropout) and `partition.seed` (group assignment). Seed effects therefore combine partition, initialisation, and training order; they are not isolated.

### Partitions (5 clients, whole training groups only)

| Label | Strategy | Role |
|---|---|---|
| `natural` | `random_group_equal_count` (the Day 10 strategy): seeded permutation of the 33 training groups, cut into 7/7/7/6/6 groups. Labels are not used. | **Primary** robustness experiment |
| `controlled` | `controlled_group_partition`: whole groups, visited rarest content first, each assigned to the open client (at most 7 groups) that lacks the most of the group's classes. Ties go to fewer groups, then fewer segments. Only training-group label counts are used. | **Diagnostic** control only; does not replace the natural experiment |

Neither partition balances, moves, duplicates, or re-weights samples. Every run passes the existing partition validation: no group or record in two clients, no validation or test data, and all 77,550 training segments covered. Every run selects its round on validation loss and evaluates the test set once.

**Structural limit of the controlled partition.** This was found before training, from training labels only. `/` and `L` each occur in only **2** of the 33 training groups. `E` and `f` also occur in 2, and `S` and `e` in 1. Under whole-group assignment, `/` and `L` therefore **cannot be held by more than 2 of the 5 clients** under any partition.

The controlled partition reaches the minimum possible number of client-class absences: 25 for seeds 42 and 2024, and 26 for seed 123, against 29–30 for the natural partitions. It also equalises client sizes. It **cannot** reduce the skew of `/` and `L`, which are the two classes lost on Day 10. It is therefore a partial control: it tests whether reducing the *achievable* skew changes the outcome. It does not test whether the `/` and `L` concentration causes the collapse.

## 5. Results

### Per-run validation (selected round) and test results

| Run | Selected round | Best val loss | Val acc | Val macro-F1 | Val weighted-F1 | Test acc | Test macro P | Test macro R | Test macro-F1 | Test weighted P | Test weighted R | Test weighted-F1 | Test macro AUROC | Classes predicted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| natural s42 (= official) | 3 | 1.1894 | 0.6912 | 0.0814 | 0.6170 | 0.7069 | 0.0759 | 0.1177 | 0.0848 | 0.6364 | 0.7069 | 0.6574 | 0.6733 | 2: N V |
| natural s123 | 2 | 1.3481 | 0.6263 | 0.0517 | 0.5347 | 0.6943 | 0.1117 | 0.1002 | 0.0986 | 0.5253 | 0.6943 | 0.5841 | 0.6463 | 3: N R V |
| natural s2024 | 17 | 1.2538 | 0.7124 | 0.0913 | 0.6380 | 0.5690 | 0.1364 | 0.1052 | 0.0717 | 0.7366 | 0.5690 | 0.5648 | 0.6226 | 5: / L N R V |
| controlled s42 | 15 | 1.3071 | 0.6832 | 0.0835 | 0.6205 | 0.6424 | 0.0754 | 0.1164 | 0.0803 | 0.6419 | 0.6424 | 0.6228 | 0.6971 | 3: N R V |
| controlled s123 | 2 | 1.4346 | 0.6414 | 0.0972 | 0.5740 | 0.7013 | 0.0721 | 0.1061 | 0.0852 | 0.5301 | 0.7013 | 0.6027 | 0.5999 | 3: N R V |
| controlled s2024 | 20 | 1.2492 | 0.6914 | 0.0851 | 0.6261 | 0.6365 | 0.0709 | 0.1165 | 0.0794 | 0.5817 | 0.6365 | 0.5924 | 0.6346 | 3: N R V |
| **centralized_cnn_v1** | epoch 1 | 1.2786 | 0.6853 | 0.1737 | — | **0.7688** | 0.1510 | 0.1431 | **0.1399** | 0.7078 | 0.7688 | **0.7208** | 0.6238 | 6: / A L N R V |

The table is `results/robustness/tables/robustness_runs.csv`. Per-class precision, recall, specificity, F1, and AUROC for every run are in `robustness_per_class.csv`. Centralized validation weighted-F1 was not recorded by the Day 5 training code.

### Variability (descriptive; n = 3 runs per partition, sample standard deviation)

| Metric | Natural: mean ± SD [min, max] | Controlled: mean ± SD [min, max] | Centralized |
|---|---|---|---:|
| Test accuracy | 0.6567 ± 0.0762 [0.5690, 0.7069] | 0.6600 ± 0.0358 [0.6365, 0.7013] | 0.7688 |
| Test macro-F1 | 0.0851 ± 0.0134 [0.0717, 0.0986] | 0.0816 ± 0.0031 [0.0794, 0.0852] | 0.1399 |
| Test weighted-F1 | 0.6021 ± 0.0489 [0.5648, 0.6574] | 0.6060 ± 0.0155 [0.5924, 0.6228] | 0.7208 |
| Test macro recall | 0.1077 ± 0.0090 [0.1002, 0.1177] | 0.1130 ± 0.0060 [0.1061, 0.1165] | 0.1431 |
| Test macro AUROC (defined classes) | 0.6474 ± 0.0254 [0.6226, 0.6733] | 0.6439 ± 0.0493 [0.5999, 0.6971] | 0.6238 |
| Classes predicted on test | 3.33 ± 1.53 [2, 5] | 3.00 ± 0.00 [3, 3] | 6 |
| Selected round | 7.3 ± 8.4 [2, 17] | 12.3 ± 9.3 [2, 20] | — |
| Best validation loss | 1.2638 ± 0.0798 | 1.3303 ± 0.0949 | 1.2786 |

The full summary is in `robustness_summary.csv`. With 3 runs per partition, these values describe the spread observed; they are not population estimates. **No significance test was performed and no result is described as statistically significant.**

### Per-class pattern (test)

| Class (support) | Centralized F1 | FedAvg F1 range, 6 runs | FedAvg recall range |
|---|---:|---|---|
| `/` (2,078) | 0.669 | 0.000–0.001 | 0.000 (1 correct `/` in natural s2024) |
| `L` (2,001) | 0.000 | 0.000 | 0.000 |
| `N` (10,197) | 0.901 | 0.812–0.967 | 0.772–1.000 |
| `V` (1,210) | 0.528 | 0.243–0.667 | 0.502–0.880 |
| `A` `F` `Q` `a` | 0.000 | 0.000 | 0.000 |

What the per-class results show:

- **Paced beats (`/`)** are effectively lost in all 6 runs. Natural s2024 predicts `/` once, and correctly.
- **`V`** F1 is below the centralized 0.528 in 5 of 6 runs. Natural s123 is the exception at 0.667, but its `N` specificity is only 0.121 because it predicts `N` for 14,910 of 15,559 segments.
- **`R` predictions** appear in 5 runs (between 1 and 145 segments). The test set contains no `R`, so every one of these is a false positive.

### Is the N/V collapse reproducible?

- **Strictly "only N and V predicted": no.** This holds for 1 of 6 runs (natural s42, the official run).
- **Effectively: yes, in all 6 runs.** `N` and `V` make up **at least 99.0%** of test predictions in every run (worst case controlled s42: 145 `R` out of 15,559). The only predictions outside `N`/`V`/`R` are in natural s2024: 1 `/` (correct) and 2 `L` (both wrong; the true classes were `F` and `V`). Across all 6 runs, FedAvg correctly identifies 1 of 2,078 paced beats and 0 of 2,001 `L` beats. The centralized model predicts `/` 1,050 times (F1 0.669) and `L` 329 times.

### Is the FedAvg degradation reproducible?

**Yes, for every main decision-based metric in every run.** All 6 runs (3 natural, 3 controlled) fall below `centralized_cnn_v1` on:

- test accuracy (maximum 0.7069 against 0.7688);
- test macro-F1 (maximum 0.0986 against 0.1399);
- test weighted-F1 (maximum 0.6574 against 0.7208);
- test macro recall (maximum 0.1177 against 0.1431).

The closest run is 0.0412 below centralized on macro-F1 and 0.0254 below on macro recall. Both gaps are larger than the full range across the 6 runs (0.0269 and 0.0175). For accuracy and weighted-F1 the runs vary more than that: the closest gaps are 0.0620 and 0.0634, against ranges of 0.1379 and 0.0927. Every run is still below the baseline on both. Macro AUROC over defined classes is higher than centralized in 4 of 6 runs (0.5999–0.6971 against 0.6238). As on Day 10, this reflects score ranking rather than decisions made at the argmax.

## 6. Convergence

The convergence figure is `results/robustness/figures/robustness_convergence.png`, and every round of every run is in `robustness_history.csv`.

- **Shape.** In every run, validation loss falls sharply over rounds 1–2. After that it rises (natural s42, natural s123, controlled s123), stays roughly flat (natural s2024), or declines slowly (controlled s42 and s2024).
- **Selected round.** The round chosen by minimum validation loss is unstable: 2, 2, and 3 in three runs, but 15, 17, and 20 in the others. In those three runs the validation-loss curve is flat and noisy after round 3. In controlled s2024 the minimum is at the final round (20), so a longer run might have selected later. The round count stays fixed at 20, as the design requires.
- **Validation macro-F1.** From round 1 onwards it stays between 0.05 and 0.10 in all 6 runs. (Round 0 is the untrained model: 0.007–0.055.) It never approaches the centralized validation macro-F1 of 0.1737.
- **Loss versus macro-F1.** Several FedAvg runs reach a lower validation loss than the centralized checkpoint (natural s42 1.189, controlled s2024 1.249, against 1.279) while macro-F1 stays about half the centralized value. Validation loss is therefore a weak guide to minority-class performance in this setting.
- **Seed 123.** In both partitions, seed 123 has the highest validation loss and the lowest validation accuracy after round 2. The partition differs between the natural and controlled runs, but the initialisation and training order are shared, so these factors cannot be separated.

## 7. Client heterogeneity

Per-client details for every run (segments, groups, classes present, dominant class and share, second class, missing classes, and the Jensen-Shannon divergence of each client's label distribution against the pooled training distribution) are in `results/robustness/tables/robustness_heterogeneity_clients.csv`. Each run also has its own client tables and heatmap under `results/robustness/{tables,figures}/federated/`.

| Run | Client segments (min–max) | Size CV | Client-class absences | Classes on all 5 clients | Mean JSD vs pooled (bits) | Mean pairwise JSD (bits) | Clients holding `/` / `L` |
|---|---|---:|---:|---:|---:|---:|---|
| natural s42 | 11,937–18,698 | 0.151 | 30 | 4 | 0.083 | 0.167 | 2 / 2 |
| natural s123 | 12,688–17,991 | 0.111 | 30 | 4 | 0.097 | 0.188 | **1** / 2 |
| natural s2024 | 14,012–18,076 | 0.110 | 29 | 4 | 0.088 | 0.179 | 2 / 2 |
| controlled s42 | 15,055–15,781 | 0.018 | 25 | 5 | 0.079 | 0.161 | 2 / 2 |
| controlled s123 | 14,659–16,770 | 0.050 | 26 | 5 | 0.077 | 0.160 | 2 / 2 |
| controlled s2024 | 14,815–16,248 | 0.038 | 25 | 5 | 0.073 | 0.147 | 2 / 2 |

The minimum achievable number of absences under whole-group assignment with 5 clients is 25.

Findings:

- **Natural partitions** consistently have 29–30 client-class absences. Only `A`, `F`, `N`, and `V` are on every client. `N` is the dominant class on every client in every run.
- **Natural seed 123** puts all paced beats on a single client.
- **Controlled partitions** reach or come within 1 of the minimum (25–26), put `R` on all 5 clients, and cut client-size variation substantially (CV 0.02–0.05 against 0.11–0.15). Label divergence falls only modestly (mean pairwise JSD 0.147–0.161 against 0.167–0.188).
- **Outcome.** The controlled partition did **not** prevent the effective collapse or the degradation. It narrowed the spread of test macro-F1 (SD 0.0031 against 0.0134), but its mean is similar (0.0816 against 0.0851).

## 8. Interpretation

1. **The degradation is reproducible.** On this data, model, configuration, and test split, plain FedAvg underperforms the centralized baseline on accuracy, macro-F1, weighted-F1, and macro recall in all 6 runs, across 3 seeds and 2 partition schemes. The Day 10 result is not a one-off artefact of seed 42.
2. **The functional collapse to `N`/`V` is reproducible; the strict two-class form is not.** No run recovers the paced or `L` classes. The extra predicted classes are almost entirely spurious `R`.
3. **Reducing achievable label skew did not change the outcome.** However, the control could not reduce the skew of the two lost classes, because `/` and `L` are each confined to 2 records in the training split. These results therefore **cannot** distinguish "extreme label skew of `/` and `L`" from "FedAvg in general" as the cause of losing those classes. This remains open.
4. **Other candidate contributors remain untested:** client drift, Adam reset each round, BatchNorm statistic averaging, the differing training budget, and the selection criterion. Each would need its own one-factor experiment.

These are empirical observations for this simulated setup. They do not establish a general property of federated learning for ECG and make no clinical claim.

## 9. Integrity

- SHA-256 hashes of all 169 pre-existing files under `results/`, `data/splits/`, `configs/`, `src/`, `notebooks/`, `tests/`, and `docs/` were recorded before Day 11. After Day 11, every artifact, configuration, and test file is unchanged. This includes the centralized checkpoint and metrics, the Day 6 test predictions, `split_metadata.json` and the split shards, and all official Day 10 `fedavg_v1` outputs, including `fedavg_v1_test_predictions.npz`. The only changed files are the intentionally edited code and docs: `src/federated/partition.py`, `config.py`, `run_fedavg.py`, `plots.py`, `evaluation.py` (an optional comparison-column label whose default is still `fedavg_v1`), `docs/research_log.md`, and `.gitignore`.
- `tests/test_federated_robustness.py` asserts the official Day 10 hashes.
- The edits to the existing code only add the new strategy and helpers. The official `random_group_equal_count` path is unchanged, as confirmed by the bit-identical seed-42 reproduction.

## 10. Limitations

1. **Few runs.** 3 seeds per partition give only a descriptive spread. Standard deviations from n = 3 are imprecise, and no significance testing was done.
2. **Confounded seed effects.** Each seed changes partition, initialisation, and training order together.
3. **Partial control.** The controlled partition cannot change the 2-client confinement of `/` and `L` (a property of the dataset under whole-group assignment), so it does not isolate the main suspected mechanism.
4. **Fixed configuration.** The budget difference relative to the centralized checkpoint remains, and the selected round varies from 2 to 20. The centralized baseline itself has one seed, so the variability of the comparator is unknown.
5. **Shared test-set limitations.** The test set has 7 records and only 8 of 15 classes with support, and the 47 groups are inferred subject-equivalence groups rather than verified patient IDs.
6. **Simulation.** Clients are partitions of a public database, there is no privacy mechanism or blockchain, and clients are assumed honest.

## 11. Artifacts

- **Run outputs:** `results/robustness/{metrics,tables,figures,models}/federated/fedavg_v1_{natural,controlled}_s{42,123,2024}_*`
- **Not tracked in Git (regenerable):** the per-run global checkpoints (`results/robustness/models/`) and per-run test prediction files (`results/robustness/metrics/federated/*.npz`) are listed in `.gitignore`. They can be rebuilt deterministically with `src.federated.robustness.run_all()`; the seed-42 run reproduced the official model bit-identically. Their content is summarised in the tracked tables. In particular, the predicted-class counts are in `robustness_runs.csv` and the per-run confusion matrices are in `results/robustness/tables/federated/`. Rebuilding the aggregate tables with `collect_results()` needs these files, so it requires re-running the experiments. The official Day 10 checkpoint and predictions under `results/{models,metrics}/federated/` remain tracked.
- **Aggregate tables:** `results/robustness/tables/`
  - `robustness_runs.csv`
  - `robustness_summary.csv`
  - `robustness_per_class.csv`
  - `robustness_heterogeneity_clients.csv`
  - `robustness_heterogeneity_summary.csv`
  - `robustness_history.csv`
- **Figures:** `results/robustness/figures/robustness_convergence.png` and `robustness_test_metrics.png`
- **Per-run comparison with the centralized baseline:** `results/robustness/tables/federated/<run_id>_centralized_vs_fedavg_{aggregate,per_class}.csv`. These were rebuilt by post-processing (`write_comparison_tables`) from each run's saved test outputs.
  - The original runs wrote these tables under one shared, unprefixed name, so each run overwrote the previous one. The surviving pair held only controlled s2024's values under a `fedavg_v1` column label. It was deleted.
  - Future runs write per-run names. The official Day 10 table names are unchanged.
  - No model was retrained or re-evaluated. All other robustness outputs were byte-identical after the rebuild.
- **Tests:** `tests/test_federated_robustness.py`
