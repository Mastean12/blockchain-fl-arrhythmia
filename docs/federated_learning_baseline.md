# Day 10 — Federated learning baseline (FedAvg)

- **Experiment ID:** `fedavg_v1`
- **Date:** 2026-10-05
- **Configuration:** [`configs/federated/fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Notebook (official run):** [`notebooks/07_federated_fedavg_baseline.ipynb`](../notebooks/07_federated_fedavg_baseline.ipynb)
- **Reference baseline:** `centralized_cnn_v1` ([`baseline_freeze.md`](baseline_freeze.md))

## 1. Motivation and scope

The research question RQ2 asks how federated learning performs compared with centralized training on heterogeneous ECG data. This experiment sets up the first federated reference point: standard Federated Averaging (FedAvg) training of the **same** CNN, on the **same** training data, evaluated with the **same** protocol as the frozen centralized baseline. Federated training is the only intended change.

Things that are deliberately **not** part of this experiment: differential privacy, homomorphic encryption, blockchain, client attacks or defences, the proposed aggregation method, class balancing, augmentation, and any architecture change.

**What this experiment is and is not:**

- It is a **simulation** of cross-silo federated training on a single machine using the public MIT-BIH database.
- The simulated clients are **not** real hospitals or institutions, and no institution took part.
- Each client's raw ECG segments stay in that client's own dataset object. Only model state (weights and BatchNorm statistics) passes between the clients and the aggregator. This mirrors the federated data flow but is not a deployed, networked system.
- **No privacy guarantee is claimed.** Model updates are sent in plaintext, and FedAvg on its own does not prevent information leaking through updates.
- **No blockchain functionality exists in this experiment.**

## 2. Architecture

```text
               aggregator (global model w_t)
        ┌──────────┬──────────┬──────────┬──────────┐
   broadcast w_t to the participating clients (all 5 each round)
        ▼          ▼          ▼          ▼          ▼
   client_00  client_01  client_02  client_03  client_04
   local data local data local data local data local data
   1 local epoch of Adam from w_t (fresh optimizer each round)
        │          │          │          │          │
        └──── upload local state w_{t+1}^k and sample count n_k ────┘
                                ▼
          FedAvg: w_{t+1} = Σ_k (n_k / Σ_j n_j) · w_{t+1}^k
                                ▼
          evaluate w_{t+1} on the validation split → next round
```

Code is in `src/federated/`:

| Module | Role |
|---|---|
| `config.py` | Load and validate the configuration (FedAvg only, validation-only model selection) |
| `partition.py` | Discover training groups, assign whole groups to clients, validate the partition, build the distribution report |
| `data.py` | `ClientECGDataset`, which opens shards only from `train/` |
| `fedavg.py` | Extract/load the state, sample-weighted `fedavg_aggregate`, `client_update`, client selection, per-client seeds |
| `evaluation.py` | Per-round validation metrics, the one-time test evaluation (reusing the frozen evaluator's `calculate_metrics`), communication cost, comparison with the baseline |
| `run_fedavg.py` | Run the rounds, select the checkpoint, write artifacts |
| `plots.py` | Convergence, client-loss, client-distribution, and confusion-matrix figures |

`src/models/` is unchanged. The model is `ECG1DCNN` (10,127 trainable parameters) with dropout 0.3. It is initialised the same way as `centralized_cnn_v1`: seed 42 is set, then the model is constructed, so FedAvg starts from the same initial weights.

## 3. Client construction and group-level partitioning

- **Eligible data:** only the 33 training groups (33 records, 77,550 segments) listed in `split_metadata.json`. Validation (7 groups, 8 records) and test (7 groups, 7 records) are never assigned to clients.
- **Unit:** one whole inferred subject-equivalence group. Every training group is a single record, because the 201/202 pair is in validation. Individual ECG windows are never split across clients.
- **Strategy `random_group_equal_count`:**
  1. Sort the 33 group IDs.
  2. Permute them with `numpy.random.default_rng(42)`.
  3. Cut the permutation into 5 contiguous chunks with `numpy.array_split`, giving 7/7/7/6/6 groups.
  4. Labels and segment counts are not used, so there is no balancing and each client keeps the natural class mix of its records.
- **Checks:** every training group goes to exactly one client, no group or record appears in two clients, no validation or test record is present, client segment totals sum to 77,550, and every client has at least `min_client_segments` = 1,000 segments. `ClientECGDataset` raises an error for any record that has no `train/` shard.
- **Why 5 clients:** a small cross-silo federation in which each client still holds several records (6–7) and 11,937–18,698 segments. No other client count was tried. The choice was not tuned.

### Client distribution (natural, not balanced)

| Client | Groups | Segments | Share | Classes present | Absent classes | Records |
|---|---:|---:|---:|---:|---|---|
| client_00 | 7 | 16,406 | 21.2% | 9 | `/` `E` `L` `a` `e` `f` | 108 115 118 205 208 228 234 |
| client_01 | 7 | 18,698 | 24.1% | 8 | `/` `J` `Q` `R` `S` `f` `j` | 109 210 213 215 220 223 230 |
| client_02 | 7 | 16,563 | 21.4% | 11 | `J` `S` `e` `j` | 102 105 117 203 207 209 221 |
| client_03 | 6 | 13,946 | 18.0% | 7 | `/` `E` `L` `Q` `S` `a` `e` `f` | 121 122 200 212 222 232 |
| client_04 | 6 | 11,937 | 15.4% | 10 | `E` `L` `S` `a` `e` | 103 104 106 116 124 231 |

The partition is strongly non-IID by label:

- Paced beats (`/`) exist on only 2 clients (client_02: 2,028; client_04: 1,379).
- `L` exists on 2 clients (client_01: 2,491; client_02: 1,457).
- `e` (16) exists only on client_01, `S` (2) only on client_00, and 105 of the 106 `E` beats are on client_02.
- `N` is the majority class on every client.

Full counts and proportions are in `results/tables/federated/fedavg_v1_client_class_counts.csv` and `fedavg_v1_client_class_proportions.csv`. The heatmap is `results/figures/federated/fedavg_v1_client_distribution.png`.

## 4. FedAvg procedure

For round t = 1 … T:

1. **Select clients.** With participation fraction C = 1.0, all K = 5 clients take part. For C < 1 the code samples round(C·K) clients without replacement using `default_rng([seed, t])`.
2. **Broadcast** the full global `state_dict` w_t.
3. **Local update.** Each client k loads w_t, creates a fresh Adam optimizer (lr 0.001), and trains for E = 1 local epoch on its own data with batch size 256 and unweighted cross-entropy. Shuffling and dropout are seeded with `42·100000 + 1000·t + k`. Optimizer state is not transmitted or kept between rounds.
4. **Upload** the local state w_{t+1}^k and the sample count n_k.
5. **Aggregate** with FedAvg: w_{t+1} = Σ_k (n_k / Σ_j n_j) · w_{t+1}^k, computed in float64 and cast back to float32. This applies to every floating-point entry: convolution and linear weights, BatchNorm affine parameters, and BatchNorm running mean and variance. The integer `num_batches_tracked` counters use the same weights and are rounded; they do not affect computation because BatchNorm uses a fixed momentum. No other aggregation rule is implemented.
6. **Validate** w_{t+1} on the validation split.

Round 0 (the initial model, before any training) is also validated, to provide a reference.

## 5. Configuration

| Setting | Value | Rationale |
|---|---|---|
| Seed | 42 | Same as the baseline |
| Model | `ECG1DCNN`, dropout 0.3 | Unchanged `centralized_cnn_v1` architecture |
| Clients K | 5 | Small cross-silo simulation (see §3) |
| Partition | `random_group_equal_count`, seed 42 | Group-level, label-agnostic |
| Communication rounds T | 20 | Gives 20 passes over the training data, enough to see the validation-loss minimum and the later trend |
| Local epochs E | 1 | The simplest FedAvg setting; limits client drift |
| Batch size | 256 | Same as the baseline |
| Optimizer | Adam, lr 0.001, reset each round | Same optimizer and learning rate as the baseline |
| Loss | `CrossEntropyLoss`, unweighted | Same as the baseline |
| Participation C | 1.0 (5 clients per round) | Full participation removes client sampling as a variable |
| Aggregation | FedAvg, weighted by client training samples | Standard FedAvg |
| Model selection | Round with minimum validation loss | Same rule as the baseline's epoch selection |

No hyperparameter search was done. This configuration was fixed before the run and was not changed afterwards. A two-round smoke run, written to a temporary directory **without** test evaluation, was used only to check the pipeline.

## 6. Validation and test protocol

- **During training:** after every round the global model is evaluated on the existing validation split (16,351 segments). The metrics are loss, accuracy, macro-F1, weighted-F1, macro precision, and macro recall, all over the 15 configured classes with `zero_division=0`.
- **Selection:** the round with minimum validation loss becomes the finalised global model (`results/models/federated/fedavg_v1_global.pt`). This turned out to be **round 3**.
- **Test:** the selected global model was evaluated **once** on the held-out test split (15,559 segments, records 100, 101, 107, 113, 214, 219, 233). This used `calculate_metrics` and `ECGTestDataset` from the frozen evaluator, so the metric definitions, OvR specificity, macro and weighted averaging, and AUROC rules are identical to the baseline. `evaluate_test_once` refuses to overwrite an existing test result. The test set was not used for training, round selection, or configuration.

## 7. Results

### Convergence (validation)

| Round | Client train loss (weighted) | Val loss | Val accuracy | Val macro-F1 | Val weighted-F1 | Val macro P | Val macro R |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 2.6343 | 0.6937 | 0.0546 | 0.5682 | 0.0462 | 0.0667 |
| 1 | 1.4695 | 1.5478 | 0.6937 | 0.0546 | 0.5682 | 0.0462 | 0.0667 |
| 2 | 0.6175 | 1.2206 | 0.6878 | 0.0787 | 0.6130 | 0.0668 | 0.1094 |
| **3 (selected)** | 0.4455 | **1.1894** | 0.6912 | 0.0814 | 0.6170 | 0.0687 | 0.1172 |
| 5 | 0.3180 | 1.3090 | 0.6959 | 0.0874 | 0.6245 | 0.0733 | 0.1281 |
| 8 | 0.2486 | 1.4122 | 0.7143 | 0.0913 | 0.6352 | 0.0760 | 0.1295 |
| 12 | 0.2182 | 1.4270 | 0.7166 | 0.0893 | 0.6348 | 0.0742 | 0.1296 |
| 17 | 0.2008 | 1.4498 | 0.6892 | 0.0845 | 0.6285 | 0.0722 | 0.1273 |
| 20 | 0.1922 | 1.6317 | 0.5907 | 0.0774 | 0.5757 | 0.0700 | 0.1180 |

All rounds are in `results/metrics/federated/fedavg_v1_round_history.csv`, and per-client losses are in `fedavg_v1_client_round_history.csv`. Training took 115 s on CPU.

Observations:

- Round 0 and round 1 have identical validation metrics. Accuracy 0.6937 equals the `N` share of the validation set, so these models predict `N` for every segment.
- Validation loss reaches its minimum at round 3 and then rises, while client training loss keeps falling, from 0.4455 to 0.1922. This is the same overfitting pattern seen in the centralized run.
- Validation macro-F1 peaks at 0.0913 (round 8). That is **below the centralized checkpoint's validation macro-F1 of 0.1737 at every round**, so the gap is not only a consequence of choosing round 3.
- Validation accuracy collapses in rounds 18–20, falling to 0.5907.
- The clients are heterogeneous. client_04 has the highest local loss in every round (0.7563 at round 3, against 0.2928 for client_01).
- On validation, the selected model predicts only `N` (13,213), `V` (3,137), and `R` (1).

### Held-out test (evaluated once, round-3 global model)

| Metric | centralized_cnn_v1 | fedavg_v1 | Δ (FedAvg − central) | Verdict |
|---|---:|---:|---:|---|
| Accuracy | 0.7688 | 0.7069 | −0.0620 | degrades |
| Macro precision (15) | 0.1510 | 0.0759 | −0.0751 | degrades |
| Macro recall (15) | 0.1431 | 0.1177 | −0.0254 | degrades |
| Macro sensitivity (8 supported) | 0.2683 | 0.2207 | −0.0476 | degrades |
| **Macro F1 (15)** | **0.1399** | **0.0848** | **−0.0550** | **degrades** |
| Weighted precision | 0.7078 | 0.6364 | −0.0714 | degrades |
| Weighted recall | 0.7688 | 0.7069 | −0.0620 | degrades |
| **Weighted F1** | **0.7208** | **0.6574** | **−0.0634** | **degrades** |
| Macro specificity (15) | 0.9664 | 0.9745 | +0.0082 | matches |
| Weighted specificity | 0.7270 | 0.9114 | +0.1844 | improves |
| Macro OvR AUROC (8 defined) | 0.6238 | 0.6733 | +0.0495 | improves |
| Weighted OvR AUROC | 0.9004 | 0.9686 | +0.0682 | improves |
| Micro OvR AUROC | 0.9624 | 0.9720 | +0.0096 | matches |

"Matches" means |Δ| < 0.01. This is a descriptive band from a single run, not a significance test.

### Per-class test metrics

| Class | Support | Precision (C / F) | Sensitivity (C / F) | Specificity (C / F) | F1 (C / F) | AUROC (C / F) | F1 verdict |
|---|---:|---|---|---|---|---|---|
| `/` | 2,078 | 0.9971 / 0.0000 | 0.5038 / 0.0000 | 0.9998 / 1.0000 | 0.6694 / **0.0000** | 0.9977 / 0.9952 | degrades |
| `A` | 50 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.9999 / 1.0000 | 0.0000 / 0.0000 | 0.7595 / 0.7678 | matches (both zero) |
| `F` | 13 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 0.4548 / 0.4351 | matches (both zero) |
| `L` | 2,001 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.9757 / 1.0000 | 0.0000 / 0.0000 | 0.6593 / 0.9913 | matches (both zero) |
| `N` | 10,197 | 0.8240 / 0.9484 | 0.9934 / 0.9861 | 0.5964 / 0.8980 | 0.9008 / 0.9669 | 0.9333 / 0.9747 | improves |
| `Q` | 4 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 0.1529 / 0.1410 | matches (both zero) |
| `V` | 1,210 | 0.4445 / 0.1902 | 0.6488 / 0.7793 | 0.9316 / 0.7203 | 0.5276 / 0.3058 | 0.8716 / 0.8539 | degrades |
| `a` | 6 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 0.1611 / 0.2275 | matches (both zero) |
| `E` `J` `R` `S` `e` `f` `j` | 0 | 0.0000 / 0.0000 | undefined | ≥0.99 / 1.0000 | 0.0000 / 0.0000 | undefined | not evaluable |

C = centralized, F = FedAvg. Full values are in `results/tables/federated/centralized_vs_fedavg_per_class.csv`.

**Confusion-matrix finding.** On test, the FedAvg global model predicts only two classes: `N` (10,602) and `V` (4,957). Of the 2,078 paced (`/`) beats, 1,900 are predicted as `V` and 178 as `N`. Of the 2,001 `L` beats, 1,961 are predicted as `V`. This is why `/` F1 falls from 0.6694 to 0, why `V` precision falls to 0.1902, and why weighted specificity rises: the model almost never predicts the rare classes, so they receive no false positives. The confusion matrix is in `results/tables/federated/fedavg_v1_confusion_matrix.csv` and the figure in `results/figures/federated/fedavg_v1_confusion_matrix.png`.

**Ranking versus decisions.** Several per-class AUROCs are higher under FedAvg (`L` 0.9913 against 0.6593; `N` 0.9747 against 0.9333). The softmax scores therefore rank `L` beats well even though the argmax decision never chooses `L`. The AUROC gains do **not** mean better classification at the decision rule actually used. They show that class-score information is present but not reflected in argmax predictions.

### Overall verdict

**FedAvg degrades performance** relative to `centralized_cnn_v1` on accuracy (−0.062), weighted-F1 (−0.063), and macro-F1 (−0.055). It loses the paced-beat class entirely (F1 0.6694 → 0) and halves `V` F1. `N` F1 improves slightly (0.9008 → 0.9669). Five classes with test support (`A`, `F`, `L`, `Q`, `a`) have zero F1 under both models. Seven classes cannot be evaluated on this test set.

Possible contributors, stated as **hypotheses rather than established causes**:

- Label non-IID partitioning: `/` and `L` are each held by only 2 of 5 clients and are diluted by averaging.
- Client drift during local epochs.
- Optimizer-state reset each round.
- BatchNorm statistic averaging across heterogeneous clients.
- The same class imbalance and overfitting already seen in the centralized model.

This single run cannot separate these factors.

## 8. Communication cost (analytical, not measured on a network)

| Quantity | Value |
|---|---:|
| Trainable parameters | 10,127 |
| `state_dict` elements transmitted (parameters + BatchNorm running stats + 3 counters) | 10,354 |
| Payload per model transfer (float32, 3 int64 counters) | 41,428 bytes (41.4 kB) |
| In-memory `torch.save` size per model (measured locally) | 47,527 bytes |
| Participating clients per round | 5 (all 20 rounds) |
| Per round: downlink 5 × 41,428 + uplink 5 × 41,428 | 414,280 bytes |
| Total over 20 rounds (downlink 4,142,800 + uplink 4,142,800) | 8,285,600 bytes (≈ 8.29 MB) |
| Up to the selected round 3 | 1,242,840 bytes (≈ 1.24 MB) |

Assumptions:

- The full `state_dict` is sent in both directions every round.
- Each client receives its own copy of the global model (no multicast).
- Tensors are uncompressed native dtypes.
- Transport, protocol, serialization, and encryption overhead are excluded.

All values are calculated for a simulated federation; no network traffic was measured. For comparison, the raw training ECG never leaves the clients. The 77,550 float32 training windows alone would be about 134 MB.

## 9. Reproducibility

- Seed 42 everywhere, `torch.use_deterministic_algorithms(True)`, and deterministic per-client seeds.
- Re-running the first 3 rounds into a temporary directory, without test evaluation, reproduced the saved round-3 global model **bit-identically**. The partition and the round 0–3 validation history were also identical.
- Software: Python 3.12.10, PyTorch 2.14.1+cpu, NumPy 2.5.3, scikit-learn 1.9.1, CPU with 8 threads.
- The frozen centralized artifacts (checkpoint, Day 6 metrics, predictions, split metadata, configs, and `src/models`) were checked by SHA-256 before and after the run and are unchanged.

## 10. Limitations

1. **Simulation only.** The clients are partitions of one public database, not institutions. There is no real network, heterogeneous hardware, or client dropout.
2. **No privacy or security.** Updates are plaintext and no DP, HE, or secure aggregation is used. Clients are assumed honest. Nothing here supports a privacy claim.
3. **Single run.** One seed, one partition, and one configuration, with no repeated runs or confidence intervals. The "matches" band is descriptive.
4. **Training budgets are not matched.** The selected FedAvg model has had 3 local epochs per client (about 3 passes over the training data). The centralized checkpoint had 1 epoch with continuous Adam state. Both used the same validation-loss selection rule, and FedAvg's validation macro-F1 stays below the centralized value in all 20 rounds, but the budgets differ.
5. **All baseline limitations carry over.**
   - Severe imbalance.
   - Only 8 of 15 classes have test support.
   - A small fixed 7-record test set.
   - 47 inferred subject-equivalence groups, which are not verified patient IDs.
   - A provisional 15-symbol taxonomy.
   - Lead heterogeneity across records.
6. **Partition effects are confounded.** A label-agnostic random group partition produced extreme label skew because MIT-BIH classes are concentrated in a few records. The results reflect this specific partition.
7. **No clinical claim.** Nothing here shows clinical usefulness, deployment readiness, or diagnostic validity.

## 11. Artifacts

- **Model:** `results/models/federated/fedavg_v1_global.pt` (round-3 global state, config, partition)
- **Metrics:** `results/metrics/federated/`
  - `fedavg_v1_round_history.csv`
  - `fedavg_v1_client_round_history.csv`
  - `fedavg_v1_test_metrics.json`
  - `fedavg_v1_test_predictions.npz`
  - `fedavg_v1_classification_report.txt`
  - `fedavg_v1_roc_metrics.csv`
  - `fedavg_v1_communication_cost.json`
  - `fedavg_v1_client_partition.json`
  - `fedavg_v1_run_summary.json`
- **Tables:** `results/tables/federated/`
  - `fedavg_v1_client_summary.csv`
  - `fedavg_v1_client_class_counts.csv`
  - `fedavg_v1_client_class_proportions.csv`
  - `fedavg_v1_per_class_metrics.csv`
  - `fedavg_v1_confusion_matrix.csv`
  - `centralized_vs_fedavg_aggregate.csv`
  - `centralized_vs_fedavg_per_class.csv`
- **Figures:** `results/figures/federated/`
  - `fedavg_v1_convergence.png`
  - `fedavg_v1_client_losses.png`
  - `fedavg_v1_client_distribution.png`
  - `fedavg_v1_confusion_matrix.png`
- **Tests:** `tests/test_federated.py`
