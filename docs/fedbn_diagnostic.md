# Day 12 — BatchNorm diagnostic (FedBN-style local BatchNorm)

- **Date:** 2026-10-05
- **Configuration:** [`configs/federated/fedbn_diagnostic_v1.json`](../configs/federated/fedbn_diagnostic_v1.json), which uses the unchanged base [`fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Code:**
  - `aggregation.batchnorm = "local"` in `src/federated/run_fedavg.py`
  - `batchnorm_state_keys` and `client_start_state` in `src/federated/fedavg.py`
  - the paired comparison in `src/federated/diagnostics.py`
- **Outputs:** `results/diagnostics/fedbn/`

> This is a one-factor diagnostic of the Day 10/11 FedAvg degradation. It is not an attempt to optimise federated performance, and it is not the proposed algorithm.

## 1. Question

Days 10 and 11 showed that plain FedAvg falls below `centralized_cnn_v1` and effectively collapses to `N`/`V` predictions. Averaging BatchNorm across clients with very different label mixes was listed as one candidate contributor. This diagnostic asks:

**If each client keeps its own BatchNorm layers instead of averaging them, does the minority-class collapse change materially?**

## 2. Design: one controlled factor

| | Day 11 standard FedAvg (comparator) | Day 12 FedBN-style diagnostic |
|---|---|---|
| Conv1d and Linear weights/biases | Sample-weighted FedAvg, broadcast | **Same** |
| BatchNorm weight, bias, running mean/variance, counter | Sample-weighted FedAvg, broadcast to all clients | **Kept local:** each client starts every round from its own BatchNorm state from the previous round |
| Partition | Natural `random_group_equal_count`, 5 clients | **Same** (identical per seed; verified) |
| Seeds | 42, 123, 2024 (seed sets partition and training seeds) | **Same** |
| Training | 20 rounds, 1 local epoch, Adam lr 0.001 reset each round, batch 256, full participation | **Same** |
| Round selection | Minimum validation loss of the evaluated global model | **Same** |
| Test | Evaluated once per run, frozen evaluator metrics | **Same** |

Because seed, partition, initialisation, shuffling and dropout seeds are all shared, each FedBN run is **paired** with the Day 11 natural FedAvg run that has the same seed. The BatchNorm handling is the only configured difference. A test confirms that the two configurations differ only in `aggregation.batchnorm`.

**Interpretation of "keep BatchNorm local".** All BatchNorm-owned state stays on the client: the affine scale and shift, the running mean and variance, and the counter. This is the FedBN arrangement. All non-BatchNorm parameters continue through FedAvg.

**Evaluation model.** With local BatchNorm there is no single global model, and the validation and test records belong to no client. To keep the Day 11 protocol unchanged (one model per round, the same selection rule, one test evaluation), the evaluated model is the shared FedAvg weights plus the sample-weighted average of the clients' BatchNorm states.

- This average is computed only for evaluation. It is **never broadcast** back to clients.
- Under standard FedAvg, the same construction is exactly the broadcast global model, so the evaluation pipeline is identical in both arms.
- The design choice matters. A held-out subject has no "local" BatchNorm, so any single-model evaluation must combine client BatchNorm states somehow.

**Validation-only secondary diagnostic.** After every round, each client's *personalised* model (shared weights plus that client's own BatchNorm) is also evaluated on validation (`*_client_validation_history.csv`). Personalised models were **not** evaluated on test, so the test set is still used once per run.

**Implementation checks.**

- The FedBN checkpoint stores 5 distinct client BatchNorm states. The BatchNorm affine parameters differ between clients by up to 0.17, and the running means by up to 0.57 (seed 42).
- The FedBN evaluation model differs from the paired FedAvg model.
- The partition is identical between the arms.
- With `batchnorm` left at its default, the code reproduced the official Day 10 round-3 model bit-identically (re-run into a temporary directory with no test evaluation).

## 3. Results (natural partition, n = 3 paired seeds)

### Per seed

| Metric | s42 FedAvg | s42 FedBN | s123 FedAvg | s123 FedBN | s2024 FedAvg | s2024 FedBN |
|---|---:|---:|---:|---:|---:|---:|
| Selected round | 3 | 3 | 2 | 2 | 17 | 12 |
| Best validation loss | 1.1894 | 1.1954 | 1.3481 | 1.3477 | 1.2538 | 1.1513 |
| Validation accuracy | 0.6912 | 0.6882 | 0.6263 | 0.6286 | 0.7124 | 0.7165 |
| Validation macro-F1 | 0.0814 | 0.0801 | 0.0517 | 0.0518 | 0.0913 | 0.0912 |
| Validation weighted-F1 | 0.6170 | 0.6150 | 0.5347 | 0.5359 | 0.6380 | 0.6399 |
| **Test accuracy** | 0.7069 | 0.7031 | 0.6943 | 0.6928 | 0.5690 | 0.5579 |
| **Test macro-F1** | 0.0848 | 0.0844 | 0.0986 | 0.0974 | 0.0717 | 0.0706 |
| **Test weighted-F1** | 0.6574 | 0.6542 | 0.5841 | 0.5822 | 0.5648 | 0.5519 |
| **Test macro recall** | 0.1177 | 0.1172 | 0.1002 | 0.0988 | 0.1052 | 0.1049 |
| Test macro AUROC (defined classes) | 0.6733 | 0.6694 | 0.6463 | 0.6596 | 0.6226 | 0.6451 |
| Test weighted AUROC | 0.9686 | 0.9655 | 0.7538 | 0.7524 | 0.8652 | 0.8749 |
| Classes predicted on test | 2 | 2 | 3 | 3 | 5 | 3 |

Centralized reference (`centralized_cnn_v1`): test accuracy 0.7688, macro-F1 0.1399, weighted-F1 0.7208, macro recall 0.1431, macro AUROC 0.6238. The tables are `results/diagnostics/fedbn/tables/fedbn_vs_fedavg_paired.csv` and `robustness_runs.csv`.

### Paired differences (FedBN − FedAvg; descriptive, no significance test)

| Metric | Mean Δ | Range of Δ | Seeds FedBN higher / lower |
|---|---:|---|---|
| Test accuracy | −0.0055 | −0.0111 to −0.0015 | 0 / 3 |
| Test macro-F1 | −0.0009 | −0.0012 to −0.0004 | 0 / 3 |
| Test weighted-F1 | −0.0060 | −0.0129 to −0.0019 | 0 / 3 |
| Test macro recall | −0.0007 | −0.0013 to −0.0004 | 0 / 3 |
| Test macro AUROC | +0.0107 | −0.0039 to +0.0226 | 2 / 1 |
| Validation macro-F1 | −0.0004 | −0.0012 to +0.0001 | 1 / 2 |

All test decision-metric differences are small. The largest macro-F1 change is 0.0012. For scale, Day 11 test macro-F1 for FedAvg ranged over 0.0269 across seeds, and the gap to centralized is at least 0.041. The consistent sign (FedBN lower in 3 of 3 seeds) comes from only 3 pairs and is **not** evidence of a real effect.

### Prediction counts on test (from the confusion matrices)

| Seed | Arm | `/` | `L` | `N` | `V` | `R` | Share `N` or `V` | Classes predicted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | FedAvg | 0 | 0 | 10,602 | 4,957 | 0 | 100.00% | 2 |
| 42 | FedBN | 0 | 0 | 10,589 | 4,970 | 0 | 100.00% | 2 |
| 123 | FedAvg | 0 | 0 | 14,910 | 613 | 36 | 99.77% | 3 |
| 123 | FedBN | 0 | 0 | 14,936 | 589 | 34 | 99.78% | 3 |
| 2024 | FedAvg | 1 | 2 | 8,721 | 6,831 | 4 | 99.96% | 5 |
| 2024 | FedBN | 0 | 5 | 8,722 | 6,832 | 0 | 99.97% | 3 |

`A`, `F`, `Q`, `a`, and every other class are never predicted by either arm. The centralized model predicts `/` 1,050 times and `L` 329 times. The table is `fedbn_vs_fedavg_prediction_counts.csv`.

### Per-class (supported test classes)

| Class (support) | FedAvg F1, s42 / s123 / s2024 | FedBN F1, s42 / s123 / s2024 | Centralized F1 |
|---|---|---|---:|
| `/` (2,078) | 0.000 / 0.000 / 0.001 | 0.000 / 0.000 / 0.000 | 0.669 |
| `L` (2,001) | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| `N` (10,197) | 0.967 / 0.812 / 0.833 | 0.962 / 0.811 / 0.813 | 0.901 |
| `V` (1,210) | 0.306 / 0.667 / 0.243 | 0.305 / 0.649 / 0.246 | 0.528 |
| `A` `F` `Q` `a` | 0 in all runs | 0 in all runs | 0 |

Recall for `/`, `L`, `A`, `F`, `Q`, and `a` is 0.000 in every FedBN run. The FedBN run for seed 2024 makes 5 `L` predictions, all wrong; the precision, recall, and specificity for every pair are in `fedbn_vs_fedavg_per_class.csv`.

### Personalised client models (validation only, at the selected round)

Each client's own-BatchNorm model reaches validation macro-F1 between 0.037 and 0.092 (`fedbn_vs_fedavg_personalised_validation.csv`).

- **Seeds 42 and 2024:** one personalised model each slightly exceeds the averaged-BatchNorm evaluation model (client_02: 0.0816 against 0.0801; client_04: 0.0916 against 0.0912). The other four are below it.
- **Seed 123:** the evaluation model is weak (0.0518), and 4 of 5 personalised models exceed it, the best being client_03 at 0.0750.
- **Accuracy:** several personalised models have markedly lower validation accuracy (0.24–0.46). No personalised model approaches the centralized validation macro-F1 of 0.1737. Using a client's own BatchNorm on held-out records therefore does not uncover hidden minority-class ability.

### Convergence

The figure is `results/diagnostics/fedbn/figures/robustness_convergence.png`. FedBN's validation curves follow the paired FedAvg curves closely. Selected rounds are identical for seeds 42 and 123 (3 and 2). For seed 2024 FedBN selects round 12 (FedAvg: 17), with a lower best validation loss (1.1513 against 1.2538) but no better macro-F1.

### Communication

BatchNorm state is 1,816 of the 41,428 bytes per model (4.4%). Under FedBN, the downlink needs only the shared 39,612 bytes, because clients keep their own BatchNorm. The uplink here still includes BatchNorm, because the averaged-BatchNorm evaluation model needs it. These are analytical byte counts (`results/diagnostics/fedbn/metrics/fedbn_payload.json`), not network measurements.

## 4. Why the intervention is weaker than it sounds

BatchNorm running statistics are updated with momentum 0.1. Each client runs 47–74 batches per local epoch (11,937–18,698 segments, batch size 256). After one local epoch, the running statistics a client started from keep at most **0.9^47 ≈ 0.7%** of their weight. In **both** arms, each client's running statistics are therefore re-estimated almost entirely from its own data every round, and the averaged statistics used for evaluation are formed the same way in both arms.

In this configuration, keeping running statistics local is close to a no-op. The effective manipulated factor is mainly whether the BatchNorm **affine scale and shift** are shared or local. A setting with fewer local batches per round, or with running statistics frozen, could behave differently. Neither was tested.

## 5. Interpretation

- **The BatchNorm-only intervention does not materially change the minority-class collapse.** In all 3 paired seeds, FedBN-style training still predicts only `N`/`V` (plus at most 34 spurious `R` or 5 wrong `L`), still detects no paced or `L` beats, and still sits far below the centralized baseline on every decision metric. Paired differences are an order of magnitude smaller than seed-to-seed variability.
- **What this supports.** In this setup, sharing versus localising BatchNorm layers during training is not, on its own, enough to reverse the degradation. For these seeds and this configuration, BatchNorm averaging does not appear to be the main driver of the collapse.
- **What this does not support.**
  - It does not rule out BatchNorm as a contributor in other regimes (see §4).
  - It does not show that any other candidate (label skew of `/` and `L`, client drift, Adam reset, training budget, selection criterion) is the cause.
  - It does not establish anything beyond these 3 seeds, this partition scheme, and this model.

No causal claim beyond "this one change did not materially alter the outcome here" is made.

## 6. Limitations

1. **Three paired seeds.** Differences are descriptive and there is no significance test.
2. **Evaluation design.** FedBN has no natural single model for unseen subjects. The averaged-BatchNorm evaluation model is a design choice that matches the Day 11 protocol. The personalised-model check is validation-only.
3. **Weak manipulation of running statistics.** Under 1 local epoch of 47–74 batches, the running statistics are re-estimated each round in both arms, so the effective difference is BatchNorm affine locality.
4. **Inherited limitations.** All Day 10/11 limitations still apply:
   - the simulation is on public data;
   - there is no privacy mechanism or blockchain;
   - the test set has 7 records with 8 of 15 classes supported;
   - the groups are inferred subject-equivalence groups;
   - `/` and `L` are confined to 2 training records each.

## 7. Integrity

- All pre-existing files under `results/`, `data/splits/`, `configs/`, and `tests/` (257 files) are byte-identical after Day 12. This includes the centralized artifacts, the official Day 10 `fedavg_v1` model and test predictions, and all Day 11 robustness outputs. The processed data is unchanged, and all new outputs are under `results/diagnostics/fedbn/`.
- The default (`aggregate`) code path reproduces the official Day 10 model bit-identically.
- Per-run FedBN checkpoints and `.npz` prediction files are listed in `.gitignore` as regenerable, following the Day 11 convention.

## 8. Artifacts

- **Per run** (`fedbn_v1_natural_s{42,123,2024}`) under `results/diagnostics/fedbn/{metrics,tables,figures}/federated/`:
  - round history and client-round history
  - client validation history (personalised models)
  - test metrics, per-class metrics, and confusion matrix
  - ROC metrics, classification report, partition, run summary, and comparison with the centralized baseline
- **Aggregate tables:** `results/diagnostics/fedbn/tables/`
  - `robustness_*.csv`
  - `fedbn_vs_fedavg_paired.csv`
  - `fedbn_vs_fedavg_paired_summary.csv`
  - `fedbn_vs_fedavg_prediction_counts.csv`
  - `fedbn_vs_fedavg_per_class.csv`
  - `fedbn_vs_fedavg_personalised_validation.csv`
- **Figures:** `results/diagnostics/fedbn/figures/`
  - `fedbn_vs_fedavg_paired.png`
  - `robustness_convergence.png`
  - `robustness_test_metrics.png`
- **Tests:** `tests/test_federated_fedbn.py`
