# Day 18 — Proposed robust aggregation for FedAvg

- **Date:** 2026-10-05
- **Configuration:** [`configs/proposed/robust_defense_v1.json`](../configs/proposed/robust_defense_v1.json) (frozen parameters), calibrated from [`configs/proposed/calibration_v1.json`](../configs/proposed/calibration_v1.json)
- **Code:**
  - `src/federated/proposed_defense.py`: the method
  - the `defense` branch in `src/federated/run_fedavg.py`
  - `src/federated/proposed_study.py`: evaluation and report
- **Outputs:** `results/proposed/`
- **Tests:** `tests/test_proposed_defense.py`

> This is a **proposed, candidate** defense evaluated in one simulated setting. It combines established ideas (update clipping, similarity and norm screening against a robust reference, robust statistics) into one design. No novelty or superiority claim is made. DP, HE, and blockchain are not integrated.

## 1. Method (motivated by the Day 17 findings)

Day 17 found that plain FedAvg fails through **unchecked update magnitude and direction**, and specifically through **negative aggregated BatchNorm running variances**, which produce NaN outputs. The proposed aggregation addresses both. It runs on the server each round:

| Step | Operation |
|---|---|
| 1. Updates | Δ_k = w_k − w_t over **trainable parameters only**: convolution and linear weights and biases, plus BatchNorm affine weight and bias (10,127 values). |
| 2. Clipping | Δ_k ← Δ_k · min(1, C / ‖Δ_k‖₂), with fixed C. |
| 3. Similarity | cos_k = cos(Δ_k, m), where m is the **coordinate-wise median** of the client updates (the robust reference direction). |
| 4. Screening | Norm ratio r_k = ‖Δ_k‖ / median_j ‖Δ_j‖. A client is **flagged** if cos_k < τ_cos or r_k > τ_norm. Flagged clients get weight 0, and the prior FedAvg weights n_k / Σn are renormalised over accepted clients. |
| 5. Aggregation | w_{t+1} = w_t + Σ_k w_k · clip(Δ_k) on the trainable parameters. |
| 6. BatchNorm statistics | Handled **separately**: running mean and variance are the coordinate-wise median of the **accepted** clients' reported statistics, and variances are floored at 10⁻⁵ (BatchNorm ε). Integer counters are aggregated as in FedAvg. |
| Fallback | If every client is flagged, use the coordinate-wise median of the clipped updates (logged). This never occurred. |

The method uses only what FedAvg already sends, adds **no client computation and no communication**, and is deterministic.

## 2. Parameter calibration (no test or validation tuning)

The parameters were derived by **predeclared rules** from the honest client statistics of one **clean FedAvg run with seed 7**. Seed 7 is not an evaluation seed. The run was in observe mode: statistics were logged, plain FedAvg aggregation was applied, and no test evaluation was done. The parameters were frozen before any evaluation run.

| Parameter | Rule | Honest calibration statistic (100 client-rounds) | Value |
|---|---|---|---|
| C | 95th percentile of honest update norms | norms 1.17–3.13 | **2.603** |
| τ_norm | max(2.0, 1.5 × maximum honest norm ratio) | ratio ≤ 1.2545 | **2.0** |
| τ_cos | 0.5 × minimum honest cosine to the median | cosine 0.363–0.730 | **0.1814** |
| BatchNorm variance floor | Fixed a priori, equal to BatchNorm ε | — | 10⁻⁵ |

The validation split was used only by the unchanged, predefined round-selection rule (minimum validation loss). The test set was evaluated once per run.

## 3. Evaluation design

- **Conditions:** 4 conditions (clean, plus the exact Day 17 sign flip, scaled ×10, and norm-matched noise attacks, with the same attacker clients and seeds) × 3 seeds (42, 123, 2024), giving 12 defended runs.
- **Budget:** identical to Days 11 and 17 (natural 5-client partition, 20 rounds, 1 local epoch, Adam lr 0.001, batch size 256, full participation).
- **Comparators (read-only):**
  - clean FedAvg: the Day 11 runs;
  - attacked FedAvg without defense: the Day 17 runs.
- **Global-model deviation:** measured against clean FedAvg trajectories (the Day 16 archive).

## 4. Results

### Test metrics (mean ± sample SD over 3 seeds; descriptive, no significance test)

| Condition | FedAvg, no defense: accuracy | Macro-F1 | Weighted-F1 | **Proposed defense: accuracy** | **Macro-F1** | **Weighted-F1** | Macro recall | Macro AUROC |
|---|---|---|---|---|---|---|---|---|
| Clean | 0.657 ± 0.076 | 0.085 ± 0.013 | 0.602 ± 0.049 | 0.625 ± 0.097 | 0.070 ± 0.012 | 0.569 ± 0.066 | 0.095 ± 0.022 | 0.654 ± 0.023 |
| Sign flip | 0.680 ± 0.024 | 0.082 ± 0.000 | 0.597 ± 0.021 | 0.673 ± 0.025 | 0.069 ± 0.020 | 0.557 ± 0.037 | 0.082 ± 0.021 | 0.623 ± 0.045 |
| **Scaled ×10** | **0.463 ± 0.334** | **0.038 ± 0.025** | **0.350 ± 0.293** | **0.673 ± 0.025** | **0.069 ± 0.020** | **0.557 ± 0.037** | 0.082 ± 0.021 | 0.623 ± 0.045 |
| Random noise | 0.691 ± 0.025 | 0.088 ± 0.005 | 0.608 ± 0.039 | 0.686 ± 0.027 | 0.075 ± 0.020 | 0.577 ± 0.052 | 0.093 ± 0.023 | 0.628 ± 0.031 |

Per-seed values are in `results/proposed/tables/proposed_runs.csv`. Under the defense, sign flip and scaled give **bit-identical** results per seed. The attacker is excluded in every round in both cases, so training reduces to the same 4 honest clients.

### Detection and client weights

| Condition | Attack detection rate (attacker rounds flagged) | Attacker weight: FedAvg prior → defended (mean) | False-positive rate on honest clients | Honest updates clipped |
|---|---|---|---|---|
| Clean | — | — | **0 / 300** client-rounds | 5–6% |
| Sign flip | **20/20 in every seed (100%)**, flagged by cosine | 0.154 / 0.164 / 0.220 → **0** | **0%** | 5.0–6.3% |
| Scaled ×10 | **20/20 in every seed (100%)**, flagged by norm ratio | 0.154 / 0.164 / 0.220 → **0** | **0%** | 5.0–6.3% |
| Random noise | **1/20, 4/20, 2/20 (5–20%)** | 0.154 / 0.164 / 0.220 → 0.146 / 0.131 / 0.198 | **0%** | 5.0–6.3% |

Across all 12 runs, **no honest client was ever flagged**: 0 of 1,020 honest client-rounds, attacked and clean. The per-round weights before and after the defense, for every client, are in `proposed_client_weights.csv`. The figure is `proposed_client_weights.png`.

**Why norm-matched noise mostly evades screening.** The noise attacker's cosine to the median reference is **0.19–0.28**, not about 0. The coordinate-wise median **includes the attacker's own update**: in each coordinate, the large noise value pulls the median towards whichever honest value lies on the noise's side, so the reference correlates positively with the noise. Honest cosines in the evaluation runs reach as low as 0.203, so with self-inclusion no cosine threshold separates the two cleanly. The noise was flagged mainly in round 1, where its norm ratio exceeded 2. The threshold was **not** re-tuned after seeing this, because that would be tuning on evaluation results. A leave-one-out median reference is the principled fix and is left for future work. Day 17 showed this attack is largely absorbed by averaging, and the defended noise results are within seed variability.

### Breakdown prevention, final-round behaviour, and deviation

| Condition | NaN-validation rounds = negative-variance rounds (no defense → defense) | Final-round validation macro-F1 (no defense → defense) | Relative deviation from clean FedAvg at round 20 (no defense → defense) |
|---|---|---|---|
| Clean | — → 0 | 0.069–0.091 (FedAvg) → 0.068–0.076 | 0 → 0.06–0.10 |
| Sign flip | 7 / 15 / 15 → **0 / 0 / 0** | 0.0115 (broken) → **0.072–0.092** | 0.34–0.51 → 0.19–0.27 |
| Scaled ×10 | 10 / 17 / 10 → **0 / 0 / 0** | 0.0006–0.055 (broken) → **0.072–0.092** | 2.5·10² to 4.0·10⁵ → **0.19–0.27** |
| Random noise | 0 → 0 | 0.077–0.093 → 0.073–0.095 | 0.24–0.31 → 0.20–0.26 |

The final-round FedAvg validation values come from the Day 11 clean histories. No BatchNorm variance had to be floored in any defended run: the median of accepted clients' statistics was always positive.

### Overhead

| Quantity | Value |
|---|---|
| Server computation (defense per round, 5 clients) | 3.6–3.8 ms (about 0.07 s per 20-round run, about 0.07% of the 108–111 s training time) |
| Extra communication | **0 bytes** (no new messages; same payloads as FedAvg) |
| Client computation | none added |
| Memory | one stacked 5 × 10,127 update matrix per round on the server |

## 5. Interpretation

1. **Effective against the two damaging Day 17 attacks.** Sign flip and scaled ×10 were detected in 100% of attacker rounds with no false positives. The attacker's weight was driven to 0, and the BatchNorm breakdown was eliminated: 0 NaN rounds against 7–17 without the defense. The model stayed valid through round 20: final-round validation macro-F1 was 0.07–0.09 against about 0.01 for broken undefended models. Under the scaled attack, test accuracy rose from 0.463 ± 0.334 to 0.673 ± 0.025 and macro-F1 from 0.038 to 0.069. All three degenerate single-class models became 1–3-class models.
2. **Sign-flip selected-round metrics do not improve, and that is expected.** Undefended sign flip looked near clean on test (macro-F1 0.082) only because validation selection picked rounds 3–4, before the breakdown (Day 17). The defended runs score 0.069 ± 0.020 on test, but **they stay valid in every round**. The defense's benefit is robustness of the training process and of the final model, not a higher selected-round score.
3. **Utility cost on clean data.** With no attack and no false positives, the defended clean runs average macro-F1 0.070 against 0.085 for FedAvg (−0.015; per seed −0.002, −0.040, −0.004) and accuracy 0.625 against 0.657. Because nothing was flagged, the cost must come from clipping (5–6% of honest updates) and/or median BatchNorm statistics replacing the weighted mean. The design does not separate the two; an ablation is needed. The seed-123 drop is the largest single effect.
4. **Weak against stealthy, magnitude-matched noise** (5–20% detection), because of the self-inclusive median reference. The impact was small in this setting.
5. **Practically cheap.** About 4 ms per round on the server and no extra communication, so it is compatible in principle with the DP, HE, and blockchain components. Compatibility was **not** tested: in particular, clipping and screening under HE would require the server to see similarities, which conflicts with encrypted aggregation.

## 6. Limitations

1. 3 seeds, one attacker, the 3 Day 17 attack settings. No adaptive attacker (for example one tuned to stay just under τ_norm with cosine above τ_cos), no colluding attackers, no data, label, or backdoor poisoning.
2. Thresholds were calibrated on one clean seed. Honest cosines in the evaluation runs (minimum 0.203) came close to τ_cos = 0.181, so with more heterogeneous clients false positives are plausible.
3. The median reference includes the evaluated client (the noise weakness above).
4. The clean-utility cost is not attributed to clipping versus median BatchNorm statistics (no ablation).
5. Hard rejection of one in five clients removes 15–22% of the data each round. With multiple honest-but-unusual clients this could hurt minority classes held by few clients (Day 11: `/` and `L` are on only 2 clients).
6. Not combined with DP, HE, or blockchain. Robust screening conflicts with HE confidentiality and interacts with DP noise.
7. Inherited limitations: a simulation on public data, a weak and collapsed clean baseline (macro-F1 about 0.085), and a 7-record test set.

## 7. Integrity

- Without a `defense` section, the code still reproduces the official Day 10 model bit-identically.
- All 904 pre-existing files under `results/`, `data/splits/`, `configs/`, and `tests/` are byte-identical after Day 18, and the processed data is unchanged.
- **Not tracked (regenerable):** per-run checkpoints, `.npz` predictions, and trajectories.

## 8. Artifacts

- **Tables:** `results/proposed/tables/`
  - `proposed_runs.csv` (all per-run metrics, detection, false positives, breakdown, deviation, and overhead)
  - `proposed_summary.csv`
  - `proposed_client_weights.csv` (every client, every round, prior and defended weight, flags)
  - `proposed_global_deviation_by_round.csv`
- **Calibration:** `results/proposed/calibration/` (seed 7, observe mode, no test evaluation)
- **Per run** (`defended_{clean,sign_flip,scaled,random_noise}_s{42,123,2024}`):
  - `*_defense_client_diagnostics.csv`
  - `*_defense_round_diagnostics.csv`
  - round history, test metrics, confusion matrix
- **Figures:** `results/proposed/figures/proposed_vs_attacks_test_metrics.png` and `proposed_client_weights.png`
