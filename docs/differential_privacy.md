# Day 13 — Client-level differential privacy (DP-FedAvg)

- **Date:** 2026-10-05
- **Configuration:** [`configs/privacy/dp_fedavg_v1.json`](../configs/privacy/dp_fedavg_v1.json), which uses the unchanged base [`fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Code:** `src/federated/privacy.py` (mechanism and accountant), the `privacy` branch in `src/federated/run_fedavg.py`, and `src/federated/privacy_report.py`
- **Outputs:** `results/privacy/`

> This is a privacy-only intervention on the frozen standard FedAvg setup. It measures the utility cost of one declared DP configuration. It is not tuned for utility and is not the proposed algorithm.

## 1. Mechanism

DP-FedAvg at the client level (McMahan et al., 2018), fixed-weight variant, with a **trusted server**. In each round t, every participating client k trains locally as before. Then:

```text
Δ_k      = w_k − w_t                              (all floating-point state entries)
clip(Δ_k) = Δ_k · min(1, C / ‖Δ_k‖₂)              (per-client L2 clipping)
w_{t+1}  = w_t + Σ_k p_k · clip(Δ_k) + N(0, (z · C · max_k p_k)² · I)
```

- **Aggregation weights.** p_k = n_k / Σn are the usual FedAvg sample weights. They are treated as **public and fixed**.
- **Sensitivity.** Adding or removing one client's contribution (other weights fixed) changes the weighted sum by at most S = C · max_k p_k, which is 0.232–0.241 · C for these partitions. Each round is therefore a Gaussian mechanism with noise multiplier z.
- **What is clipped and noised.** Every floating-point state entry: convolution and linear weights and biases, BatchNorm affine parameters, **and BatchNorm running statistics**. The running statistics depend on the data, so they must be privatised.
- **What is not noised.** The integer BatchNorm counters are aggregated as in FedAvg; they depend only on the public sample counts.
- **Post-processing.** Clamping noisy running variances to be at least 0 is post-processing and does not affect the guarantee.
- **Noise generation.** Noise comes from a dedicated, seeded stream per round (`seed · 1000003 + 7919 + round`) so runs are reproducible.

Everything else is identical to Day 11, and each run is paired by seed with the Day 11 natural FedAvg run:

- natural 5-client whole-group partition, seeds 42, 123, and 2024;
- 20 rounds, 1 local epoch, Adam lr 0.001, batch size 256;
- full participation;
- round selection by minimum validation loss over rounds 0–20, and one test evaluation per run.

A test confirms the configurations differ only by the `privacy` section. With privacy disabled, the code still reproduces the official Day 10 model bit-identically.

## 2. Privacy parameters and accounting

| Parameter | Value |
|---|---|
| Privacy unit | **One client**: all training data of one simulated institution. Add/remove-one-client adjacency with fixed public weights. |
| Clipping norm C | 1.0, fixed a priori; not tuned on client, validation or test data |
| Noise multiplier z | 2.69: the smallest two-decimal value meeting the target (z ≥ 2.6843 is required) |
| Target | ε ≤ 8 at δ = 10⁻⁵, declared before running |
| δ | 10⁻⁵ |
| Client sampling rate q | 1.0 (full participation, 5 of 5 clients every round) |
| Rounds T | 20; every broadcast global model is counted as released |
| Accountant | μ-GDP composition (Dong, Roth & Su, 2019): T Gaussian mechanisms compose **exactly** to μ = √T / z = 1.6625. ε comes from the analytic Gaussian conversion (Balle & Wang, 2018). |
| **Resulting ε** | **7.98** at δ = 10⁻⁵, the same for all seeds |
| RDP cross-check | ε ≤ 9.36 (Mironov 2017; optimal order α ≈ 3.89). This is an upper bound and is looser than GDP, as expected. |
| Seeds | Training and partition: 42 / 123 / 2024. Noise: `seed · 1000003 + 7919 + round`. |

**Why an implemented accountant.** No DP library (Opacus, dp-accounting, TF Privacy, prv-accountant) is installed. With q = 1 there is no subsampling, so the GDP composition is exact rather than approximate. The accountant is unit-tested against closed-form values: δ(0) = 2Φ(μ/2) − 1, and the root of δ(ε) = 10⁻⁵ is recovered. It also rejects q < 1.

**What the guarantee covers, and what it does not.**

- It is **central DP against everyone except the server**: other clients and anyone who sees the released global models. Clients send clipped updates in plaintext to a server that is trusted to add noise. There is **no** protection against the server, and no secure aggregation.
- It is client-level, which is stronger than record-level: it bounds what can be learned about whether an entire simulated institution participated.
- **Round selection** uses only the validation split, which is not client data. Choosing among released models is post-processing, so it adds no privacy cost.
- **Logged diagnostics are outside the guarantee.** The per-round unclipped update norms and clip flags in `*_dp_*` files are for analysis only. A deployment would not release them.
- **Idealised sampling assumed.** The guarantee assumes ideal Gaussian sampling. The seeded PRNG used for reproducibility is not cryptographically secure, and floating-point effects are not modelled.

## 3. Results (test, evaluated once per run)

| Seed | Selected round | ε (δ = 10⁻⁵) | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Macro AUROC | Predicted class distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | **0** | 7.98 | 0.6554 | 0.0437 | 0.0667 | 0.0528 | 0.5189 | 0.4244 | `N` 15,559 |
| 123 | **0** | 7.98 | 0.0778 | 0.0052 | 0.0667 | 0.0096 | 0.0112 | 0.5823 | `V` 15,559 |
| 2024 | **0** | 7.98 | 0.0071 | 0.0126 | 0.0726 | 0.0081 | 0.0094 | 0.6320 | `Q` 9,217, `e` 5,773, `V` 567, `f` 2 |

**Paired utility degradation relative to standard FedAvg (same seed):**

| Seed | Δ accuracy | Δ macro P | Δ macro R | Δ macro F1 | Relative Δ macro F1 | Δ weighted F1 | Δ macro AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | −0.0515 | −0.0322 | −0.0510 | −0.0321 | −38% | −0.1385 | −0.2489 |
| 123 | −0.6166 | −0.1065 | −0.0335 | −0.0890 | −90% | −0.5729 | −0.0640 |
| 2024 | −0.5619 | −0.1238 | −0.0327 | −0.0637 | −89% | −0.5554 | +0.0094 |

**3-seed summary (mean ± sample SD; descriptive, no significance test):**

| Metric | DP-FedAvg | Standard FedAvg | Centralized |
|---|---|---|---:|
| Accuracy | 0.2468 ± 0.3556 | 0.6567 ± 0.0762 | 0.7688 |
| Macro precision | 0.0205 ± 0.0204 | 0.1080 ± 0.0304 | 0.1510 |
| Macro recall | 0.0686 ± 0.0034 | 0.1077 ± 0.0090 | 0.1431 |
| Macro F1 | 0.0235 ± 0.0254 | 0.0851 ± 0.0134 | 0.1399 |
| Weighted F1 | 0.1798 ± 0.2937 | 0.6021 ± 0.0489 | 0.7208 |
| Macro AUROC (defined classes) | 0.5462 ± 0.1084 | 0.6474 ± 0.0254 | 0.6238 |
| Selected round | 0 ± 0 | 7.3 ± 8.4 | — |

**Communication and compute overhead.**

- Clients send the same full-state payload as in FedAvg: 41,428 bytes per transfer, 414,280 bytes per round, and 8.29 MB over 20 rounds (analytical).
- DP adds **0 bytes**, because clipping is done on values the server already receives and the noise is added at the server.
- Wall-clock training time was 108–110 s per run, against 110–120 s for the paired FedAvg runs. Timing was not controlled, so no compute-overhead claim is made beyond "no material increase observed".

## 4. What happened

- **The round-0 model was selected in every seed.** No DP-trained round (rounds 1–20) reached a lower validation loss than the untrained initial model. The reported test results therefore describe the **untrained, data-independent initial networks** (each seed has a different random initialisation): all-`N` for seed 42, all-`V` for seed 123, and mostly `Q`/`e` for seed 2024. They are not the result of DP-trained learning. Day 11 FedAvg also scored round 0 but never selected it.
- **What the guarantee means here.** The ε = 7.98 guarantee applies to the whole training transcript, that is, all 20 broadcast models. The selected round-0 model itself depends on no client data.
- **Noise dominated the signal, as predicted before the runs.**
  - The noise L2 norm was 62.7–66.8 per round, against a clipped weighted client signal of 0.95–1.00, a noise-to-signal ratio of 63–69. The prediction made before running was z · max p_k · √d ≈ 65.
  - All 5 client updates were clipped in every round.
  - The per-coordinate noise standard deviation (0.62–0.65) is 4–11× the initial weight scale (0.06–0.15) in a single round.
  - Noise on the BatchNorm running variances (initially 1.0) drives some below 0. These are clamped to 0, so BatchNorm divides by √ε ≈ 0.003.
- **Effect on validation.** Validation loss rose from about 2.6 (round 0) to 1.3·10⁶–1·10⁸ after round 1. It stayed above 10⁶ for all 20 rounds, peaking near 3·10¹². Validation macro-F1 never exceeded 0.055 in any DP round, the level of a single-class predictor. The occasional validation accuracy of 0.694 is the all-`N` rate.

## 5. Interpretation

At the declared client-level guarantee (ε = 7.98, δ = 10⁻⁵, 5 clients, 20 rounds), DP-FedAvg **destroys utility**. No DP-trained global model improves on random initialisation, and the paired degradation relative to standard FedAvg is large in every seed (macro-F1 −38% to −90%; accuracy −0.05 to −0.62).

This is the expected consequence of client-level DP with very few clients. The noise needed to hide one client out of 5 scales with C · max p_k ≈ C / 4, while the useful signal is bounded by C. DP-FedAvg obtains usable utility only when the noise is averaged over many participants (in practice hundreds to thousands of clients per round).

It does **not** show that differential privacy is unusable for this task in general. Other privacy units (record-level DP-SGD inside each client), more clients, different ε, or DP-compatible architectures (for example GroupNorm instead of BatchNorm) were not tested.

No stronger guarantee than the one stated above is claimed.

## 6. Limitations

1. **One declared configuration.** There was no ε sweep and no clipping-norm study, so the utility-privacy curve is not mapped.
2. **The selection rule chose untrained models.** With the shared rule, the test metrics measure the initial networks, and DP-trained models were never test-evaluated. Their behaviour is documented on validation only.
3. **BatchNorm running statistics are privatised by noise.** This is required for the guarantee but especially destructive. Architectural alternatives were out of scope, because the model is frozen.
4. **Trust model.** Central DP with a trusted, honest server. There is no secure aggregation, homomorphic encryption, or protection of updates in transit.
5. **Accounting scope.** The accountant covers full participation only (q = 1). Public client sample counts are assumed. Floating-point and PRNG effects are not modelled.
6. **Inherited limitations.** All Day 10/11 limitations still apply: a simulation on public data, 3 seeds, a 7-record test set with 8 of 15 classes supported, and inferred subject-equivalence groups.

## 7. Artifacts

- **Per run** (`dp_fedavg_v1_natural_s{42,123,2024}`) under `results/privacy/{metrics,tables,figures}/federated/`:
  - privacy accounting
  - DP round diagnostics and client update norms
  - round history, test metrics, per-class metrics, confusion matrix, ROC metrics
  - partition, communication cost, run summary
- **Summary tables:** `results/privacy/tables/`
  - `dp_per_seed_privacy_utility.csv`
  - `dp_vs_fedavg_3seed_summary.csv`
  - `dp_vs_fedavg_paired*.csv`
  - `dp_vs_fedavg_prediction_counts.csv`
  - `dp_vs_fedavg_per_class.csv`
  - `dp_round_diagnostics_all_seeds.csv`
  - `robustness_*.csv`
- **Figures:** `results/privacy/figures/`
  - `dp_noise_vs_signal.png`
  - `dp_vs_fedavg_paired.png`
  - `robustness_convergence.png`
  - `robustness_test_metrics.png`
- **Not tracked (regenerable):** per-run checkpoints and `.npz` predictions (`.gitignore`).
- **Tests:** `tests/test_federated_privacy.py`
