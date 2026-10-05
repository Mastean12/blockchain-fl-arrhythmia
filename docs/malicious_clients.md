# Day 17 — Malicious-client attacks against plain FedAvg (threat model and vulnerability)

- **Date:** 2026-10-05
- **Configuration:** [`configs/security/attacks_v1.json`](../configs/security/attacks_v1.json), which uses the unchanged base [`fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Code:**
  - `src/federated/attacks.py`: attacks and threat model
  - the `attack` branch in `src/federated/run_fedavg.py`
  - `src/federated/security.py`: study runner and report
- **Outputs:** `results/security/`
- **Tests:** `tests/test_attacks.py`

> This study establishes the threat model and quantifies the vulnerability of **standard FedAvg without any defense**. No defense, robust aggregation, DP, HE, or blockchain is used. Its purpose is to inform the Day 18 defense design.

## 1. Threat model

| Aspect | Assumption |
|---|---|
| Adversary | **One** compromised client out of 5, the same client in every round. With full participation, that means exactly one malicious update per round. |
| Selection | Seeded draw, `default_rng([seed, 1717]).integers(5)`, recorded per run: seed 42 → `client_04` (15.4% of training samples), seed 123 → `client_03` (16.4%), seed 2024 → `client_01` (21.9%). |
| Knowledge | The broadcast global model and its own honest local update. No collusion and no access to other clients' updates. |
| Capability | It trains honestly, then transmits w_t + Δ′ instead of w_t + Δ, over **all floating-point state entries** (weights, biases, BatchNorm affine parameters and running statistics). Integer counters are unchanged. It reports its **true** sample count, so sample-count inflation is out of scope. |
| Server | Unmodified sample-weighted FedAvg. No update validation, norm checks, clipping, or robust aggregation. Round selection still uses the server's **clean validation split** (the Day 10 protocol). |
| Out of scope | Data and label poisoning, backdoors, sample-count lying, multiple or colluding attackers, adaptive attacks, attacks on DP, HE, or blockchain runs. |

**Attacks** (declared before any run; not tuned):

| Name | Transmitted update | Intent |
|---|---|---|
| `sign_flip` | Δ′ = −Δ | Reverse the honest learning direction at the same magnitude |
| `scaled` | Δ′ = 10·Δ | Boost the attacker's influence (a model-replacement-style magnitude attack) |
| `random_noise` | Δ′ = ‖Δ‖ · z / ‖z‖, z ~ N(0, I), seeded per seed and round | Random direction with the honest magnitude: a stealthy, norm-matched corruption |

**Fixed budget:** identical to Day 11 (natural 5-client partition, 20 rounds, 1 local epoch, Adam lr 0.001, batch size 256, full participation, round selection by minimum validation loss, one test evaluation per run).

**Clean comparator:** the Day 11 natural FedAvg run with the same seed. **Global-model deviation** is ‖w_attacked − w_clean‖ / ‖w_clean‖ per round, against the clean per-round global states archived by the Day 16 recorded runs, which are bit-identical to clean FedAvg.

## 2. Results

### Test metrics (selected round, evaluated once per run)

| Attack | Seed | Selected round (clean) | Accuracy (clean) | Macro-F1 (clean) | Weighted-F1 (clean) | Macro recall | Macro AUROC (clean) | Predicted classes |
|---|---|---|---|---|---|---:|---|---|
| sign flip | 42 | 3 (3) | 0.7020 (0.7069) | 0.0821 (0.0848) | 0.6202 (0.6574) | 0.1069 | 0.6202 (0.6733) | `N` 12,254, `V` 3,305 |
| sign flip | 123 | 3 (2) | 0.6551 (0.6943) | 0.0823 (0.0986) | 0.5805 (0.5841) | 0.1013 | 0.6278 (0.6463) | `N` 12,405, `V` 2,389, `R` 765 |
| sign flip | 2024 | 4 (17) | 0.6836 (0.5690) | 0.0823 (0.0717) | 0.5911 (0.5648) | 0.1047 | 0.5092 (0.6226) | `N` 12,931, `V` 2,628 |
| **scaled ×10** | 42 | **0** (3) | 0.6554 (0.7069) | 0.0528 (0.0848) | 0.5189 (0.6574) | 0.0667 | 0.4244 (0.6733) | **all `N`** |
| **scaled ×10** | 123 | 4 (2) | **0.0778** (0.6943) | **0.0096** (0.0986) | **0.0112** (0.5841) | 0.0667 | 0.6690 (0.6463) | **all `V`** |
| **scaled ×10** | 2024 | 2 (17) | 0.6554 (0.5690) | 0.0528 (0.0717) | 0.5189 (0.5648) | 0.0667 | 0.6558 (0.6226) | **all `N`** |
| random noise | 42 | 3 (3) | 0.7060 (0.7069) | 0.0837 (0.0848) | 0.6489 (0.6574) | 0.1140 | 0.6542 (0.6733) | `N` 11,002, `V` 4,557 |
| random noise | 123 | 3 (2) | 0.6617 (0.6943) | 0.0940 (0.0986) | 0.5718 (0.5841) | 0.0988 | 0.6514 (0.6463) | `N` 13,916, `V` 917, `R` 726 |
| random noise | 2024 | 3 (17) | 0.7038 (0.5690) | 0.0864 (0.0717) | 0.6043 (0.5648) | 0.1085 | 0.5954 (0.6226) | `N` 13,176, `V` 2,383 |

**3-seed summary (mean ± sample SD; descriptive, no significance test).** Clean FedAvg: accuracy 0.657 ± 0.076, macro-F1 0.085 ± 0.013, weighted-F1 0.602 ± 0.049.

| Attack | Accuracy | Macro-F1 | Weighted-F1 | Macro recall | Δ macro-F1 vs clean (range) |
|---|---|---|---|---|---|
| sign flip | 0.680 ± 0.024 | 0.0823 ± 0.0001 | 0.597 ± 0.021 | 0.104 ± 0.003 | −0.016 to +0.011 |
| **scaled ×10** | **0.463 ± 0.334** | **0.038 ± 0.025** | **0.350 ± 0.293** | **0.0667 ± 0** | **−0.089 to −0.019** |
| random noise | 0.691 ± 0.025 | 0.088 ± 0.005 | 0.608 ± 0.039 | 0.107 ± 0.008 | −0.005 to +0.015 |

### Prediction collapse

- **Scaled ×10:** the selected model is a **single-class predictor** in every seed (all `N` twice, all `V` once), giving macro recall 1/15 = 0.0667. The seed-42 selection is round 0, the untrained initial model, because no attacked round beat it on validation loss.
- **Sign flip and random noise:** 2–3 predicted classes, close to clean, with `N` taking 71–89% of predictions.
- **Clean FedAvg:** already collapsed to mostly `N`/`V` (Days 10–11), so it has little room to show further collapse except to a single class.

### Model breakdown and global-model deviation

| Attack | Seed | Rounds with negative BatchNorm running variance = rounds with NaN validation loss | First such round | Relative weight deviation from clean at round 20 |
|---|---|---|---|---|
| sign flip | 42 | 7 / 20 | 12 | 0.34 |
| sign flip | 123 | 15 / 20 | 6 | 0.35 |
| sign flip | 2024 | 15 / 20 | 6 | 0.51 |
| scaled ×10 | 42 | 10 / 20 | 1 | 2.5·10² |
| scaled ×10 | 123 | 17 / 20 | 1 | 9.0·10² |
| scaled ×10 | 2024 | 10 / 20 | 1 | 4.0·10⁵ |
| random noise | 42 / 123 / 2024 | 0 / 20 | — | 0.24 / 0.26 / 0.31 |

- **What breaks the model.** No global model ever had non-finite **weights**. The sign-flip and scaled attacks drive aggregated BatchNorm **running variances negative**, so BatchNorm's square root returns NaN outputs. The rounds with negative variance match the rounds with NaN validation loss exactly, in count and in first occurrence, for every run.
- **Scaled ×10 grows without bound.** The deviation grows exponentially, by about ×1.3–×1.9 per round between rounds 10 and 20: the attacker boosts an update that was itself computed from an already corrupted global model, so the honest update norm grows too (mean 747 to 435,025 against 2.8–4.7 for the other attacks).
- **Sign flip stays small in distance but is still destructive.** Its weight deviation stays modest (0.3–0.5), but it still breaks BatchNorm in the later rounds.
- **Random noise is absorbed.** Its deviation stays near the noise-injection level, about 0.15–0.31.

The figures are `results/security/figures/attack_test_metrics.png` and `attack_global_deviation.png`.

## 3. Interpretation

1. **Plain FedAvg is vulnerable to a single magnitude-boosting client.** One client holding 15–22% of the data and scaling its update by 10 destroys the model in every seed. Training breaks from round 1, the selected model degenerates to one class, and macro-F1 falls by 0.019–0.089.
2. **Sign flipping breaks the model later, and validation-based selection hides it.** The global model becomes invalid, with NaN outputs, from round 6 or 12 onward. The reported test metrics stay near clean only because the server selects an early round (3–4) from before the breakdown, using its trusted clean validation split.
   - Without that validation data, or if the final model were deployed, the sign-flip result would be a broken model.
   - The apparent accuracy *gain* for seed 2024 (0.684 against 0.569) is a selection-timing artefact: the clean run selected round 17.
   - Server-side validation is therefore acting as an implicit, partial defense, and it should not be counted as robustness.
3. **Norm-matched random noise from one of 5 clients is absorbed by averaging.** Test metrics stay within clean seed variability, and the deviation stays bounded at about 0.24–0.31.
4. **Shared root cause.** Every damaging case relies on **unchecked update magnitude and direction**, including BatchNorm statistics. FedAvg has no norm bound, no outlier rejection, and no sanity checks such as non-negative variances.

**Implications for the Day 18 defense.** A defense should at least bound or normalise update norms, reject or down-weight updates whose direction or magnitude is anomalous relative to the other clients, and treat BatchNorm running statistics separately, or validate them. It should be evaluated both on the selected round and on the final round, so that validation selection cannot mask an attack.

## 4. Limitations

1. One attacker, three fixed attack settings (scale 10, flip factor 1, noise multiplier 1), and three seeds. Severity at other magnitudes and attacker counts is not mapped.
2. The attacker's data share differs by seed (15–22%) and is confounded with seed.
3. Attacks are non-adaptive. A stealthy attacker aware of a defense would behave differently.
4. Test metrics come from validation-selected rounds, which partly mask the attacks. Final-round degradation is documented on validation only, to keep test use at one evaluation per run.
5. Data and label poisoning and backdoors are not evaluated.
6. Inherited limitations: a simulation on public data, an already weak and collapsed clean baseline (macro-F1 about 0.085), and a 7-record test set.

## 5. Integrity

- With no `attack` section, the code still reproduces the official Day 10 model bit-identically.
- All 688 pre-existing files under `results/`, `data/splits/`, `configs/`, and `tests/` are byte-identical after Day 17, including the official Day 10 model and predictions and every Day 6–16 results tree. The processed data is unchanged.
- **Not tracked (regenerable):** per-run checkpoints, `.npz` predictions, and global trajectories.

## 6. Artifacts

- **Tables:** `results/security/tables/`
  - `attack_runs.csv` (all per-run metrics, collapse, breakdown, and deviation)
  - `attack_summary.csv`
  - `attack_global_deviation_by_round.csv`
  - `attack_per_class.csv` (including clean)
- **Per run** (`attack_{sign_flip,scaled,random_noise}_s{42,123,2024}`) under `results/security/{metrics,tables,figures}/federated/`:
  - `*_attack_round_diagnostics.csv` (honest and malicious update norms, cosine, attacker weight)
  - round history, test metrics, per-class metrics, confusion matrix, run summary
- **Figures:** `results/security/figures/attack_test_metrics.png` and `attack_global_deviation.png`
