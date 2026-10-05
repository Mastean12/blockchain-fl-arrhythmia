# Day 19 — Ablation of the proposed defense

- **Date:** 2026-10-05
- **Configuration:** [`configs/ablation/ablation_v1.json`](../configs/ablation/ablation_v1.json)
- **Code:**
  - component switches in `src/federated/proposed_defense.py`
  - `src/federated/ablation.py`: runner and report
- **Outputs:** `results/ablation/`
- **Tests:** `tests/test_ablation.py`

> This is a component ablation of the **frozen** Day 18 defense. The thresholds are unchanged (C = 2.603, τ_cos = 0.1814, τ_norm = 2.0, variance floor 10⁻⁵), and so are the seeds (42, 123, 2024), attacks, attacker draws, natural 5-client partition, budget (20 rounds, 1 local epoch), and round-selection rule. No retuning, no new attacks, no new defense, no DP, HE, or blockchain.

## 1. Variants

| # | Variant | Clipping | Cosine filter | Norm filter | Separate BatchNorm median + variance floor | Source |
|---|---|:-:|:-:|:-:|:-:|---|
| 1 | Plain FedAvg | – | – | – | – | Reused: Day 11 (clean), Day 17 (attacks) |
| 2 | Clipping only | ✓ | – | – | – | **New** |
| 3 | Clipping + cosine | ✓ | ✓ | – | – | **New** |
| 4 | Clipping + norm | ✓ | – | ✓ | – | **New** |
| 5 | Clipping + cosine + norm | ✓ | ✓ | ✓ | – | **New** |
| 6 | Full method | ✓ | ✓ | ✓ | ✓ | Reused: Day 18 |

- **Switches.** Each component has a boolean switch in the defense config. Absent switches mean the full method. A test checks that the default is identical to explicit all-on, and a re-run confirmed that the default code reproduces Day 18 runs **bit-identically** (rounds 0–3 of clean and scaled, seed 42).
- **BatchNorm when the robust component is off (variants 2–5).** Running statistics are the weighted mean over **accepted** clients, with the same weights as the parameters. This is the FedAvg treatment of buffers, with no median and no floor. A filtered client is therefore excluded from both parameters and statistics.
- **Disabled filters.** They are still computed and logged but never flag a client. Clipping applies to trainable parameters only, as in Day 18.
- **Runs.** There are 48 new runs (4 variants × 4 conditions × 3 seeds), each with one test evaluation. Variants 1 and 6 reuse 24 existing frozen runs with identical seeds, attacker draws, and code path.

## 2. Results (mean over 3 seeds; per-seed values in `results/ablation/tables/ablation_runs.csv`)

### Test macro-F1 (15 classes) and model breakdown

| Variant | Clean macro-F1 | Sign flip macro-F1 | Sign flip breakdown rounds* | Scaled ×10 macro-F1 | Scaled breakdown rounds* | Noise macro-F1 |
|---|---:|---:|---|---:|---|---:|
| 1. Plain FedAvg | 0.0851 | 0.0823 | **7 / 15 / 15** | 0.0384 | **10 / 17 / 10** | 0.0880 |
| 2. Clipping only | 0.0802 | 0.0832 | **7 / 15 / 15** | 0.0528 | **17 / 10 / 10** | 0.0856 |
| 3. + cosine filter | 0.0802 | **0.0854** | **0 / 0 / 0** | 0.0528 | **17 / 10 / 11** | 0.0855 |
| 4. + norm filter | 0.0802 | 0.0832 | **7 / 15 / 15** | **0.0854** | **0 / 0 / 0** | 0.0863 |
| 5. + cosine + norm | 0.0802 | **0.0854** | **0 / 0 / 0** | **0.0854** | **0 / 0 / 0** | 0.0863 |
| 6. Full (+ BatchNorm median/floor) | 0.0698 | 0.0688 | 0 / 0 / 0 | 0.0688 | 0 / 0 / 0 | 0.0750 |

\*Rounds with NaN validation loss out of 20, for seeds 42 / 123 / 2024. In every run they coincide exactly with the rounds whose global model has a negative BatchNorm running variance.

### Other metrics (mean over 3 seeds)

| Condition | Variant | Accuracy | Weighted-F1 | Macro recall | Macro AUROC | Detection rate | Honest false-positive rate | Final-round validation macro-F1 | Predicted classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 1 | 0.657 | 0.602 | 0.108 | 0.647 | — | — | 0.079 | 3.3 |
| Clean | 2–5 (identical†) | 0.635 | 0.587 | 0.103 | 0.656 | — | 0 | 0.080 | 3.3 |
| Clean | 6 | 0.625 | 0.569 | 0.095 | 0.654 | — | 0 | 0.073 | 3.3 |
| Sign flip | 1 | 0.680 | 0.597 | 0.104 | 0.586 | — | — | 0.012 (broken) | 2.3 |
| Sign flip | 2 / 4 (identical) | 0.687 | 0.605 | 0.105 | 0.596 | — / 0% | 0 | 0.012 (broken) | 2.3 |
| Sign flip | 3 / 5 (identical) | 0.673 | 0.597 | 0.105 | 0.627 | **100%** | 0 | 0.085 | 2.3 |
| Sign flip | 6 | 0.673 | 0.557 | 0.082 | 0.623 | **100%** | 0 | 0.081 | 2.0 |
| Scaled ×10 | 1 | 0.463 | 0.350 | 0.067 | 0.583 | — | — | 0.022 (broken) | 1.0 |
| Scaled ×10 | 2 | 0.655 | 0.519 | 0.067 | 0.599 | — | 0 | 0.040 (broken) | **1.0** |
| Scaled ×10 | 3 | 0.655 | 0.519 | 0.067 | 0.599 | 1.7% | 0 | 0.026 (broken) | **1.0** |
| Scaled ×10 | 4 / 5 (identical) | 0.673 | 0.597 | 0.105 | 0.627 | **100%** | 0 | 0.085 | 2.3 |
| Scaled ×10 | 6 | 0.673 | 0.557 | 0.082 | 0.623 | **100%** | 0 | 0.081 | 2.0 |
| Noise | 1 | 0.691 | 0.608 | 0.107 | 0.634 | — | — | 0.085 | 2.3 |
| Noise | 2 | 0.701 | 0.614 | 0.109 | 0.643 | — | 0 | 0.085 | 2.3 |
| Noise | 3 | 0.700 | 0.614 | 0.110 | 0.641 | 1.7% | 0 | 0.085 | 2.3 |
| Noise | 4 / 5 (identical) | 0.698 | 0.607 | 0.105 | 0.633 | 6.7% | 0 | 0.086 | 2.3 |
| Noise | 6 | 0.686 | 0.577 | 0.093 | 0.628 | 11.7% | 0 | 0.082 | 1.7 |

†No client is ever flagged in clean runs, so variants 2–5 are bit-identical there and differ from plain FedAvg only by clipping. Whenever the attacker is excluded in every round, variants 3 and 5 under sign flip and variants 4 and 5 under scaling give **identical** models: the same 4 honest clients with clipping and FedAvg-style BatchNorm. Detection under variant 2 is "—" because no filter is active.

**Computational overhead.** Server-side defense time is 3.2–3.9 ms per round for every defended variant (all statistics are computed even when a component is switched off), against 0 for plain FedAvg. No variant adds communication or client computation.

The figure is `results/ablation/figures/ablation_overview.png`.

## 3. Attribution

| Question | Responsible component | Evidence |
|---|---|---|
| **What prevents the sign-flip failure?** | **The cosine filter** | Breakdown persists with clipping only (7/15/15 NaN rounds, the same as plain FedAvg) and with clipping + norm (norm filter detection 0%; a flipped update has the honest magnitude). It disappears whenever the cosine filter is on (variants 3, 5, 6: 100% detection, 0 NaN rounds). |
| **What prevents the ×10 scaling failure?** | **The norm filter** | Clipping alone does **not** prevent it (10–17 NaN rounds, single-class models). The cosine filter does not either (detection 1.7%; a scaled update points in the honest direction). It disappears whenever the norm filter is on (variants 4, 5, 6: 100% detection, 0 NaN rounds). |
| Why does clipping alone fail against scaling? | The failure channel is BatchNorm running statistics, which clipping does not touch | Clipping bounds the attacker's *parameter* update, yet the model still breaks: the attacker's 10× boosted running statistics still enter the weighted average and drive variances negative. Only *excluding* the attacker, which removes its statistics too, stops the breakdown. |
| **What causes the clean-performance degradation?** | **Mainly the separate BatchNorm median component; clipping contributes a little** | Clean macro-F1: plain 0.0851 → clipping 0.0802 (−0.005; −0.0005/−0.0097/−0.0044 per seed) → full 0.0698 (a further −0.010; seed 123: 0.0889 → 0.0591). The same pattern holds under every attack once the attacker is filtered: variant 5 gives 0.0854 against 0.0688 for variant 6. |
| **What provides BatchNorm stability?** | **Attacker filtering (cosine + norm), not the BatchNorm median or floor** | Variant 5, with no median and no floor, had 0 negative-variance and 0 NaN rounds under every attack. The full method never applied its variance floor (0 entries floored in any run). Its median aggregation added no stability here and cost utility. Whether the BatchNorm median alone, without filters, could prevent breakdown was not tested; it is not one of the six requested variants. |

## 4. Interpretation

1. **Each filter covers exactly one attack type.** The cosine filter handles direction attacks (sign flip) and the norm filter handles magnitude attacks (scaling). Both are needed to cover both Day 17 failure modes. Neither prevents the other's failure.
2. **Clipping alone is not a defense here.** It changes little against either attack, because the damage travels through unclipped BatchNorm statistics. Its main measurable effect is a small clean-utility cost.
3. **The separate BatchNorm component of the frozen Day 18 method is not supported by this ablation.** In these 3 attack settings it was redundant, since filtering already removed the attacker's statistics and the floor was never triggered, and it carried most of the clean-utility cost. Variant 5 (clipping + cosine + norm with FedAvg-style statistics over accepted clients) matched or exceeded the full method on macro-F1 in every condition.
   - This is observed on 3 seeds and is driven largely by seed 123. It is not a significance claim.
   - The BatchNorm component might still matter for attacks that evade both filters, such as an adaptive attacker manipulating only running statistics. That was not tested.
4. **Implication.** A revised method (v2) should drop the BatchNorm median, or keep only the floor as a cheap safeguard, and fix the self-inclusive median reference (Day 18). It must be declared as a new version and validated on fresh evaluation rather than retuned on these results. Day 18's frozen v1 is unchanged.
5. **Norm-matched noise remains largely undetected** (1.7–11.7%) and harmless in all variants, consistent with Days 17–18.

## 5. Limitations

1. 3 seeds and one attacker per run. Several effects (notably the BatchNorm-median clean cost) rest mostly on one seed (123). No significance testing.
2. The 6 requested variants do not isolate the BatchNorm median *without* filters, or the floor without the median.
3. Thresholds were frozen from one calibration seed. Interactions might differ under other thresholds, but no retuning was done, by design.
4. Selected-round test metrics can mask breakdown (Day 17). Breakdown is therefore also reported through NaN rounds and final-round validation macro-F1.
5. Overhead timings come from one machine without load control.
6. Inherited limitations: a simulation on public data, a weak and collapsed baseline, a 7-record test set, and non-adaptive attacks.

## 6. Integrity

- With default switches, the code reproduces Day 18 bit-identically.
- Clean FedAvg (no defense section) still reproduces the official Day 10 model bit-identically.
- All 1,224 pre-existing files under `results/`, `data/splits/`, `configs/`, and `tests/` are byte-identical after Day 19, and the processed data is unchanged.
- **Not tracked (regenerable):** per-run checkpoints, `.npz` predictions, and trajectories.

## 7. Artifacts

- **Tables:** `results/ablation/tables/`
  - `ablation_runs.csv` (72 rows: 6 variants × 4 conditions × 3 seeds, including the reused runs)
  - `ablation_summary.csv` (mean, SD, minimum, and maximum per variant and condition)
- **Per new run** (`ablation_{clip_only,clip_cos,clip_norm,clip_cos_norm}_{condition}_s{seed}`) under `results/ablation/{metrics,tables,figures}/federated/`:
  - defense client and round diagnostics
  - round history, test metrics, confusion matrix
- **Figure:** `results/ablation/figures/ablation_overview.png`
