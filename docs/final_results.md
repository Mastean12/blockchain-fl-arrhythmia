# Final results (research freeze, Day 20)

- **Study:** A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection
- **Freeze date:** 2026-10-05
- **Machine-readable versions:**
  - `results/final/final_summary.json`
  - per-dimension CSVs in `results/final/`
  - `results/final/experiment_manifest.json`
- **Related documents:**
  - methodology: [`final_methodology.md`](final_methodology.md)
  - limitations: [`final_limitations.md`](final_limitations.md) (they apply to every statement below)
  - reproducibility: [`reproducibility.md`](reproducibility.md)

> **Scope of every result:** a simulated 5-client federation on public MIT-BIH data, run on one CPU machine. Seed counts are small (3 per condition, plus 3 fresh seeds), so every comparison is **descriptive**. No significance testing was performed and no result is called statistically significant. Nothing here demonstrates clinical usefulness or diagnostic validity.

## 1. Day 20 independent confirmation (fresh seeds 101, 202, 303)

**Design, declared before running** (`configs/final/final_validation_v1.json`):

- **Defense:** Variant 5 (L2 clipping + cosine filter + norm filter, FedAvg-style BatchNorm statistics over accepted clients, no BatchNorm median or floor) with the frozen Day 18 thresholds (C = 2.603, τ_cos = 0.1814, τ_norm = 2.0).
- **Setting:** the same preprocessing, split, natural 5-client partition, 20 rounds, 1 local epoch, Day 17 attacks and attacker-selection rule.
- **Seeds:** 101, 202 and 303 have never been used for evaluation or calibration.
- **Controls:** matching FedAvg runs, i.e. the defense in observe mode, which aggregates with plain FedAvg. It reproduces the official Day 10 model bit-identically.
- **Runs:** 24 in total. Nothing was tuned from these results.

### Test results (mean ± sample SD [min, max] over 3 fresh seeds)

| Condition | Arm | Accuracy | Macro-F1 | Weighted-F1 | Macro recall | Macro AUROC |
|---|---|---|---|---|---|---|
| Clean | FedAvg control | 0.706 ± 0.003 | 0.0946 ± 0.0086 [0.089, 0.105] | 0.607 ± 0.014 | 0.112 | 0.643 |
| Clean | Variant 5 | 0.709 ± 0.002 | 0.0905 ± 0.0031 [0.087, 0.093] | 0.613 ± 0.016 | 0.115 | 0.638 |
| Sign flip | FedAvg control | 0.650 ± 0.074 | 0.0854 ± 0.0206 [0.062, 0.100] | 0.560 ± 0.038 | 0.088 | 0.588 |
| Sign flip | Variant 5 | **0.704 ± 0.011** | 0.0854 ± 0.0017 [0.084, 0.087] | **0.622 ± 0.012** | **0.113** | 0.626 |
| Scaled ×10 | FedAvg control | **0.437 ± 0.378** [0.000, 0.655] | **0.0352 ± 0.0305** [0.000, 0.053] | 0.346 ± 0.300 | 0.044 | 0.608 |
| Scaled ×10 | Variant 5 | **0.704 ± 0.011** | **0.0854 ± 0.0017** | **0.622 ± 0.012** | **0.113** | 0.626 |
| Noise | FedAvg control | 0.707 ± 0.001 | 0.0891 ± 0.0027 | 0.607 ± 0.008 | 0.112 | 0.638 |
| Noise | Variant 5 | 0.702 ± 0.004 | 0.0982 ± 0.0118 [0.085, 0.105] | 0.601 ± 0.010 | 0.109 | 0.632 |

### Robustness, selection, deviation and overhead

| Condition | Arm | Selected rounds (101 / 202 / 303) | Breakdown (NaN-validation) rounds | Final-round validation macro-F1 | Attack detection (applied) | Honest false-positive rate | Relative deviation from clean FedAvg, round 20 |
|---|---|---|---|---|---|---|---|
| Clean | Control | 3 / 18 / 2 | 0 / 0 / 0 | 0.085 | — | — | 0 |
| Clean | Variant 5 | 3 / 18 / 3 | 0 / 0 / 0 | 0.087 | — | **0%** | 0.04–0.07 |
| Sign flip | Control | 3 / 4 / 4 | **16 / 16 / 16** | **0.011 (broken)** | — | — | 0.37–0.57 |
| Sign flip | Variant 5 | 3 / 14 / 3 | **0 / 0 / 0** | 0.084 | **100%** | **0%** | 0.16–0.28 |
| Scaled ×10 | Control | 4 / **0** / 2 | **10 / 20 / 10** | **0.022 (broken)** | — | — | 164 to 7.6·10⁵ |
| Scaled ×10 | Variant 5 | 3 / 14 / 3 | **0 / 0 / 0** | 0.084 | **100%** | **0%** | 0.16–0.28 |
| Noise | Control | 4 / 5 / 3 | 0 / 0 / 0 | 0.077 | — | — | 0.21–0.36 |
| Noise | Variant 5 | 3 / 19 / 3 | 0 / 0 / 0 | 0.081 | 5–10% | 0% | 0.19–0.31 |

- **Selected round 0 in the scaled control (seed 202).** No attacked round beat the untrained model, so the untrained model was selected. Its test accuracy is 0.000 because it predicts only classes absent from the test set.
- **Identical defended runs.** Under Variant 5, sign flip and scaled give **identical** results per seed: the attacker is excluded in every round, as on Days 18 and 19.
- **Overhead.** Variant 5 takes 3.4–3.7 ms per round on the server, about 0.07 s per run against about 110 s of training (about 0.06%). It adds 0 bytes of communication and no client computation.
- **What the control's detection columns mean.** The observe-mode control also logs *would-be* flags, which never affect aggregation: sign flip and scaling 100%, noise 10–55%, honest 0%. These are reported in `final_confirmation_runs.csv` and not counted as detection.

### Confirmation verdict

The Day 19 attribution held on fresh seeds:
- **Attacks neutralised:** Variant 5 neutralised the sign-flip and ×10 attacks (100% detection, 0 false positives, 0 breakdown rounds, final models valid). Under scaling, accuracy went from 0.437 to 0.704 and macro-F1 from 0.035 to 0.085.
- **Clean cost:** −0.004 macro-F1, within seed variability. That is smaller than the −0.015 of the Day 18 v1 method on the original seeds, consistent with Day 19 attributing most of v1's clean cost to the BatchNorm median.
- **Noise:** largely undetected (5–10%) and harmless, as before.
- **Sign-flip test numbers:** they were equal in the two arms (0.0854 macro-F1 each), because selection picked pre-breakdown rounds for the control. The difference shows in model validity and final-round behaviour.

## 2. Consolidated findings by dimension

### Utility (`results/final/final_utility.csv`)

| Setting | Test accuracy | Test macro-F1 | Source |
|---|---|---|---|
| Centralized CNN (1 seed) | 0.7688 | 0.1399 | Day 6 |
| FedAvg (3 seeds) | 0.657 ± 0.076 | 0.085 ± 0.013 | Day 11 |
| FedAvg, fresh seeds (3) | 0.706 ± 0.003 | 0.095 ± 0.009 | Day 20 |
| FedBN-style | 0.651 | 0.084 | Day 12 |
| DP-FedAvg (ε = 7.98) | 0.247 ± 0.356 | 0.024 ± 0.025 | Day 13 |
| HE-FedAvg (CKKS) | 0.665 ± 0.062 | 0.086 ± 0.012 | Day 14 |
| FedAvg + blockchain recording | bit-identical to FedAvg | bit-identical | Day 16 |

- **FedAvg against centralized.** All **37** clean federated runs across Days 10–20 scored below the single centralized baseline run on accuracy, macro-F1 and weighted-F1. The best federated run reached 0.712 / 0.105 / 0.658. This count excludes DP and attacked runs. Federated predictions collapse to mostly `N`/`V` (Days 10–11).
- **Classes barely detected.** Across all **112** federated runs, at most 1 of 2,078 paced (`/`) beats and 4 of 2,001 `L` beats were classified correctly.
- **Weak baseline.** The centralized baseline itself is weak in macro terms (macro-F1 0.14, 8 of 15 classes with test support).

### Privacy and confidentiality (`final_privacy_confidentiality.csv`)

- **FedAvg:** raw ECG stays on the simulated clients by design. There is no formal guarantee.
- **DP-FedAvg:**
  - **Guarantee:** client-level (ε = 7.98, δ = 10⁻⁵)-DP over all 20 released models, with a trusted server and q = 1. Exact μ-GDP accounting gives μ = 1.6625; the RDP cross-check gives ε ≤ 9.36.
  - **Utility:** it **destroyed utility** at this unit with 5 clients. The noise-to-signal ratio was about 65, and round 0 was selected in every seed.
  - **Not tested:** other ε values and other privacy units.
- **HE-FedAvg:**
  - **What it protects:** an honest-but-curious server never sees individual updates (CKKS at 128-bit security).
  - **What it does not:** it gives **no** guarantee about what the aggregate or the global model reveals.
  - **Key model:** one shared secret key.
- **Blockchain:** provides no confidentiality. It stores only hashes, IDs and counts.

### Integrity and auditability (`final_integrity_auditability.csv`)

- **Synthetic ledger (Day 15):**
  - **Without an anchor:** inconsistent edits are detected. Consistent rewrites and truncations go undetected (0%).
  - **With an external anchor:** all 13 tamper types are detected (100%).
- **Real FedAvg runs (Day 16):**
  - **Verification:** 363 of 363 recorded client and global model hashes verify against the off-chain artifacts.
  - **Tampering:** detected with the anchor in 100% of 8,400 trials, including a 10⁻⁶ change to one archived weight.
  - **No effect on training:** recording is bit-identical to FedAvg.
- **Not demonstrated:** a distributed or permissioned blockchain, consensus, signatures, or client authentication.

### Robustness (`final_robustness.csv`)

- **FedAvg without a defense:**
  - **Scaled ×10:** destroyed by one malicious client in every seed.
  - **Sign flip:** the model breaks within 6–12 rounds through negative BatchNorm running variances.
  - **Norm-matched noise:** absorbed.
- **Day 18 defense v1 and Days 19–20 Variant 5:**
  - **Results:** 100% detection of sign flip and scaling, 0 honest false positives across all runs, and no breakdown.
  - **Which filter does the work:** Day 19 found that the cosine filter stops sign flip and the norm filter stops scaling. Clipping alone stops neither, and the BatchNorm median was unnecessary.
- **Not demonstrated:** robustness to adaptive, colluding, multiple, or data-poisoning attackers.

### Communication (`final_communication.csv`)

| Setting | Bytes per 20-round run (5 clients) | Ratio to FedAvg |
|---|---:|---:|
| FedAvg (plaintext) | 8.29 MB (41,428 B per model transfer) | 1.0 |
| DP-FedAvg | 8.29 MB (0 extra bytes) | 1.0 |
| HE-FedAvg (all ciphertext traffic) | 169.9 MB | **20.5** |
| Blockchain ledger | 22.3 KB per run | 0.0027 |
| Proposed defense (v1 / Variant 5) | 0 extra bytes (server-side) | 1.0 |

The FedAvg and DP figures are analytical; the HE and ledger figures are serialised sizes. None are network measurements.

### Computation (`final_computation.csv`; one CPU machine, uncontrolled load)

| Component | Time |
|---|---|
| FedAvg training run (20 rounds) | about 110–120 s |
| HE: encryption per client per round | 0.017 s |
| HE: server aggregation per round | 0.020 s |
| HE: total per run | 2.2 s (about 2% of training) |
| Blockchain: hashing + blocks per run | 0.029 s (about 0.025%) |
| Defense (v1 or Variant 5): server time per round | 3.4–3.8 ms |
| DP: clipping and noise | no material increase observed |

## 3. Research questions

The research questions are those in AGENTS.md §4. Status values are defined in `final_summary.json`.

| RQ | Question (abridged) | Status | Evidence | What prevents a conclusive answer |
|---|---|---|---|---|
| **RQ1** | Accuracy of ECG deep learning centrally and federated | **Partially answered** | Centralized test accuracy 0.769 and macro-F1 0.140 (Day 6); FedAvg macro-F1 0.085–0.095 (Days 11, 20); per-class analysis (Days 6, 8, 10, 11) | One dataset; provisional 15-symbol taxonomy; only 8 of 15 classes have test support; weak minority-class detection |
| **RQ2** | FL against centralized and local training under non-IID data | **Partially answered** | FedAvg below centralized in every run, with N/V collapse (Days 10–11). FedBN does not explain the gap (Day 12). `/` and `L` are confined to 2 records each, so no whole-group partition can spread them (Day 11). | No local-only baseline; collapse mechanism unresolved; budgets not matched |
| **RQ3** | DP and HE effects on utility, privacy, cost and communication | **Partially answered** | DP at ε = 7.98, client-level: utility destroyed, 0 extra bytes (Day 13). HE (CKKS): utility preserved within about 10⁻⁷, about 20.5× communication, about 2% compute (Day 14). | One ε; client-level unit only; no record-level DP or ε sweep; DP and HE never combined |
| **RQ4** | Blockchain for integrity, traceability, verification and auditability | **Partially answered** | Tamper-evident ledger, 100% detection with an anchor (Days 15–16); 363/363 hashes verified; behaviour-neutral; about 0.025% overhead | Single-node prototype; no permissioned network, consensus, signatures or authentication; server-computed hashes |
| **RQ5** | A proposed mechanism that improves an identified limitation at acceptable cost | **Partially answered** | Identified limitation: FedAvg's lack of protection against magnitude and direction attacks, including BatchNorm breakdown (Day 17). Defense v1 (Day 18) and ablation (Day 19) led to Variant 5, confirmed on fresh seeds (Day 20): attacks neutralised, clean cost −0.004 macro-F1, about 3.5 ms per round, 0 bytes. | One non-adaptive attacker and 3 attack settings; Variant 5 chosen after Day 19 (confirmed on fresh seeds but on the same dataset and split); not combined with DP, HE or blockchain; no novelty established through a literature review |

**No research question is fully answered.** Each has supporting evidence within the simulated scope, and the gaps are listed above.

## 4. Overclaiming review (applied to all Day 20 text)

- The words "significant", "novel", "state-of-the-art", "superior" and "clinically useful" are not used as claims.
- The defense is called "proposed" or "candidate". Its components (clipping, similarity and norm screening against a robust reference) are established ideas.
- Every privacy statement names its unit, trust model and parameters. HE is not described as protecting the model or the aggregate.
- Blockchain is described as a single-node, tamper-evident ledger, not a distributed blockchain.
- Robustness is stated only for the three tested attacks with one non-adaptive attacker.
- Selection masking is stated explicitly wherever selected-round test metrics look benign (sign flip).

## 5. Final artifacts

| Path | Content |
|---|---|
| `results/final/final_summary.json` | Machine-readable summary: all dimensions, the Day 20 confirmation, research-question status, and a list of what was not demonstrated |
| `results/final/final_utility.csv` | Utility for every study arm (mean, SD, minimum, maximum) |
| `final_privacy_confidentiality.csv`, `final_integrity_auditability.csv`, `final_robustness.csv`, `final_communication.csv`, `final_computation.csv` | One table per dimension |
| `final_confirmation_runs.csv`, `final_confirmation_summary.csv` | The 24 Day 20 runs and their summary |
| `experiment_manifest.json` | Every experiment (day, configuration, seeds, outputs, document), SHA-256 of key frozen artifacts, git history at freeze, and the list of regenerable untracked artifacts |
| `figures/final_utility_overview.png`, `figures/final_confirmation.png`, `figures/final_costs.png` | Final figures |
| `results/final/runs/` | Per-run Day 20 outputs: CSV/JSON tracked; checkpoints, `.npz` files and per-run figures ignored |

## 6. Unresolved issues that prevent calling the project research-complete

1. **The literature review is not done.** `docs/literature_matrix.md` is an empty template. AGENTS.md requires design choices and any contribution claim to be justified by a documented literature review. That justification is missing, so positioning the proposed defense against prior work is impossible.
2. **The core task is weak.** Centralized macro-F1 is 0.14, only 8 of 15 classes have test support, the taxonomy is provisional, and the federated N/V collapse is unexplained. The task definition (taxonomy, and possibly dataset or split) would need to be settled before claims about arrhythmia detection quality.
3. **A local-only baseline is missing** (RQ2).
4. **Privacy evaluation is single-point.** There is no ε sweep, no record-level DP, and no combined DP+HE.
5. **The blockchain is a single-node prototype.** There is no permissioned network, signatures or consensus.
6. **The system is not integrated.** DP, HE, the blockchain and the defense were each evaluated in isolation. The full framework named in the study title was never run end to end.
7. **The statistics are small.** There are 3 + 3 seeds, one centralized seed, and no significance testing.

These are documented, not hidden. The current freeze is an **experimentally complete prototype evaluation within a simulated, single-dataset scope**. It is not a research-complete answer to RQ1–RQ5.
