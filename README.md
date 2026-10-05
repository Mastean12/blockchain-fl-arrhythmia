# A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection

An academic research prototype that studies how federated learning (FL), differential privacy (DP), homomorphic encryption (HE), a blockchain-style audit ledger, and a robust aggregation defense affect ECG beat classification on the MIT-BIH Arrhythmia Database.

> **Status: research freeze (2026-10-05).** All planned experiments are complete and frozen for a **simulated, single-dataset** setting: 5 simulated clients, one public dataset, one CPU machine. The research questions are **partially answered**. The open issues are listed under [Status and open issues](#status-and-open-issues). This repository does not demonstrate clinical usefulness or diagnostic validity.

## Research aim

To develop and evaluate a blockchain-enabled federated learning framework for privacy-preserving ECG-based arrhythmia detection. The research investigates collaborative model training across institutions without exchanging raw ECG records, and evaluates the benefits and costs of federated learning, privacy mechanisms, and blockchain coordination.

## Objectives

1. Develop an ECG-based deep learning model for arrhythmia detection using publicly available ECG datasets.
2. Design a federated learning architecture that supports collaborative training without exchanging raw ECG data.
3. Investigate differential privacy and homomorphic encryption as privacy-preserving mechanisms.
4. Design a permissioned blockchain coordination and verification mechanism for federated model updates.
5. Develop and evaluate a proposed aggregation and/or coordination method addressing an identified limitation.
6. Compare the framework with appropriate centralized, federated, privacy-preserving, and blockchain-enabled baselines.
7. Evaluate predictive performance, privacy, computational cost, communication overhead, blockchain overhead, convergence, and robustness to relevant attacks.

## What was built

Each component was evaluated as a **separate, one-factor experiment** against a matched baseline. They were not run together as one integrated system.

| Component | What it is | Document |
|---|---|---|
| Data pipeline | MIT-BIH v1.0.0, 48 records. 0.5–40 Hz band-pass filter, 216-sample beat windows, per-window z-scoring. 15 beat classes, 109,460 segments. Train/validation/test split by 47 inferred subject-equivalence groups. | [`preprocessing.md`](docs/preprocessing.md), [`dataset_splits.md`](docs/dataset_splits.md) |
| Centralized baseline | 1D CNN (10,127 parameters), frozen as the reference | [`baseline_freeze.md`](docs/baseline_freeze.md) |
| Federated learning | FedAvg over 5 simulated clients built from whole training groups; 20 rounds | [`federated_learning_baseline.md`](docs/federated_learning_baseline.md), [`fedavg_robustness.md`](docs/fedavg_robustness.md) |
| Differential privacy | Client-level DP-FedAvg (clipping + Gaussian noise, trusted server) with an exact Gaussian-DP accountant | [`differential_privacy.md`](docs/differential_privacy.md) |
| Homomorphic encryption | CKKS encrypted aggregation (TenSEAL / Microsoft SEAL, 128-bit parameters) | [`homomorphic_encryption.md`](docs/homomorphic_encryption.md) |
| Audit ledger | A single-node, SHA-256 hash-chained ledger that records each round's client and global model hashes, with an external anchor | [`blockchain.md`](docs/blockchain.md), [`blockchain_fl_integration.md`](docs/blockchain_fl_integration.md) |
| Attack study | One malicious client: sign flip, ×10 scaling, or norm-matched noise | [`malicious_clients.md`](docs/malicious_clients.md) |
| Proposed defense | Server-side update clipping plus cosine-similarity and norm-ratio screening against a robust median reference; parameters fixed on a separate calibration seed | [`proposed_method.md`](docs/proposed_method.md), [`ablation.md`](docs/ablation.md) |

## Key results

All results come from held-out test data evaluated once per run, with 3 seeds per condition. The numbers are **descriptive**: no significance testing was done. Full tables are in [`docs/final_results.md`](docs/final_results.md) and [`results/final/`](results/final/).

| Question | Finding in this setting |
|---|---|
| Centralized vs federated | The centralized CNN reached test accuracy 0.769 and macro-F1 0.140. All 37 clean federated runs scored lower (best: accuracy 0.712, macro-F1 0.105). Federated models mostly predict the two majority classes (`N`, `V`). |
| Differential privacy | Client-level DP at ε = 7.98, δ = 10⁻⁵ with only 5 clients **destroyed utility**: the noise was about 65× the signal. No other ε or privacy unit was tested. |
| Homomorphic encryption | Utility was preserved: aggregation error about 10⁻⁷. The cost was about **20.5×** the communication of plaintext FedAvg and about 2% extra compute. HE hides individual updates from the server but not the aggregate or the final model. |
| Audit ledger | Recording did not change training (bit-identical results). 363 of 363 recorded hashes verified against the real models. Every tamper attempt was detected **when an independently stored anchor is available**. Overhead was about 0.025% of training time. |
| Attacks on FedAvg | One malicious client scaling its update ×10 broke plain FedAvg in every seed. A sign-flipping client broke it within 6–12 rounds. Norm-matched noise was absorbed. |
| Proposed defense | On fresh confirmation seeds it **stopped the sign-flip and ×10 attacks** (100% detection, no honest client flagged, no model breakdown), at about 3.5 ms per round of server time and no extra communication. The clean-data cost was −0.004 macro-F1. The ablation showed the cosine filter stops sign flip and the norm filter stops scaling. Tested only against one non-adaptive attacker. |

![Study overview](results/final/figures/final_utility_overview.png)

## Status and open issues

The study is an experimentally complete prototype evaluation within its stated scope. It is **not** a research-complete answer to the research questions, because:

1. **The literature review is not documented.** [`docs/literature_matrix.md`](docs/literature_matrix.md) is still a template, so the design choices and the proposed defense are not yet positioned against prior work.
2. **The core classification task is weak.** Macro-F1 is 0.14, only 8 of 15 classes appear in the test set, the label taxonomy is provisional, and the federated `N`/`V` collapse is unexplained.
3. **Baselines and privacy evaluation are incomplete.** There is no local-only training baseline. Privacy was tested at a single ε, with no record-level DP and no combined DP + HE.
4. **The ledger is a prototype.** It runs on a single node: there is no permissioned network, consensus, signatures, or client authentication.
5. **The framework was never integrated.** DP, HE, the ledger, and the defense were evaluated in isolation, never together.
6. **Statistics are small.** There are 3 + 3 seeds and one centralized run, with no significance testing.

All 38 known limitations are in [`docs/final_limitations.md`](docs/final_limitations.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/preprocessing/` | Loading, filtering, segmentation, normalisation, splitting and validation |
| `src/models/` | 1D CNN, centralized training and evaluation |
| `src/federated/` | FedAvg, partitioning, diagnostics, DP (`privacy.py`), attacks, the proposed defense (`proposed_defense.py`), ablation and final validation |
| `src/he/` | CKKS encryption and encrypted aggregation |
| `src/blockchain/` | Hash-chained ledger, tamper tests, benchmarks and FL recorder |
| `src/final_report.py` | Builds the consolidated `results/final/` tables and figures |
| `configs/` | Frozen configuration for every experiment |
| `results/` | Per-experiment metrics, tables and figures. Consolidated results are in `results/final/`. |
| `docs/` | One document per experiment plus final results, methodology, limitations and reproducibility |
| `tests/` | 110 unit and integration tests |
| `notebooks/` | Exploration, validation and evaluation notebooks |

`experiments/`, `src/privacy/` and `src/algorithms/` are empty placeholders from the initial project layout. The corresponding code lives in the modules listed above.

## Reproducing the results

The MIT-BIH data is **not included**. Download MIT-BIH Arrhythmia Database v1.0.0 from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) into `data/raw/mitbih/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Step-by-step commands for every experiment, the seeds, the environment versions and the expected artifact hashes are in [`docs/reproducibility.md`](docs/reproducibility.md). The development environment was Python 3.12.10 and PyTorch 2.14.1 on CPU. Model checkpoints and per-run prediction files are regenerable and are not tracked in Git, except for the official centralized and FedAvg baselines.

## Documentation

- **Results and conclusions:** [`final_results.md`](docs/final_results.md)
- **Methodology:** [`final_methodology.md`](docs/final_methodology.md)
- **Limitations:** [`final_limitations.md`](docs/final_limitations.md)
- **Reproducibility:** [`reproducibility.md`](docs/reproducibility.md)
- **Day-by-day record:** [`research_log.md`](docs/research_log.md)
- **Dataset audits:** [`dataset_mitbih.md`](docs/dataset_mitbih.md), [`label_taxonomy_audit.md`](docs/label_taxonomy_audit.md), [`patient_mapping_audit.md`](docs/patient_mapping_audit.md)

## Research principles

The project followed these rules throughout:
- No result is fabricated; every reported number comes from a saved experiment output.
- The test set is evaluated once per run and never used for tuning.
- Claims are limited to what was measured. The defense is described as a *proposed* candidate built from established techniques, and no novelty or superiority is claimed.
