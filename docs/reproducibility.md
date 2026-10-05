# Reproducibility guide (research freeze, Day 20)

## 1. Environment

| Item | Version |
|---|---|
| OS used | Windows 11 Pro for Workstations (CPU only) |
| Python | 3.12.10 (project `.venv`) |
| PyTorch | 2.14.1 (CPU) |
| NumPy / SciPy / pandas | 2.5.3 / 1.18.1 / 3.0.6 |
| scikit-learn | 1.9.1 |
| WFDB | 4.3.1 |
| matplotlib / seaborn | 3.11.2 / 0.13.2 |
| TenSEAL (HE only) | 0.3.18 (Microsoft SEAL backend; official `cp312-win_amd64` wheel) |

All direct dependencies are pinned in [`requirements.txt`](../requirements.txt). Install them with `python -m pip install -r requirements.txt` inside `.venv`. No GPU was used, and all training runs set `torch.use_deterministic_algorithms(True)`.

## 2. Data (not tracked in Git)

1. Download MIT-BIH Arrhythmia Database v1.0.0 from PhysioNet into `data/raw/mitbih/` (48 records with `RECORDS`, `.hea`, `.dat`, `.atr`, and `mitdbdir/`).
2. Preprocess: `src.preprocessing.pipeline.process_dataset("data/raw/mitbih", "data/processed/mitbih_v1", "configs/preprocessing/mitbih_v1.json")` (or `python -m src.preprocessing.pipeline`) writes 109,460 segments.
3. Split: `src.preprocessing.splitting.build_splits("data/processed/mitbih_v1", "data/splits/mitbih_v1", "configs/preprocessing/mitbih_split_v1.json", "configs/preprocessing/mitbih_v1.json")` writes the split. Validate with `validate_saved_splits("data/splits/mitbih_v1", "configs/preprocessing/mitbih_v1.json")`.
4. Expected hash: `data/splits/mitbih_v1/split_metadata.json` SHA-256 `229b0db300778b0e5203b7d13537840410d52ce86752279f4e419195596831c2`.

## 3. Re-running each experiment

Every runner below skips completed runs and loads them instead, so the test set is never re-evaluated by accident. `run_experiment` also refuses to overwrite an existing run.

| Day | Experiment | Command (from the repository root, inside `.venv`) |
|---|---|---|
| 5–6 | Centralized CNN and test evaluation | `python -m src.models.train_cnn`, then `python -m src.models.evaluate_cnn` |
| 10 | Official FedAvg | `python -m src.federated.run_fedavg` |
| 11 | FedAvg robustness | `python -c "from src.federated.robustness import run_all, write_reports; run_all(); write_reports()"` |
| 12 | FedBN diagnostic | `run_all("configs/federated/fedbn_diagnostic_v1.json")`, `write_reports(...)`, then `src.federated.diagnostics.write_diagnostic_reports()` |
| 13 | DP-FedAvg | `run_all("configs/privacy/dp_fedavg_v1.json")`, then `src.federated.privacy_report.write_privacy_reports()` |
| 14 | HE-FedAvg | `run_all("configs/he/he_fedavg_v1.json")`, then `src.he.report.write_he_reports()` |
| 15 | Ledger benchmark | `python -c "from src.blockchain.benchmark import run_benchmarks; run_benchmarks()"` |
| 16 | FedAvg + ledger | `run_all("configs/blockchain/fedavg_blockchain_v1.json")`, then `src.blockchain.integration_report.write_integration_reports()` |
| 17 | Attacks | `src.federated.security.run_all()`, then `write_security_reports()` |
| 18 | Proposed defense | calibration per `configs/proposed/calibration_v1.json` (seed 7, observe mode, `evaluate_test=False`), then `src.federated.proposed_study.run_all()` and `write_reports()` |
| 19 | Ablation | `src.federated.ablation.run_all()`, then `write_reports()` |
| 20 | Final confirmation and consolidation | `src.federated.final_validation.run_all()`, then `python -m src.final_report` |

In the table, `run_all` and `write_reports` without a module refer to `src.federated.robustness`.

**Runtime.** A 20-round FedAvg run takes about 110 s on the CPU used. Day 19's 48 runs took about 90 minutes, and Day 20's 24 runs about 48 minutes.

## 4. Seeds

| Use | Seeds |
|---|---|
| Split | 42 |
| Centralized training | 42 |
| Official FedAvg | 42 |
| Multi-seed evaluation (Days 11–19) | 42, 123, 2024 (training seed = partition seed) |
| Defense calibration (Day 18) | 7 |
| Fresh confirmation (Day 20) | 101, 202, 303 |
| Attacker draw | `default_rng([seed, 1717]).integers(5)` |
| Attack noise | `seed · 1,000,033 + 31337 + round` |
| DP noise | `seed · 1,000,003 + 7919 + round` |
| Client shuffling and dropout | `seed · 100,000 + 1,000 · round + client` |

HE (Day 14) uses SEAL's unseeded CSPRNG, so it reproduces only to within CKKS precision.

## 5. Determinism checks performed

- The plain FedAvg path reproduces the official Day 10 round-3 model **bit-identically**. This was re-checked after every code change on Days 11–20, including the observe-mode control on Day 20.
- The seed-42 natural robustness run equals the official run bit-identically (Day 11).
- Blockchain recording equals plain FedAvg bit-identically (Day 16).
- The default defense components equal Day 18 bit-identically (Day 19).
- Day 12's reconstructed `diagnostics.py` reproduces the FedBN tables exactly (commit split).

## 6. Tracked versus regenerable artifacts

- **Tracked:** configurations, code, tests, docs, all per-run CSV/JSON metrics and tables, summary figures, the official centralized and FedAvg checkpoints and predictions, ledgers and anchors, and `results/final/`.
- **Not tracked (regenerable):** raw and processed data, split shards, per-run checkpoints (`*/models/`), per-run `.npz` predictions and trajectories, the Day 16 off-chain model archive, and per-run figures for Days 19–20.
- **Ignore rules:** see `.gitignore`.
- **Hashes:** the hashes of key frozen artifacts are recorded in `results/final/experiment_manifest.json`.

## 7. Tests

`python -m unittest discover -s tests` runs from the repository root. Some integration tests run the real pipeline on a small synthetic split and read frozen baseline files by repository-relative path.
