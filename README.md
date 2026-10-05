# A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection

## Research aim

To develop and evaluate a blockchain-enabled federated learning framework for privacy-preserving ECG-based arrhythmia detection. The research investigates collaborative model training across institutions without exchanging raw ECG records, and evaluates the benefits and costs of federated learning, privacy mechanisms, and blockchain coordination.

## Objectives

1. Develop an ECG-based deep learning model for arrhythmia detection using publicly available ECG datasets.
2. Design a federated learning architecture that supports collaborative training without exchanging raw ECG data.
3. Investigate differential privacy and homomorphic encryption as privacy-preserving mechanisms.
4. Design a permissioned blockchain coordination and verification mechanism for federated model updates.
5. Develop and evaluate a proposed aggregation and/or coordination method addressing a limitation identified through literature review.
6. Compare the framework with appropriate centralized, federated, privacy-preserving, and blockchain-enabled baselines.
7. Evaluate predictive performance, privacy, computational cost, communication overhead, blockchain overhead, convergence, and robustness to relevant attacks.

These are working objectives. Architectural and algorithmic choices will be justified by literature, research questions, and experimental requirements. No novelty or performance claims will be made without supporting evidence.

## High-level architecture

```text
Distributed ECG data at participating institutions
  -> local ECG preprocessing and local model training
  -> privacy mechanism (differential privacy and/or homomorphic encryption)
  -> permissioned blockchain coordination, verification, and audit trail
  -> federated aggregation and candidate proposed coordination method
  -> global model returned to participating clients
  -> evaluation against centralized, local, and federated baselines
```

Raw ECG data remains at participating institutions in the federated setting. Experiments will measure utility alongside privacy, computational and communication costs, blockchain overhead, convergence, and robustness.

## Repository layout

- `data/`: raw, processed, and split data artifacts (datasets are not included).
- `notebooks/`: exploratory and analysis notebooks.
- `src/`: research code, organized by preprocessing, models, federated learning, privacy, blockchain, and algorithms.
- `experiments/`: baseline, federated, privacy, blockchain, and ablation experiments.
- `results/`: generated figures, tables, and metrics.
- `docs/`: research notes and literature review.
- `tests/`: verification and regression tests.

## Research environment

Use Python 3.12 and the repository virtual environment. In VS Code, select `.venv` as the Python interpreter. Install the pinned dependencies with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Datasets are not included in the repository. The current status is a frozen centralized baseline, defined in [`docs/baseline_freeze.md`](docs/baseline_freeze.md); progress is recorded in [`docs/research_log.md`](docs/research_log.md).
