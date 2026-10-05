# Final methodology (research freeze, Day 20)

- **Study:** A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection
- **Freeze date:** 2026-10-05

This document consolidates the methodology as frozen across Days 2–20. Each day's full detail is in its own document, linked below. Nothing here supersedes a frozen configuration file. Where wording differs, the configuration file is authoritative.

## 1. Dataset

- **Source:** MIT-BIH Arrhythmia Database v1.0.0 (PhysioNet). 48 two-channel ambulatory ECG records at 360 Hz. Read only and not tracked in Git ([`dataset_mitbih.md`](dataset_mitbih.md)).
- **Subjects:** PhysioNet reports 47 subjects. Records 201 and 202 are documented as the same subject. This supports **47 inferred subject-equivalence groups**. Explicit patient identifiers are not available for every record, so this is **not** independently verified patient-level separation ([`patient_mapping_audit.md`](patient_mapping_audit.md)).
- **Annotations:** 23 raw annotation symbols. **15 beat symbols** are retained as model classes by identity mapping, with no merges. 8 non-beat/event symbols are excluded from beat windows ([`label_taxonomy_audit.md`](label_taxonomy_audit.md)).

## 2. Preprocessing ([`preprocessing.md`](preprocessing.md); config `configs/preprocessing/mitbih_v1.json`)

1. Zero-phase 4th-order Butterworth band-pass filter, 0.5–40 Hz (`sosfiltfilt`), applied per record.
2. One 216-sample window (0.6 s) per beat annotation: 72 samples before and 144 after. Windows that cross a record boundary are dropped (34 windows).
3. Per-segment, per-channel z-score normalisation (float32).
4. Provenance (record, group, symbol, sample offsets) is preserved in each per-record shard. This gives 109,460 segments of shape `[216, 2]`.

## 3. Split ([`dataset_splits.md`](dataset_splits.md); config `configs/preprocessing/mitbih_split_v1.json`)

- **Method:** seeded (42) random permutation of whole subject-equivalence groups, apportioned 70/15/15 by largest remainder. The split is not label-stratified.
- **Sizes:** train 33 groups (77,550 segments); validation 7 groups and 8 records (16,351; includes 201/202); test 7 groups and records (15,559; records 100, 101, 107, 113, 214, 219, 233).
- **Checks:** no record or group overlap and no duplicate segment+label across splits. Validated by `validate_saved_splits` on every day.
- **Test support:** only **8 of 15 classes** have positive test support.

## 4. Model and centralized baseline ([`baseline_freeze.md`](baseline_freeze.md), [`baseline_specification.md`](baseline_specification.md))

- **Model:** `ECG1DCNN`, three Conv1d–BatchNorm–ReLU blocks (2→16→32→64 channels, kernels 7/5/3), max-pooling, adaptive average pooling, dropout 0.3, and a 64→15 linear head. 10,127 trainable parameters.
- **Training:** Adam (lr 0.001), batch 256, unweighted cross-entropy, 5 epochs, seed 42, deterministic algorithms. The checkpoint is selected at minimum validation loss (epoch 1).
- **Evaluation:** the test set is evaluated **once**. Metrics are accuracy; macro precision, recall and F1 over all 15 classes (`zero_division=0`); weighted metrics; one-vs-rest specificity; and one-vs-rest AUROC (macro over defined classes, weighted, micro).

## 5. Federated learning protocol ([`federated_learning_baseline.md`](federated_learning_baseline.md))

- **Clients:** 5 simulated clients, formed **only from training groups**. The natural partition (`random_group_equal_count`) is a seeded permutation of the 33 training groups, cut into 7/7/7/6/6 whole groups with no balancing. The partition seed equals the run seed.
- **FedAvg:** full participation. Each round has 1 local epoch of Adam (lr 0.001, reset each round, batch 256). The server takes a sample-weighted average of the full `state_dict`. 20 rounds.
- **Initialisation:** the same as the centralized model.
- **Selection:** the global model with **minimum validation loss** over rounds 0–20. The test set is evaluated once per run.
- **Seeds:** evaluation seeds 42, 123 and 2024 (Days 11–19); calibration seed 7 (Day 18); fresh confirmation seeds 101, 202 and 303 (Day 20).
- **Reproducibility:** the plain FedAvg code path reproduces the official Day 10 model bit-identically. This was verified repeatedly through Day 20.

## 6. One-factor diagnostics and components

| Component | Design (frozen) | Document |
|---|---|---|
| FedBN-style diagnostic (Day 12) | Clients keep all BatchNorm state locally; conv and linear layers use FedAvg; evaluation uses shared weights plus averaged BatchNorm | [`fedbn_diagnostic.md`](fedbn_diagnostic.md) |
| Differential privacy (Day 13) | Client-level DP-FedAvg with a trusted server: per-client L2 clipping of the full float update (C = 1.0), fixed public weights, Gaussian noise z = 2.69 on the aggregate, δ = 10⁻⁵, q = 1, T = 20. Exact μ-GDP accountant gives **ε = 7.98** (RDP bound ≤ 9.36). | [`differential_privacy.md`](differential_privacy.md) |
| Homomorphic encryption (Day 14) | CKKS via TenSEAL 0.3.18 / SEAL: N = 8192, modulus [60, 40, 40, 60] (200 bits, 128-bit security), scale 2⁴⁰. Clients encrypt Δ_k; the server holds only the public context and computes Enc(Σ p_k Δ_k); key holders decrypt. | [`homomorphic_encryption.md`](homomorphic_encryption.md) |
| Blockchain ledger (Day 15) | Single-node, append-only SHA-256 hash chain over canonical JSON, recording `index → previous_hash → timestamp → round → participants {client_id, num_samples, update_hash} → aggregation_hash → block_hash`. Includes full validation and an external anchor (length + tip hash). Hashes, IDs and counts only. | [`blockchain.md`](blockchain.md) |
| Blockchain–FL integration (Day 16) | A read-only recorder hashes each uploaded client state and the global state every round (canonical tensor serialisation). States are archived off-chain for verification. | [`blockchain_fl_integration.md`](blockchain_fl_integration.md) |
| Attack model (Day 17) | One persistent malicious client out of 5 (seeded draw), full participation, non-colluding, true sample count. It sends sign flip (−Δ), scaled (10Δ) or norm-matched Gaussian noise over all float entries. The server runs plain FedAvg. | [`malicious_clients.md`](malicious_clients.md) |
| Proposed defense v1 (Day 18) | Trainable-parameter L2 clipping (C = 2.603), cosine to the coordinate-wise median (τ_cos = 0.1814), norm ratio to the median (τ_norm = 2.0), weight 0 for flagged clients, and BatchNorm statistics as the median of accepted clients with a 10⁻⁵ floor. Thresholds come from fixed rules on the honest statistics of a clean **seed-7** calibration run. | [`proposed_method.md`](proposed_method.md) |
| Ablation (Day 19) | Component switches with the frozen thresholds. Variant 5 is clipping + cosine + norm with FedAvg-style BatchNorm statistics over accepted clients. | [`ablation.md`](ablation.md) |
| Final confirmation (Day 20) | Variant 5 against FedAvg controls (the defense in observe mode, i.e. plain FedAvg) on fresh seeds 101, 202 and 303, under clean and the three Day 17 attacks. | [`final_results.md`](final_results.md) |

## 7. Assumptions

- **Simulation:** one machine, with clients as partitions of one public database. Clients are honest except the single modelled attacker on Days 17–20.
- **Server trust:** a trusted server for DP; an honest-but-curious server for HE; a trusted ledger writer plus an independently held anchor for blockchain integrity; a trusted server holding a clean validation split for round selection.
- **Public information:** client sample counts and FedAvg weights are treated as public.
- **Calibration:** defense thresholds come from one clean calibration seed and are not re-tuned.

## 8. Protocol rules applied throughout

- The test set was evaluated once per run and never used for selection or tuning.
- Every multi-seed study used predeclared seeds and configurations.
- Every intervention changed one factor relative to its comparator, and pairing by seed was verified in tests.
- No result from an earlier day was modified. Integrity was checked by SHA-256 hashes before and after each day.
