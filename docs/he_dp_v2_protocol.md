# Integrated HE+DP v2 Protocol

## Status and scope

`configs/he_dp_fedavg_v2.json` defines a versioned, fixed-roster follow-up to the historical `he_dp_fedavg_v1` study. The v1 configuration, three-seed outputs, and round-0 selection failure remain unchanged. V2 uses MIT-BIH's existing `mitbih_he_dp_v1` split and CNN. It is a simulation, not a deployment with process or network isolation.

The protocol and accountant implement a **conditional client-level guarantee** for recipients outside a trusted decryptor. They do not protect the trusted decryptor, the process orchestrating simulated clients, or client identity/participation. The run uses a seeded PyTorch Gaussian generator; the mathematical accounting assumes ideal independent Gaussian samples and does not establish cryptographic randomness. Do not describe this as a production privacy guarantee.

## Protocol

The public fixed roster is `client_00` through `client_04`, with public weights `p_k = 0.2`. All five slots participate every round. Weights are passed without renormalization; the accountant and aggregation code reject invalid sums, roster-size mismatches, or missing participants. The local dataset size does not affect the public aggregation weights.

For each client `k` at round `t`:

1. Client training produces a model delta from the previous global state. All floating-point state entries, including BatchNorm running means and variances, are flattened as one vector.
2. The client clips the complete vector to L2 norm `C = 1`: `u_k = delta_k * min(1, C / ||delta_k||_2)`.
3. The client encrypts `u_k` with the CKKS public context. The aggregation-only `CKKSServer` object has no secret key and computes `Enc(sum_k p_k u_k)` by public scalar multiplication and ciphertext addition.
4. The trusted decryptor, which owns the secret key, decrypts the clean weighted aggregate. It sees this clean aggregate and is explicitly outside the DP guarantee.
5. The trusted decryptor draws one Gaussian vector `N(0, sigma^2 I)` and adds it once to the clean aggregate before forming the next global model. The noised model is used for validation and later rounds. Negative BatchNorm running variances are clamped to zero as post-processing. Integer `num_batches_tracked` values are carried from the previous global state unchanged; they are neither encrypted nor aggregated from client data.
6. Validation metrics are computed from the noised model. Checkpoint selection uses minimum validation loss only. The held-out test split is evaluated once after selection.

No clean aggregate or noise realization is intentionally serialized. The private replay seed and clean-aggregate-dependent numerical diagnostics are written below `results/he_dp_v2/private/`, which is ignored by Git. Public outputs omit client sample counts, partition composition, client update norms, and clean-aggregate diagnostics.

## Privacy unit, adjacency, sensitivity, and accounting

The privacy unit is one participating client's **complete local dataset slot**. Neighboring federations have the same public five-slot roster, public weights, and all other client slots; one slot's clipped contribution is replaced by zero. This is an add/remove-one-client-slot definition, not record-level privacy and not a guarantee for arbitrary roster or weight changes.

Clipping guarantees `||u_k||_2 <= C`. Under fixed weights, removing slot `k` changes the clean aggregate by at most `p_k C`; therefore worst-case sensitivity is `S = C max_k(p_k) = 1 * 0.2 = 0.2`. V2 sets `sigma = z S`, where `z = 2.69`, so the aggregate Gaussian standard deviation is `0.538` per coordinate. The implementation computes sensitivity from the configured vector and rejects renormalization.

The five clients participate in all `T = 20` rounds (`q = 1`). Each round's adaptive Gaussian release has multiplier `z`; composition uses `mu = sqrt(T) / z`, then the analytic GDP-to-`(epsilon, delta)` conversion at `delta = 1e-5`. An independent RDP upper-bound calculation is emitted as a cross-check. There is no subsampling amplification. Under these stated mathematical assumptions the configured GDP conversion is expected to be approximately `epsilon = 7.98`; the recorded run artifact is authoritative for the exact value. This number is conditional on the stated neighbor relation, ideal Gaussian mechanism, fixed weights/roster, complete round accounting, and trusted-decryptor exclusion.

The accounted transcript must include **every round's noised global model**, since each result becomes the next round's starting model and is used for validation (and would be distributed to clients in a deployment), plus post-processing such as validation metrics and selection of the final checkpoint. Accounting only the selected checkpoint would not cover this protocol. It does not cover the clean aggregate observed by the decryptor, client identities, public roster, update timing, implementation side channels, or diagnostics kept by the trusted execution role. A single-process runner retains plaintext client states in its orchestrator; the role objects demonstrate API-level key separation only and do not enforce process isolation. A compromised orchestrator/decryptor, collusion, secret-seed disclosure, or non-Gaussian implementation invalidates the stated threat assumptions.

## Key and access boundaries

- The experiment generates one CKKS keypair in the training process. Clients and the aggregation server receive public encryption material through role views. The decryptor is the only intended role with the secret key.
- The server view receives serialized ciphertexts and public weights, and cannot decrypt them. The trusted decryptor can decrypt the aggregate and sees the clean aggregate before noise.
- The seeded noise generator and seed are held by the trusted-decryptor code path. The seed file is private local reproducibility material, not a secure key-management system.
- The orchestration process currently creates and holds client update state objects in memory. The simulation therefore does not prove that an independently controlled server process cannot inspect those objects. Deployment isolation, distributed key ownership, collusion resistance, and secure entropy remain unimplemented.
- The released checkpoint is DP-protected only under the assumptions above. CKKS protects ciphertext values from a party without the secret key under the scheme's assumptions; it does not provide DP. DP does not encrypt the clean update or guarantee confidentiality from a decryptor.

## BatchNorm and numerical behavior

All floating-point BatchNorm state participates in clipping, encrypted aggregation, and aggregate noise. Running-variance clamping is post-processing. Integer tracking counters remain at the public initial value for all rounds. The protocol is incompatible with the existing local-BatchNorm aggregation mode, which remains rejected for privacy experiments. CKKS approximation error is measured against the clean weighted aggregate and retained only in the private diagnostics artifact; public cost records contain timings, dimensions, and serialized byte counts.

### Combined-run numerical evidence and accounting boundary

In the preserved combined run, the private per-round diagnostics report maximum absolute CKKS clean-aggregate error up to `8.016845931546612e-8`; per-round RMS error ranges from `3.0737774066477107e-9` to `3.170211994490261e-9`. This is an empirical comparison to the exact weighted clipped sum for that run, not a proven worst-case CKKS error bound. The GDP/RDP accountant models an ideal Gaussian mechanism under the stated fixed-weight sensitivity; it does **not** incorporate a proved bound for CKKS approximation, finite-precision state conversion, or the seeded PRNG's distribution. Thus `epsilon=7.979543723560064` is a conditional mathematical accounting result for the idealized mechanism and assumptions, not a formally established end-to-end implementation guarantee. No HE+DP parameter tuning or replacement run is authorized by this evidence.

The combined run's historical summary records `config.output_root` as `results/he_dp_v2_blockchain_lifecycle`, while its `model_path` and actual artifacts are under `results/he_dp_v2_blockchain_lifecycle_recovered_20261016`. Preserve that summary as historical evidence; do not edit it. The runner now records the effective output root for future runs, and the one-round integration test asserts that metadata. The privacy configuration and recorded result are otherwise preserved.

The frozen config's `release_set` description names selected checkpoints and their validation metrics, but the actual adaptive protocol consumes each round's noised global model and evaluates it; a deployment would also broadcast those round models. The accountant composes all 20 rounds, which is the relevant transcript scope. The old config and historical output are preserved; an approved, versioned design should make this release set explicit before any new HE+DP experiment. Do not interpret the config wording as approval of a narrower privacy scope.

## Property and leakage boundary

| Property | Mechanism/evidence in this simulation | Does not establish |
|---|---|---|
| Confidentiality of client updates in transit to the HE server | CKKS ciphertexts and server role without the secret key; ciphertext sizes and empirical aggregate approximation error are measured. | Confidentiality from the trusted decryptor, the in-process orchestrator, or an attacker with the secret key; enforced deployment isolation or a proven end-to-end approximation bound. |
| Statistical privacy of released model updates | Client-slot add/remove adjacency, whole-update L2 clipping, one aggregate Gaussian vector per round, and full-participation GDP accounting. | Record-level privacy, hiding client identity/participation, protecting the decryptor's clean aggregate view, or a deployment guarantee for the seeded PRNG and finite-precision CKKS implementation. |
| Integrity of submitted payloads and accepted global state | The lifecycle gate checks committed update hashes before aggregation and the noised global-state hash before validation/checkpoint selection. | Authentication of client identities, correctness of a malicious coordinator, independent availability, or an immutable chain without a separately held anchor. |
| Availability and recovery | Local artifacts support reproducibility; an interrupted prepared round aborts. | Replication, consensus, recovery from a failed node, or protection from deletion of the ledger and archive. |

The public ledger reveals the configured roster, client IDs, round participation, public fixed weights, update hashes, protocol metadata, and global-model hashes. It does not reveal actual sample counts or store client update tensors/ciphertexts. Global model states are kept in the off-chain archive and final checkpoint; this policy is not a claim that released models contain no information about training data. Private reproducibility seeds and clean-aggregate-dependent diagnostics stay in the ignored `private/` result directory. These boundaries should be reviewed again before changing the release set.

## Tests and reproducibility

Run:

```powershell
python -m unittest discover -s tests -p test_he_dp_v2.py -v
python -c "from src.federated.run_fedavg import run_experiment; run_experiment('configs/he_dp_fedavg_v2.json', evaluate_test=True)"
```

The v2 tests check fixed-weight sensitivity and rejection of renormalization, GDP/RDP cross-checking, public-only client/server contexts, clipping before encryption, aggregate-noise placement and calibration, CKKS numerical error, BatchNorm counters, integrated execution in a single training run, private diagnostics, and validation-only checkpoint selection.

## Comparison with v1

V1 clips and noises each client update before encryption, then uses sample-count-derived weights. Its three measured runs all selected round 0; preserve the recorded poor utility. V1's sensitivity/accounting question is whether its data-derived weights and client-level adjacency match the stated fixed-weight assumptions. V2 instead fixes public weights, encrypts clipped but unnoised deltas, decrypts one weighted sum at a trusted decryptor, and adds one Gaussian vector to that aggregate. Under add/remove-one-slot adjacency the sensitivity is `C max(p)`. V2 shifts trust: the decryptor sees the clean aggregate, and is excluded from the guarantee. The v2 result must not be interpreted as a fix to v1 or as evidence that HE alone provides DP.

## Remaining approval and implementation gaps

- The configured fixed roster, uniform weights, add/remove-one-slot adjacency, trusted-decryptor trust, and complete release set are the v2 design basis supplied for this sprint; the researcher should review these assumptions before relying on the epsilon value.
- No deployable role isolation, distributed key management, cryptographically secure noise generator, or protection from a compromised orchestrator is implemented. Consequently, this is evidence for the mathematical protocol in a trusted simulation, not a deployment-level privacy claim.
- The acceptable predictive-utility threshold remains an **OPEN DECISION**. The v2 run is reported as observed and is not tuned on test results.
- The blockchain-enabled v2 configuration is [`configs/blockchain/he_dp_fedavg_v2_lifecycle.json`](../configs/blockchain/he_dp_fedavg_v2_lifecycle.json), with a distinct output root at `results/he_dp_v2_blockchain_lifecycle/`. It invokes this same aggregation function and accountant; it does not replace or overwrite the existing v2 experiment or its three-seed result.
- In the combined path, CKKS encrypts each clipped client update, the server homomorphically computes the weighted sum, and the trusted decryptor decrypts the clean aggregate then adds the aggregate Gaussian noise. The lifecycle ledger commits client IDs, update hashes, protocol metadata, fixed public weights, and the previous/global model hashes. It records no actual client sample counts and archives no client update tensors. This does not change the privacy unit, adjacency, accountant, noise placement, or trusted-decryptor exclusion described above.
- HE and DP remain distinct properties: CKKS protects encrypted values from roles without the secret key under the scheme and key assumptions; DP provides the accountant's conditional guarantee for its stated adjacency and releases. Neither property implies the other.
- The blockchain is a single-node hash-chained, tamper-evident log, not a consensus-based distributed blockchain. The trusted decryptor limitation and lack of enforced process isolation remain.
- Run the distinct combined experiment from the repository root with:

  ```powershell
  $env:PYTHONPATH = "$PWD\tests;$PWD"
  .\.venv\Scripts\python.exe -c "from src.federated.run_fedavg import run_experiment; run_experiment('configs/blockchain/he_dp_fedavg_v2_lifecycle.json', evaluate_test=True, output_root='results/he_dp_v2_blockchain_lifecycle_recovered_20261016', overrides={'data': {'split_root': 'data/splits/mitbih_v1'}})"
  ```

## Combined 20-round run (2026-10-16)

The configured 20-round run completed successfully with test evaluation enabled. The configured `data/splits/mitbih_he_dp_v1` directory and its regenerated processed-data source were inaccessible to the current Windows identity. They were left unchanged. Instead, the run used the readable `data/splits/mitbih_v1` copy: its `split_metadata.json` SHA-256 is `229b0db300778b0e5203b7d13537840410d52ce86752279f4e419195596831c2`, matching the documented frozen metadata digest; its split config exactly matches `configs/preprocessing/mitbih_split_v1.json`; and `validate_saved_splits` passed on the shards (48 records, 47 groups, 109,460 segments, with the saved overlap, schema, label, and duplicate checks passing). This was a runtime path override, not a regenerated or altered split. Because the configured output root already contained residue from the earlier access-denied attempt, the completed run used the separate run-specific output root in the command above. No existing result was overwritten.

The run took 127.16 seconds and selected round 0 by validation loss. The held-out test set was evaluated once after selection (`test_data_used_for_training_or_model_selection: false`). The resulting initial-checkpoint metrics were accuracy `0.6554`, macro-F1 `0.0528`, weighted-F1 `0.5189`, and macro AUROC `0.4244` over 8 of 15 classes with test support. These are weak macro results and do not demonstrate a useful trained detector; they must not be presented as evidence of clinical utility.

The run's recorded privacy accounting was epsilon `7.979544` at delta `1e-5`, sensitivity `0.2`, and aggregate noise standard deviation `0.538` for 20 full-participation rounds. This remains conditional on the protocol assumptions above, including excluding the trusted decryptor that sees the clean aggregate.

The ledger contains 41 blocks (genesis plus 20 prepared/completed round pairs), is 48,553 bytes, and validates against its saved anchor. The selected checkpoint hash matches the round-0 ledger hash. All 21 off-chain archive files contain only the global state; no client update tensors are archived, and no actual client sample counts appear in the ledger. Prepared commitments include five client IDs, update hashes, and the configured `0.2` public weights. At completion, mean per-round HE costs were 79.70 ms encryption, 18.52 ms server aggregation, 3.35 ms trusted decryption, and 0.265 ms aggregate-noise generation. Blockchain phase timings and ledger bytes are separately recorded in the output metrics. These are observed simulation timings, not network measurements. The anchor is saved beside the ledger in the same result tree; no independent custody is evidenced.

The output inventory and raw measured artifacts are under `results/he_dp_v2_blockchain_lifecycle_recovered_20261016/`; the summary, test metrics, privacy accounting, HE costs, blockchain timing/summary, ledger, anchor, checkpoint, and global-only archives should be treated as the authoritative run evidence. The original configured output root was not altered by this completed run.
