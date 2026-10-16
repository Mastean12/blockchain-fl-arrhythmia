# Integrated HE+DP federated-learning extension

## Status and scope

This is a post-freeze extension to the separately evaluated Day 11 FedAvg, Day 13 client-level DP, Day 14 CKKS, and Day 16 ledger experiments. It uses a new versioned configuration, `configs/he_dp_fedavg_v1.json`, and a separate output root, `results/he_dp/`. It does not rewrite those historical experiments.

The round implementation is exercised together by `tests/test_he_dp_integration.py` on a synthetic split and by a three-seed MIT-BIH study. The original split directory is inaccessible and its tracked files were already marked deleted in the workspace. I regenerated preprocessing and splits under `data/processed/mitbih_he_dp_v1/` and `data/splits/mitbih_he_dp_v1/`; the new metadata SHA-256 exactly matches the committed frozen metadata (`229b0db300778b0e5203b7d13537840410d52ce86752279f4e419195596831c2`) and split validation passes. The new experiment config points to this validated, exact-membership copy; the original directory was not changed.

## Protocol

For each full-participation round:

1. Each client trains locally and forms its state delta from the broadcast global state. Raw ECG examples remain client-local.
2. Each client clips the L2 norm of the concatenated floating-point update (including BatchNorm affine values and running statistics) to `C=1.0`.
3. Client `k` draws independent Gaussian noise with standard deviation `sigma_k = z*C*max(p)/p_k`, where `p_k` is its fixed, public sample weight and `z=2.69`. It adds that noise to its clipped update **before encryption**. The integer BatchNorm counter remains a plaintext weighted function of public counts.
4. The client encrypts the noisy update with CKKS. The simulated clients share one secret key; the aggregation-only server view contains only a public context and has no decrypt method or secret context.
5. The server multiplies ciphertexts by public FedAvg weights and sums them. The server receives ciphertexts and public IDs/counts; it does not receive individual plaintext updates or the DP noise values.
6. Key holders decrypt the noisy weighted aggregate and apply it to the global state. CKKS approximation error is measured against the exact weighted sum of the already-noised client vectors in the receiver-side simulation.

Independent noise shares mean each client's weighted noise contribution has standard deviation `z*C*max(p)`. Even if the other clients' noise shares are known, the remaining target-client Gaussian noise has multiplier `z*max(p)/p_k >= z` relative to that client's clipped update sensitivity `C`. The accountant therefore uses the minimum multiplier `z`, full participation, and all `T=20` rounds without subsampling amplification.

## Privacy accounting and trust assumptions

- Privacy unit: one simulated client; add/remove-one-client contribution with all other public sample weights fixed.
- Accountant: the existing full-participation GDP composition and analytic Gaussian conversion, independently tested for the configured round count, noise multiplier, and delta. With `T=20`, `z=2.69`, and `delta=1e-5`, it reports approximately `epsilon=7.98`.
- Noise seeds: generated from OS entropy and held in `results/he_dp/private/` so a local run can be replayed. The directory is excluded from Git. Seed secrecy is necessary: anyone who learns these seeds can reconstruct the noise and invalidate the intended DP guarantee. Public run summaries contain neither the seeds nor their values.
- Threat model: honest-but-curious aggregation server without a secret key; clients follow the clipping/noise protocol and protect their seed. The accountant covers the released noised updates/model under its stated adjacency and Gaussian assumptions. It does not protect against seed disclosure, implementation side channels, malicious clients, server/client collusion with seed access, or a compromised key holder outside the stated DP mechanism.
- HE keys: all clients share one secret key in this single-process simulation. Any holder of that key can decrypt ciphertexts it obtains. The design does not model secure key distribution, threshold decryption, transport security, authenticated identities, or a real multi-process network. CKKS confidentiality is therefore only claimed against the aggregation-only server view under the HE assumptions.
- DP noise uses a reproducible PyTorch generator seeded from private entropy material; formal accounting models ideal independent Gaussian draws and does not model finite-precision or PRNG imperfections. This caveat limits deployment claims.

HE provides ciphertext confidentiality for the server's aggregation operation; DP provides the stated client-level bound for Gaussian-perturbed updates. Neither mechanism implies the other. Because this protocol noises each client update locally, its noise/utility is not directly interchangeable with the historical trusted-server DP-FedAvg experiment.

## Measurements and outputs

Each run writes the existing run summary and test metrics plus per-round HE diagnostics (encryption, aggregation and decryption time, ciphertext bytes, CKKS error) and a privacy-accounting record. Classification metrics are written only if the held-out test split is available and evaluated. Client-level training losses and update norms are suppressed in this integrated arm to avoid adding unaccounted data-dependent releases. The current automated integration test verifies that the same training round writes both HE and DP diagnostics, that the privacy accountant is present, that the server role has no secret key/decrypt method, and that CKKS error is bounded.

## Measured MIT-BIH study

The configured three-seed study was run with the exact frozen split membership. Every seed selected round 0 by validation loss, so the reported test scores are the initial model and show that this configured HE+DP training did not produce a selected trained checkpoint. In seeds 123 and 2024, the initial model predicts only one class or otherwise has extremely poor classification utility. This is a negative result and must not be presented as evidence of useful arrhythmia detection. The mean and sample standard deviation are descriptive only; no significance claim is made.

| Metric | Mean ± sample SD across seeds 42, 123, 2024 |
|---|---:|
| Accuracy | 0.246760 ± 0.355630 |
| Macro F1 | 0.023498 ± 0.025377 |
| Weighted F1 | 0.179842 ± 0.293666 |
| Macro AUROC | 0.546224 ± 0.108361 |

All three runs report `epsilon=7.979544` at `delta=1e-5` under the client-level accountant and assumptions above. Mean end-to-end training time was 128.758 seconds. Mean maximum CKKS absolute error per run was `8.968e-7`. Mean total traffic counting all client uploads and all client downloads was 169,940,682 bytes per run (about 20.51 times the historical plaintext FedAvg traffic of 8,285,600 bytes under its recorded accounting). This estimate excludes the approximately 1.86 MB public CKKS context setup per run. See [`results/he_dp/tables/he_dp_comparison_v1.csv`](../results/he_dp/tables/he_dp_comparison_v1.csv), [`results/he_dp/tables/he_dp_cost_privacy_v1.csv`](../results/he_dp/tables/he_dp_cost_privacy_v1.csv), and [`results/he_dp/tables/robustness_runs.csv`](../results/he_dp/tables/robustness_runs.csv) for per-seed measurements and historical comparison. In this implementation, HE+DP scores equal the historical DP arm because the selected checkpoint for every seed is round 0; this is not evidence of equivalence between the protocols.

The study establishes execution and records costs, but its utility failure means the integrated configuration is not a successful detection model. Do not tune this version in place or overwrite these outputs; any follow-up should use a new versioned configuration and output directory, investigate the failure, and preserve these results.

Reproduce the configured three-seed MIT-BIH study with:

```powershell
python -c "from src.federated.robustness import run_all; run_all('configs/he_dp_fedavg_v1.json')"
```

Retain the ignored private-seed artifact under `results/he_dp/private/` for exact reruns; do not distribute it.
