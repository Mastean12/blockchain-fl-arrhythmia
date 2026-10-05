# Day 14 — Homomorphic encryption for federated aggregation (HE-FedAvg)

- **Date:** 2026-10-05
- **Configuration:** [`configs/he/he_fedavg_v1.json`](../configs/he/he_fedavg_v1.json), which uses the unchanged base [`fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Code:** `src/he/`
  - `ckks.py`: parameters, session, encrypt / aggregate / decrypt
  - `aggregation.py`: HE-FedAvg round
  - `config.py`: validation
  - `report.py`: reporting
- **Integration:** the `homomorphic_encryption` branch in `src/federated/run_fedavg.py`
- **Outputs:** `results/he/`

> This is an isolated, one-factor experiment: the server aggregates encrypted client updates instead of plaintext ones. It is not combined with DP, blockchain, attacks, or the proposed algorithm, and none of those are implemented here.

## 1. Library and scheme

| Item | Value |
|---|---|
| Library | **TenSEAL 0.3.18** (OpenMined; Python bindings for Microsoft SEAL). It was not previously installed. It was added to the project `.venv` from an official prebuilt wheel (`cp312-win_amd64`) and pinned in `requirements.txt`. No other package changed. Pyfhel had no wheel for this platform. |
| Scheme | **CKKS** (approximate homomorphic arithmetic on real-valued vectors) |
| Polynomial modulus degree N | 8192, giving 4,096 slots per ciphertext |
| Coefficient modulus | [60, 40, 40, 60] bits, 200 bits in total |
| Scale | 2⁴⁰ |
| Security | 128-bit classical. 200 ≤ 218 bits, the maximum for N = 8192 in the HomomorphicEncryption.org standard (uniform ternary secret), as enforced by SEAL. |
| Encoding precision | About 40 fractional bits at encoding; observed relative error of decrypted aggregates ≤ 5.8·10⁻⁷ (§4) |
| Packing | 10,351 floating-point state entries per update, giving **3 ciphertexts** per client per round |

## 2. Protocol (per round)

1. **Clients (plaintext).** Each of the 5 clients trains for 1 local epoch exactly as in FedAvg. It then computes its update Δ_k = w_k − w_t over all floating-point state entries: weights, biases, BatchNorm affine parameters and running statistics. It encrypts Δ_k with the shared public key and serialises the ciphertexts.
2. **Server (encrypted).** The server holds only a **public context with no secret key**; a test verifies that it cannot decrypt. It deserialises the ciphertexts and computes **Enc(Σ_k p_k Δ_k)** by multiplying each ciphertext by the public weight p_k = n_k / Σn and adding the results. This is exactly sample-weighted FedAvg. The server returns the encrypted aggregate.
3. **Key holders.** The clients decrypt the aggregate, and every client sets w_{t+1} = w_t + Σ p_k Δ_k. Training then continues.

The integer BatchNorm counters stay in plaintext and are aggregated as in FedAvg; they are a public function of the sample counts. All other settings are identical to Day 11, and each run is paired by seed with the Day 11 natural FedAvg run:

- natural 5-client partition, seeds 42, 123, and 2024;
- 20 rounds, 1 local epoch, Adam lr 0.001, batch size 256;
- full participation;
- selection by minimum validation loss, and one test evaluation per run.

A test confirms that the configurations differ only by the `homomorphic_encryption` section. With HE disabled, the code still reproduces the official Day 10 model bit-identically. Combining HE with DP or local BatchNorm is rejected.

**What HE protects, and what it does not.**

- **Protected:** an honest-but-curious server never sees an individual client update in plaintext.
- **Not protected:**
  - The **decrypted aggregate and the global model are visible to all key holders**, exactly as in FedAvg. HE gives no protection against inference from the aggregate or the model, which is DP's role, and is not combined here.
  - All clients share one secret key, so a client that intercepted another client's ciphertext could decrypt it. Transport security, threshold or multi-key HE, and key distribution are not modelled.
  - Malicious behaviour (wrong ciphertexts, a dishonest server) is out of scope.

## 3. Correctness test (written and passed before any FL run)

Tolerance: **max |HE − plaintext| ≤ 10⁻⁵ · max(1, max |plaintext|)**. This was set after measuring a relative error of about 1.2·10⁻⁷ on vectors with |x| up to 10, so it leaves about 80× margin.

`tests/test_he.py` (9 tests) checks:

- a small weighted average;
- a model-sized, 3-chunk aggregation of 5 vectors with the real FedAvg weights;
- that `he_fedavg_aggregate` matches plaintext `fedavg_aggregate` on real `ECG1DCNN` states, entry by entry, with counters exact;
- that the server context has no secret key and cannot decrypt;
- that encryption is randomised;
- that parameters above the 128-bit limit are rejected;
- the HE config validation;
- that the study differs from FedAvg only by HE;
- that HE combined with DP is rejected.

## 4. Results

### Utility (test, evaluated once per run; paired with plaintext FedAvg)

| Seed | Arm | Selected round | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Macro AUROC | Predicted classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | FedAvg | 3 | 0.7069 | 0.0759 | 0.1177 | 0.0848 | 0.6574 | 0.6733 | `N` 10,602, `V` 4,957 |
| 42 | **HE** | 3 | 0.7069 | 0.0759 | 0.1177 | 0.0849 | 0.6575 | 0.6734 | `N` 10,601, `V` 4,958 |
| 123 | FedAvg | 2 | 0.6943 | 0.1117 | 0.1002 | 0.0986 | 0.5841 | 0.6463 | `N` 14,910, `V` 613, `R` 36 |
| 123 | **HE** | 2 | 0.6943 | 0.1117 | 0.1002 | 0.0986 | 0.5841 | 0.6463 | `N` 14,910, `V` 613, `R` 36 |
| 2024 | FedAvg | 17 | 0.5690 | 0.1364 | 0.1052 | 0.0717 | 0.5648 | 0.6226 | `N` 8,721, `V` 6,831, `R` 4, `L` 2, `/` 1 |
| 2024 | **HE** | 17 | 0.5935 | 0.0707 | 0.1077 | 0.0740 | 0.5817 | 0.6348 | `N` 9,069, `V` 6,486, `R` 2, `L` 2 |

**3-seed summary (mean ± sample SD; descriptive, no significance test):**

| Metric | HE-FedAvg | Plaintext FedAvg |
|---|---|---|
| Accuracy | 0.6649 ± 0.0621 | 0.6567 ± 0.0762 |
| Macro precision | 0.0861 ± 0.0223 | 0.1080 ± 0.0304 |
| Macro recall | 0.1085 ± 0.0088 | 0.1077 ± 0.0090 |
| Macro F1 | 0.0858 ± 0.0123 | 0.0851 ± 0.0134 |
| Weighted F1 | 0.6078 ± 0.0431 | 0.6021 ± 0.0489 |
| Macro AUROC | 0.6515 ± 0.0198 | 0.6474 ± 0.0254 |

**Seeds 42 and 123.** HE matches plaintext FedAvg to four decimals, with one prediction moved between `N` and `V`.

**Seed 2024.** Both arms select round 17, but the HE model differs: accuracy +0.025, macro-F1 +0.0023. The drop in macro precision (−0.066) comes from the single correct `/` prediction made by the plaintext model, which gives `/` a precision of 1.0. The HE model makes no `/` prediction. The minority-class collapse is unchanged in both arms: `/` and `L` recall is 0, and at least 99.95% of predictions are `N` or `V`.

### Numerical error (CKKS approximation)

| Quantity, per round, all seeds and 20 rounds | Value |
|---|---|
| Max absolute error of the decrypted aggregate against the exact float64 weighted sum | ≤ 1.30·10⁻⁷ |
| RMS error of the decrypted aggregate | about 3.5·10⁻⁹ (mean over rounds) |
| Max relative error (max abs error / max abs aggregate value) | ≤ 5.8·10⁻⁷ |
| Max absolute difference between the decrypted global state and plaintext FedAvg's float32 state from the same client states | ≤ 1.34·10⁻⁷ |

Every round passes the documented tolerance. The figure is `results/he/figures/he_numerical_error.png`.

**Trajectory sensitivity.** Per-round errors are at float32 precision, yet the paired HE and plaintext trajectories drift apart. Validation loss differs by more than 10⁻⁴ from round 2 onward, by up to 0.097 at most. At the selected rounds, the global states differ by up to 3.2·10⁻³ (seed 42, round 3), 5.3·10⁻⁴ (seed 123, round 2), and 0.15 (seed 2024, round 17). Validation macro-F1 differs by at most 0.0019 in any round.

A plausible mechanism, not tested here, is Adam's normalised step (lr · m / √v): for coordinates with near-zero gradients, a 10⁻⁷ perturbation can change the sign of the step, which moves a weight by about lr = 10⁻³. The seed-2024 difference should therefore be read as training's sensitivity to tiny perturbations, not as a systematic effect of HE. Encryption randomness comes from SEAL's unseeded internal CSPRNG, so HE runs are not bit-reproducible. The plaintext runs are.

### Computational overhead (measured locally, CPU)

| Operation | Time |
|---|---|
| Key and context generation (once per run) | about 0.37 s |
| Encryption, per client per round (3 ciphertexts) | 0.0173 ± 0.0003 s |
| Homomorphic aggregation, server, per round (5 clients × 3 chunks) | 0.0197 ± 0.0005 s |
| Decryption, per round | 0.0037 s |
| Total HE time over 20 rounds | 2.20 ± 0.04 s |
| End-to-end training time | 109–113 s against 110–120 s for plaintext FedAvg |

Wall-clock timing was not controlled (single machine, other load possible). HE adds about 2% of training time in this simulation. Encryption, aggregation, and decryption all run in one process, so real deployments would add serialisation over the network and per-party computation.

### Ciphertext size and communication overhead (measured serialised sizes)

| Quantity | HE-FedAvg | Plaintext FedAvg | Ratio |
|---|---:|---:|---:|
| Upload per client per round | 994,171 B (3 ciphertexts) | 41,428 B | **24.0×** |
| Download per client per round (encrypted aggregate, after rescaling) | 705,238 B | 41,428 B | **17.0×** |
| Total over 20 rounds, 5 clients | 169.94 MB | 8.29 MB | **20.5×** |
| One-time public context sent to the server (public key, parameters) | 1.86 MB | — | — |

The encrypted aggregate is smaller than the uploads because the scalar multiplication consumes one modulus level. The sizes are SEAL's default serialisation, which includes its built-in compression. Distributing the shared secret key to clients is not counted.

## 5. Interpretation

- **HE preserves FedAvg utility.** Encrypted aggregation reproduces plaintext FedAvg to within CKKS precision in every round (≤ 1.3·10⁻⁷). Test utility matches plaintext FedAvg in 2 of 3 seeds. The third differs by an amount within the Day 11 seed-to-seed spread, which is attributable to training sensitivity rather than systematic error. HE does not change the minority-class collapse, and was not expected to.
- **The cost is communication.** Each round moves about 20.5× more bytes than plaintext FedAvg (about 170 MB against 8.3 MB over 20 rounds), plus a one-time 1.9 MB public context. Compute overhead is small for this 10k-parameter model (about 0.11 s per round in total).
- **The privacy benefit is narrow and specific.** The server does not see individual updates. HE provides no formal guarantee about what the aggregate or the released model reveals, which DP addresses, and it relies on the key-management and honest-but-curious assumptions above.

These findings are for this small model, 5 clients, and these CKKS parameters. Larger models scale communication with the number of ciphertexts, about ⌈parameters / 4096⌉ per client per round.

## 6. Limitations

1. **Simulated key management.** All parties run in one process, and one shared secret key is held by all clients. There is no threshold or multi-key HE, no key distribution, and no transport security.
2. **Threat model.** The server is honest but curious. There is no protection against malicious clients or server, and no integrity checks on ciphertexts.
3. **Not bit-reproducible.** CKKS encryption randomness is unseeded, so HE runs reproduce only to within CKKS precision. The paired comparison is therefore affected by training's sensitivity to tiny perturbations.
4. **Timing and sizes.** Timing comes from one machine without control for load. Sizes depend on SEAL's serialisation and compression settings.
5. **Inherited limitations.** All Day 10/11 limitations still apply: a simulation on public data, 3 seeds, a 7-record test set with 8 of 15 classes supported, and inferred subject-equivalence groups.

## 7. Artifacts

- **Per run** (`he_fedavg_v1_natural_s{42,123,2024}`) under `results/he/{metrics,tables,figures}/federated/`:
  - `*_he_round_diagnostics.csv` (per-round timings, sizes, and errors)
  - `*_he_setup.json` (parameters, key-generation time, public-context size, TenSEAL version)
  - round history, test metrics, per-class metrics, confusion matrix, run summary
- **Summary tables:** `results/he/tables/`
  - `he_per_seed_utility_overhead.csv`
  - `he_vs_fedavg_3seed_summary.csv`
  - `he_round_diagnostics_all_seeds.csv`
  - `he_vs_fedavg_paired*.csv`
  - `he_vs_fedavg_prediction_counts.csv`
  - `he_vs_fedavg_per_class.csv`
  - `robustness_*.csv`
- **Figures:** `results/he/figures/`
  - `he_numerical_error.png`
  - `he_vs_fedavg_paired.png`
  - `robustness_convergence.png`
  - `robustness_test_metrics.png`
- **Not tracked (regenerable):** per-run checkpoints and `.npz` predictions (`.gitignore`).
- **Tests:** `tests/test_he.py`
