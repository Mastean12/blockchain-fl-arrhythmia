# Day 16 — Blockchain ledger integrated with FedAvg rounds

- **Date:** 2026-10-05
- **Configuration:** [`configs/blockchain/fedavg_blockchain_v1.json`](../configs/blockchain/fedavg_blockchain_v1.json), which uses the unchanged base [`fedavg_v1.json`](../configs/federated/fedavg_v1.json)
- **Code:**
  - `src/blockchain/fl_recorder.py`: state hashing, recorder, off-chain verification
  - the `blockchain` branch in `src/federated/run_fedavg.py`
  - `src/blockchain/integration_report.py`
- **Ledger:** the Day 15 component (`src/blockchain/ledger.py`)
- **Outputs:** `results/blockchain_integration/`
- **Tests:** `tests/test_blockchain_fl_integration.py`

> The Day 15 and Day 16 plain-FedAvg experiments remain unchanged. Their recorder path does not involve DP, HE, attacks, or the proposed algorithm; the separate HE+DP v2 lifecycle experiment is documented below.

## 1. What is recorded

After aggregation in each FedAvg round, one block is appended with:

| Field | Source |
|---|---|
| `round` | FedAvg communication round. Genesis is round 0 and records the initial model hash. |
| `participants[].client_id` | Participating simulated clients (`client_00` … `client_04`) |
| `participants[].num_samples` | Each client's training-sample count, the FedAvg weight |
| `participants[].update_hash` | SHA-256 of the **model state the client uploaded** that round (what plaintext FedAvg transmits) |
| `aggregation_hash` | SHA-256 of the aggregated global model state |
| `timestamp`, `previous_hash`, `block_hash` | Ledger fields from Day 15 |
| `metadata` | `{"aggregation": "FedAvg", "participants": 5}`. Genesis also stores the seed and partition strategy. |

**State hash.** SHA-256 over a canonical serialisation of the state dict: for each entry, the key, dtype and shape, then the raw little-endian values. It covers weights, biases, BatchNorm parameters, running statistics and counters, so it is independent of `torch.save` file layout. Any single-value change alters it (tested down to a change of 10⁻⁶).

**Off-chain.** The exact client states and global state of every round are saved to `results/blockchain_integration/offchain/<run>/round_NNN.pt` (about 5.7 MB per run). This archive is not tracked in Git and can be regenerated. It is what the on-chain hashes are verified against. ECG data never leaves the clients and is never hashed or recorded. No model parameters or (encrypted) updates enter the ledger.

**Recording cannot change training.** The recorder only reads tensors: no in-place changes and no random-number use. The "without blockchain" arm is the Day 11 natural FedAvg run with the same seed, which uses the same code path, partition, initialisation and training order. With blockchain recording disabled, the code still reproduces the official Day 10 model bit-identically.

## 2. Results (seeds 42, 123, 2024; natural 5-client partition; 20 rounds)

### Model-performance equivalence

| Seed | Selected round (with / without) | Test accuracy | Test macro-F1 | Test weighted-F1 | Selected model | Test predictions | Round history | Test metrics |
|---|---|---:|---:|---:|---|---|---|---|
| 42 | 3 / 3 | 0.7069 | 0.0848 | 0.6574 | bit-identical | identical | identical | identical |
| 123 | 2 / 2 | 0.6943 | 0.0986 | 0.5841 | bit-identical | identical | identical | identical |
| 2024 | 17 / 17 | 0.5690 | 0.0717 | 0.5648 | bit-identical | identical | identical | identical |

"Round history" excludes wall-clock elapsed time. FedAvg with blockchain recording is **exactly** equivalent to FedAvg without it.

### Verification of recorded updates and models

| Seed | Chain valid (with anchor) | On-chain hashes verified against off-chain artifacts | Selected checkpoint hash matches its block |
|---|---|---|---|
| 42 | yes | 121 / 121 (100 client states + 21 global states) | yes (round 3) |
| 123 | yes | 121 / 121 | yes (round 2) |
| 2024 | yes | 121 / 121 | yes (round 17) |

Recorded sample counts equal the partition's client segment counts (integration test). The saved global checkpoint used for test evaluation is the global state recorded on-chain for the selected round. That links the reported test result to the ledger.

### Overhead (local wall clock, uncontrolled load; mean per round over 20 rounds)

| Quantity | Seed 42 | Seed 123 | Seed 2024 |
|---|---:|---:|---:|
| Hashing 6 states (5 clients + global), ms/round | 1.37 | 1.21 | 1.27 |
| Hashing per state, ms | 0.23 | 0.20 | 0.21 |
| Block creation and append, ms/round | 0.135 | 0.123 | 0.131 |
| **Hashing + blocks, whole run** | **0.031 s** | **0.027 s** | **0.028 s** |
| Share of training time | 0.026% | 0.024% | 0.024% |
| Full-chain validation (21 blocks, once per run), ms | 1.11 | 1.43 | 1.37 |
| Ledger size (21 blocks) | 22,250 B | 22,272 B | 22,294 B |
| Ledger bytes per round block | 1,074 | 1,075 | 1,076 |
| Off-chain archive save, ms/round (separate cost) | 5.4 | 4.8 | 5.2 |
| Off-chain archive size | 5.72 MB | 5.72 MB | 5.72 MB |
| Training time with / without (s) | 116.2 / 110.9 | 114.8 / 120.5 | 116.8 / 112.7 |

- **Ledger cost.** About 1.4 ms per round, against a FedAvg round of about 5.8 s. That is roughly 4,000× smaller, and well below the run-to-run wall-clock noise.
- **Storage.** The ledger adds about 22 KB per run, roughly 0.27% of the 8.29 MB that plaintext FedAvg transmits.
- **The off-chain archive dominates.** It stores 126 model states, at about 5 ms per round and 5.7 MB per run. It is needed only for after-the-fact verification and could be replaced by artifacts that clients or the server already retain.
- **Wall-clock comparison.** The total training-time differences (−5.6 to +5.3 s) go in both directions. They are timing noise, not recording cost.
- **Figure:** `results/blockchain_integration/figures/blockchain_integration_overhead.png`.

### Tamper detection (on the three real ledgers; 200 seeded trials per scenario per seed = 600 per scenario)

| Scenario | Detected by `validate()` | Detected with anchor |
|---|---:|---:|
| Edit update hash / `num_samples` / aggregation hash / timestamp / round; remove participant; delete, swap or insert block; flip a byte in the saved file | 100% (each 600/600) | 100% |
| Edit an update hash and re-hash that block | 566/600 (94.3%); missed only when the tip block is edited | 100% |
| Full rewrite with recomputed hashes | 0% (by construction) | 100% |
| Truncate the newest blocks | 0% (by construction) | 100% |
| **Perturb one value in an off-chain client or global state** (by 10⁻⁶) | 600/600 (100%), via hash mismatch | 100% |

With the anchor, all 14 scenarios are detected in all 8,400 trials. Without it, consistent rewrites and truncation are undetectable. That is the Day 15 limitation of a single-node chain, and it is unchanged here.

## 3. Bug found and fixed during Day 16

On the real ledgers, the byte-flip trials produced a number such as `1e406`, which JSON parses as infinity. Canonical re-encoding then raised an exception inside `Ledger.validate()` instead of reporting the chain invalid. `validate()` now reports "content cannot be canonically encoded" as a problem. A regression test was added (`tests/test_blockchain.py`). This changes behaviour only on corrupted input. All Day 15 trials completed without triggering it, so the Day 15 results are unaffected, and the Day 15 output files are byte-identical.

## 4. Integration tests (`tests/test_blockchain_fl_integration.py`)

These run the real `run_experiment` pipeline end to end on a small synthetic split (3 clients, 2 rounds), once without and once with recording:

1. **Recorded hashes match off-chain artifacts.** Every client and global hash verifies, the selected checkpoint's hash equals its block's `aggregation_hash`, and the recorded sample counts equal the partition.
2. **Tampering is detected.** An edited ledger fails validation. A full rewrite passes internal validation but fails against the anchor. A 10⁻⁴ change to one archived client weight is pinpointed to exactly that client and round.
3. **FedAvg results are equivalent.** The selected model, round history and test predictions are bit-identical, and the test metrics are equal.
4. **No sensitive content is on-chain.** Blocks have exactly the ledger fields. There are no floats anywhere on-chain, no string over 256 characters, and every block is under 2 KB. Neither raw weight bytes nor formatted weight values appear in the file, and nor do the words `weight` or `segments`.

There is also a state-hash determinism and sensitivity test. The earlier combination guard rejected DP/HE with blockchain; the v2 lifecycle section below documents the single newly supported HE+DP v2 + CKKS combination and its dedicated tests. Other combinations remain unsupported.

## 5. Interpretation

Integrating the ledger with FedAvg is **behaviour-neutral and cheap**. Training is unchanged bit for bit. Every recorded client and global model hash verifies against the actual artifacts. The ledger costs about 0.03% of training time and about 22 KB per run.

The integrity it provides is the Day 15 integrity: it detects inconsistent changes to the record and to the off-chain models. Detecting a consistent rewrite or truncation of the whole ledger still depends on an independently held anchor.

**What this does not provide yet:**

- Clients are not authenticated (client IDs are self-declared, with no signatures).
- There is no consensus or replication.
- The server computes the hashes. A malicious server could record hashes of states other than the ones it actually aggregated, unless clients publish their own update hashes and verify the global hash, which is a natural next step.

No claim is made about privacy. The ledger reveals only participation, sample counts and hashes.

## 6. Limitations

1. Single-node ledger with no consensus, replication or signatures, as on Day 15.
2. The server-side recorder trusts the server to hash what it aggregates. Client-side commitments are not implemented.
3. Plain FedAvg only. Recording DP or HE rounds is not supported in this version.
4. The off-chain archive is local and untracked. Verification assumes it is available.
5. Timing comes from one machine without load control.
6. Inherited limitations: a simulation on public data and 3 seeds.

## 7. Artifacts

- **Tables:** `results/blockchain_integration/tables/`
  - `blockchain_integration_per_seed.csv` (equivalence and overhead)
  - `blockchain_integration_verification.csv`
  - `blockchain_integration_tamper_detection.csv`
  - `blockchain_vs_fedavg_*` (seed-paired utility comparison)
  - `robustness_*`
- **Ledgers and anchors:** `results/blockchain_integration/ledgers/` and `anchors/`, per seed (tracked)
- **Per-run outputs:** `results/blockchain_integration/{metrics,tables,figures}/federated/`, including `*_blockchain_round_timings.csv` and `*_blockchain_summary.json`
- **Not tracked (regenerable):** the off-chain archive, checkpoints, and `.npz` predictions

## 8. Lifecycle-gated experiment v2

`configs/blockchain/fedavg_lifecycle_v2.json` adds a versioned gate to the existing ledger/recorder. In each plain-FedAvg round, the runner hashes the selected client states and appends a `prepared` phase block before invoking the aggregation callback. The recorder verifies the ledger chain, fixed participant roster, sample-count entries, and each update hash against the payload it is about to aggregate. Altered, missing, duplicate, out-of-roster, or stale-base submissions raise an error and prevent aggregation. After FedAvg completes, a `complete` phase block records the global-state hash; the chain and resulting hash are checked before the runner installs/broadcasts that model to the next round. The prepared and completed blocks share a round number and are validated as an ordered pair. Existing one-block-per-round v1 ledgers remain valid.

Tests exercise a successful callback, changed-payload rejection before the callback, missing-roster rejection, completed-round integrity, tamper detection, and an end-to-end plain FedAvg run through the new lifecycle gate. Per-phase hashing, verification, block, off-chain archive, final validation, and ledger byte counts are recorded so overhead can be compared with the no-ledger path. Model-state archives remain off-chain; the ledger contains participant identifiers, sample counts, state hashes, round metadata, and global-state hashes.

This is an active **single-process integrity gate** between the prepared update payloads and the aggregation/broadcast operations. It does not authenticate who created an update: the same coordinator creates the hashes and receives the update objects. It has no consensus, digital signatures, independent replicas, smart contracts, access control, or Byzantine resilience. A malicious coordinator can replace an update before preparing its commitment or rewrite the entire local chain; the external anchor can detect a rewritten/truncated history only when it is retained independently. If a verification error occurs, the current run aborts before aggregation or broadcast; restart from the last preserved checkpoint after correcting the cause. A prepared failure is held in memory and is not guaranteed to survive process termination. No recovery/rollback protocol across independent nodes is implemented.

The lifecycle verifier accepts a prepared block only as the valid current ledger tip during the pre-aggregation gate. Ordinary `validate()` and finalization reject an unfinished prepared tip; pre-aggregation validation opts in to that one pending-tip state. A prior prepared round must be immediately followed by its matching completion block, lifecycle rounds must advance contiguously, and a prepared commitment's previous-model hash must equal the prior block's global-model hash. These checks prevent stale prepared-block replay and out-of-order completion while leaving legacy one-block plain-FedAvg ledgers valid.

The gate demonstrates one measurable role for a blockchain-style ledger: a commitment check controls whether FL aggregation proceeds, and the completed record controls whether a global model is broadcast. It does not by itself resolve the supervisor's open interpretation of what “blockchain enables the experiment or algorithm” must mean. Requirement 4 remains partially evidenced pending supervisor confirmation; no consensus-based distributed-chain claim is made.

## 6. HE+DP v2 lifecycle integration

`configs/blockchain/he_dp_fedavg_v2_lifecycle.json` is a separate configuration, normally writing under `results/he_dp_v2_blockchain_lifecycle/`. It reuses the fixed-roster `he_dp_v2_aggregate_noise` + CKKS protocol and accountant from `configs/he_dp_fedavg_v2.json`; it does not alter that experiment's configuration or outputs. The combined runner is supported only for this protocol, CKKS, aggregate BatchNorm, and `lifecycle_gate_v2`. Other DP/HE/blockchain combinations remain rejected.

Before aggregation, the recorder commits the fixed roster, each client update hash, each configured public weight, the prior global-model hash, round number, and protocol metadata. The ledger schema for this mode contains no `num_samples`; no client update tensors or ciphertexts are stored in the off-chain archive. Only the initial and resulting global states are archived off-chain. The gate revalidates the chain, round sequence, previous model hash, roster, weights, and update hashes immediately before calling the existing `aggregate_he_dp_v2`. A completed block commits the resulting noised global-model hash; chain validation and hash comparison must pass before validation or checkpoint selection. The selected checkpoint hash is checked against its completed ledger record before the checkpoint is written.

Blockchain timings and serialized ledger size are emitted to `*_blockchain_round_timings.csv` and `*_blockchain_summary.json`. HE encryption, CKKS aggregation, trusted decryption, and aggregate-noise generation timings remain in the separate `*_he_dp_v2_costs.csv` and sensitive diagnostics outputs. Blockchain timing is not folded into HE+DP operation timings.

The ledger remains a single-node, hash-chained, tamper-evident log. It has no consensus, replication, membership enforcement, or signatures; internal hashes alone cannot detect a full rewrite or truncation without an independently retained anchor. For the preserved 2026-10-16 run, the anchor is saved beside the ledger in the same result tree, so validation demonstrates consistency with that co-located anchor—not protection against an attacker able to rewrite both. `src.blockchain.anchor_tool` now provides an explicit, non-overwriting export and verification interface. Exporting an anchor does not establish independent custody: an operator must retain it through a separate trusted system or channel.

After a completed run, export its anchor to a new path outside the result tree, transfer it through the project's separately controlled custody process, and later verify the ledger against that retained copy:

```powershell
$env:PYTHONPATH = "$PWD\tests;$PWD"
.\.venv\Scripts\python.exe -m src.blockchain.anchor_tool export --ledger "results\he_dp_v2_blockchain_lifecycle_reproduction_20261016\ledgers\he_dp_fedavg_v2_blockchain_lifecycle.jsonl" --anchor "D:\independent-ledger-anchors\he_dp_fedavg_v2_blockchain_lifecycle.json"
.\.venv\Scripts\python.exe -m src.blockchain.anchor_tool verify --ledger "results\he_dp_v2_blockchain_lifecycle_reproduction_20261016\ledgers\he_dp_fedavg_v2_blockchain_lifecycle.jsonl" --anchor "D:\independent-ledger-anchors\he_dp_fedavg_v2_blockchain_lifecycle.json"
```

The CLI validates the chain before export, refuses to overwrite an existing anchor, and reports whether a supplied anchor matches the ledger ID, length, and tip hash. Independent custody and authenticity of the transfer remain operational responsibilities; this tool does not sign anchors or add consensus. CKKS confidentiality relies on the key assumptions. The trusted decryptor sees the clean aggregate and is excluded from the DP guarantee; process isolation is not enforced in this simulation. No stronger blockchain or privacy claim is implied.

To reproduce the combined experiment, first obtain the outstanding written approval for the privacy assumptions/release set and confirm the split and utility criteria. Use a fresh, empty output root; do not run the command below against the preserved output tree:

```powershell
$env:PYTHONPATH = "$PWD\tests;$PWD"
.\.venv\Scripts\python.exe -c "from src.federated.run_fedavg import run_experiment; run_experiment('configs/blockchain/he_dp_fedavg_v2_lifecycle.json', evaluate_test=True, output_root='results/he_dp_v2_blockchain_lifecycle_reproduction_20261016', overrides={'data': {'split_root': 'data/splits/mitbih_v1'}})"
```

The preserved 20-round run completed on 2026-10-16 using the runtime split-root override and the distinct `results/he_dp_v2_blockchain_lifecycle_recovered_20261016/` output root. That historical path already exists and must not be reused with `overwrite=True`. The inaccessible `mitbih_he_dp_v1` directory was not modified: the accessible `mitbih_v1` metadata SHA-256 matched its documented frozen digest, its split config matched the versioned config, and all saved split shards passed the repository validator. The default combined output root contained residue from the prior failed preflight, so the completed experiment neither overwrote nor reused it.

Observed lifecycle evidence: the resulting ledger has 41 blocks and is 48,553 bytes; it validates against its anchor, contains no client sample counts, and its completed global-state hashes match all 21 global-only off-chain archives. The selected round-0 checkpoint hash matches the genesis model hash. The 20 blockchain phase records are separate from the 20 HE+DP cost records; the blockchain summary records 3.43 ms final ledger validation, while per-phase timings are in the associated CSV. This demonstrates the lifecycle gate behavior in this simulation, but does not settle the supervisor's open interpretation of Requirement 4.
