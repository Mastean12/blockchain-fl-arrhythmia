# Day 15 — Blockchain component: hash-chained ledger of FL round metadata (isolated prototype)

- **Date:** 2026-10-05
- **Configuration:** [`configs/blockchain/ledger_v1.json`](../configs/blockchain/ledger_v1.json)
- **Code:** `src/blockchain/`
  - `ledger.py`: blocks, hashing, validation, persistence
  - `tamper.py`: tamper scenarios
  - `benchmark.py`: benchmarks
- **Outputs:** `results/blockchain/`
- **Tests:** `tests/test_blockchain.py`

> This is a standalone, single-node, append-only, SHA-256 hash-chained ledger for research experimentation. It is **not integrated with FL training**: no FL run is read or modified, and the benchmarks use synthetic, deterministic FL-shaped records. It is not combined with DP, HE, attacks, or the proposed algorithm.

## 1. What it is, and what it is not

- **What it is:** a tamper-evident log. Each block commits to its content and to the previous block's hash, so any inconsistent change is detected by re-validation.
- **What it is not:** a distributed blockchain. There is **no consensus protocol, no replication across nodes, no permissioned membership, and no digital signatures**. It does not provide Byzantine fault tolerance or non-repudiation.
- **Its central limitation:** a single party with write access can rewrite the whole chain and recompute every hash, or silently drop the newest blocks. Such changes are detectable only against an **independently stored anchor** (the expected length and tip hash; §4). The prototype provides and measures anchoring. Making the anchor trustworthy (replication across institutions, signatures, a permissioned network) is future work for the actual blockchain layer (RQ4).

## 2. Block structure

```text
index → previous_hash → timestamp → round → experiment_id → participants → aggregation_hash → metadata → block_hash
```

| Field | Content |
|---|---|
| `index` | Position in the chain (genesis = 0) |
| `previous_hash` | `block_hash` of the previous block. Genesis uses 64 zeros. |
| `timestamp` | UTC ISO-8601 with microseconds. The clock is injectable, so tests and the example ledger are deterministic. |
| `round` | FL communication round. Genesis is round 0 (the initial model), and rounds must strictly increase. |
| `experiment_id` | Fixed for the whole chain |
| `participants` | One record per participating client, `{client_id, num_samples, update_hash}`, sorted by `client_id` with no duplicates |
| `aggregation_hash` | SHA-256 of the aggregated (global) model for that round. In genesis, the initial model hash. |
| `metadata` | Scalar values only (string, number, bool), for example the aggregation method |
| `block_hash` | SHA-256 over the canonical JSON of all fields above |

- **Canonical JSON:** sorted keys, compact separators, ASCII escaping, no NaN or Infinity.
- **Pinned determinism:** a test pins the digests of the seed-42 genesis block and round-2 block, so any change to the format or hashing is caught.

**Nothing sensitive goes on-chain.** Only SHA-256 digests, client identifiers, sample counts, and small scalar metadata are stored. The schema **rejects** the following:

- extra participant fields (for example `weights`);
- list or nested metadata;
- strings over 256 characters;
- non-hex hashes;
- blocks over 16 KB.

So ECG data, model parameters, and plaintext or encrypted updates cannot be written through the API. Verification works by hashing the off-chain payload: `verify_update(round, client, bytes)` and `verify_aggregate(round, bytes)`.

## 3. Operations

- **Block creation:** `append_round` validates the schema, links to the tip, computes the SHA-256 block hash, and appends.
- **Chain validation:** `validate(anchor=None)` checks, for every block:
  - the schema;
  - that the index equals the position;
  - that the recomputed hash equals the stored hash;
  - the genesis form;
  - that `previous_hash` links to the previous block;
  - strictly increasing rounds;
  - non-decreasing timestamps;
  - a constant `experiment_id`.

  With an anchor, it also checks the length and tip hash. It returns `(ok, problems)`, listing every problem found.
- **Persistence:** JSON Lines, a header line followed by one canonical block per line. Writes are atomic (temporary file, then `os.replace`). `load()` never trusts the file; validation is a separate, explicit step.
- **Anchor:** `anchor()` returns `{ledger_id, length, tip_hash}`, to be stored independently of the ledger file.

## 4. Benchmark results

These are median local wall-clock times (`time.perf_counter`, 5 repeats) on one CPU machine, without load control. Each round block holds 5 client records, as in the 5-client FedAvg setup. The full output is `results/blockchain/metrics/blockchain_benchmark.json`.

### Block creation, validation, and storage

| Chain length | Create (total) | Create per block | Validate (total) | Validate per block | Save | Load | File size | Bytes per block |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1.25 ms | 62 µs | 0.87 ms | 43 µs | 1.7 ms | 0.26 ms | 20.8 KB | 1,042 |
| 100 | 6.9 ms | 69 µs | 4.5 ms | 45 µs | 3.2 ms | 0.85 ms | 106 KB | 1,060 |
| 1,000 | 65 ms | 65 µs | 45 ms | 45 µs | 22 ms | 9.5 ms | 1.07 MB | 1,065 |
| 10,000 | 0.67 s | 67 µs | 0.45 s | 45 µs | 0.19 s | 0.12 s | 10.7 MB | 1,068 |

- **Scaling:** creation and validation scale linearly; per-block cost is constant. Anchored validation costs the same as plain validation.
- **Per-run cost:** a 20-round FL run (genesis plus 20 blocks) produces a 21.9 KB ledger. That is about 0.26% of the 8.29 MB plaintext FedAvg traffic over the same rounds, and it takes about 1 ms to create.
- **Figure:** `results/blockchain/figures/ledger_scaling.png`.

### Hashing overhead

| Quantity | Time |
|---|---|
| SHA-256 of one canonical block (about 1.06 KB) | 0.93 µs |
| Canonical JSON encoding of one block | 16.5 µs |
| SHA-256 of a plaintext model update (41,428 B, the Day 10 payload size) | 19 µs (about 2.2 GB/s) |
| SHA-256 of an HE ciphertext update (994,171 B, the Day 14 payload size) | 0.44 ms (about 2.3 GB/s) |
| Hashing per round: 5 updates + 1 aggregate, plaintext size | 0.11 ms |
| Hashing per round: 5 updates + 1 aggregate, ciphertext size | 2.6 ms |

The block hash itself is negligible. Block creation time is dominated by schema checks and canonical encoding. Hashing an update grows linearly with payload size, but stays in milliseconds even at HE ciphertext size. The payloads were synthetic random bytes of these sizes. No HE code was used and no FL data was read.

### Tamper detection (200 seeded trials per scenario, 21-block chain)

| Scenario | Detected by `validate()` | Detected with anchor |
|---|---:|---:|
| Edit a client update hash | 200/200 (100%) | 100% |
| Edit an update hash **and re-hash that block** | 187/200 (**93.5%**) | 100% |
| Edit `num_samples` | 100% | 100% |
| Remove a participant record | 100% | 100% |
| Edit an aggregation hash (any block, including genesis) | 100% | 100% |
| Edit a timestamp | 100% | 100% |
| Edit a round number | 100% | 100% |
| Delete a block | 100% | 100% |
| Swap two adjacent blocks | 100% | 100% |
| Insert a forged, correctly hashed block | 100% | 100% |
| **Full rewrite** (edit, then recompute all later hashes and links) | **0/200 (0%)** | 100% |
| **Truncate the newest blocks** | **0/200 (0%)** | 100% |
| Flip one hex or digit character in the saved file | 100% | 100% |

Reading the table:

- **Edit and re-hash one block.** This is caught by the next block's `previous_hash` link, except when the edited block is the **tip**, which has no successor. That happened in 13 of 200 trials (the expected rate is 1 in 20 positions), so the internal rate is 93.5%. A test verifies the middle-versus-tip behaviour explicitly.
- **Full rewrite and truncation.** These produce internally valid chains **by construction**. They are the inherent limit of a single-node hash chain. Only an external anchor detects them, and with one, all 13 scenarios are detected in all 2,600 trials.

## 5. Interpretation

The prototype provides the requested building blocks:

- immutable-by-detection recording of round metadata;
- per-client participation records;
- hash-based verification of updates and aggregates;
- round numbers and timestamps;
- aggregation hashes;
- tamper detection.

Its overhead is negligible at FL scale: microseconds per block, about 1 KB per round, and milliseconds of hashing per round even for ciphertext-sized updates.

The security value depends entirely on the **anchor**. Within one file, the chain detects inconsistent edits but not a consistent rewrite or truncation by whoever controls the file. Trustworthy coordination needs the tip hash to be held independently, for example replicated across participating institutions, signed, or committed on a permissioned network. That is the design question for integrating the ledger with FL. No claim is made that this prototype provides consensus, Byzantine tolerance, or non-repudiation.

## 6. Limitations

1. Single node: no consensus, replication, membership control, or signatures. Clients are not authenticated, so the `client_id` values are self-declared.
2. Anchors are produced but not protected. Where to store them and who trusts them is unresolved.
3. Timestamps come from the writer's clock and are only checked for monotonicity. There is no trusted time source.
4. Synthetic benchmark records. Timing comes from one machine without load control, and sizes depend on the JSON Lines encoding.
5. Not integrated with FL: no on-chain recording of the Day 10–14 runs and no smart-contract or coordination logic yet.

## 7. Artifacts

- **Tables:** `results/blockchain/tables/`
  - `ledger_scaling.csv`
  - `hashing_overhead.csv`
  - `tamper_detection.csv`
- **Metrics:** `results/blockchain/metrics/blockchain_benchmark.json` (full configuration and all results)
- **Example ledger:** `results/blockchain/ledgers/example_20_rounds.jsonl` (genesis + 20 synthetic rounds, deterministic clock, 21,909 B). Its anchor is in `example_20_rounds_anchor.json`; the tip hash is `e837cccb…41fe`.
- **Figure:** `results/blockchain/figures/ledger_scaling.png`
