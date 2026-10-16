# Final submission readiness report

**Evidence cutoff:** 2026-10-16

**Scope:** repository and saved artifacts present in the local worktree; no commit, push, historical-result overwrite, ACL change, or new full HE+DP training run was performed for this report.

This report distinguishes the seven supervisor statements from implementation choices and open decisions. A passing integration test demonstrates execution on its test fixture; it does not establish deployment security, clinical utility, or a distributed blockchain.

## Supervisor requirements

| # | Requirement (verbatim) | Status | Evidence and remaining work |
|---|---|---|---|
| 1 | Prepare the project structure well. | **PARTIAL** | Code, versioned configs, tests, results, and research documentation are organized in project directories. The literature matrix contains an initial source-verified scoping review, not a systematic review; full-text ECG-FL appraisal and broader comparisons remain outstanding. Some historical/freeze documents required post-freeze qualifications, and no supervisor rubric for “well” was provided. |
| 2 | Use at least two datasets; one dataset is insufficient for validation. | **SATISFIED WITH LIMITATIONS** | MIT-BIH is the federated development dataset; SVDB v1.0.0 is a second, held-out external evaluation. The locked processed/split paths were not changed. Separate recovery outputs were generated from raw files checked against the SHA-256 manifest. Recovered split membership (12 test records / 27,244 segments), checkpoint hash, and every metric exactly match the historical external-evaluation summary. See [`svdb_integration.md`](svdb_integration.md#recovered-provenance-verification-2026-10-16) and the historical/recovered metrics at [`baseline_metrics.json`](../results/svdb_external_v1/metrics/baseline_metrics.json) and [`recovered baseline_metrics.json`](../results/svdb_external_v1_recovered_20261016/metrics/baseline_metrics.json). SVDB supplies no patient identifiers: this is record-disjoint, not demonstrated patient-disjoint or cross-dataset subject-independent evaluation. SVDB was not federated-training data. |
| 3 | Conduct effective experiments with multiple configurations and comparisons. | **PARTIAL** | Many frozen baseline, privacy, HE, attack, defense, ablation, and confirmation experiments exist. The combined HE+DP v2/blockchain result is one 20-round run and selects round 0; there is not a predeclared, common comparison meeting an approved utility criterion. Do not tune on the test set. The utility threshold and comparison plan remain open. |
| 4 | Demonstrate how blockchain enables the experiment or algorithm. | **PARTIAL / OPEN DECISION** | `lifecycle_gate_v2` verifies the current prepared tip, fixed roster, update commitments, previous global hash, and round sequence before aggregation; a callback cannot execute on a failed check. Completion commits the resulting noised model hash and validation occurs only after chain/hash validation. The preserved run has a 41-block ledger; its selected checkpoint (round 0/genesis) hash matches the ledger record. The saved anchor is co-located with the ledger; independent custody is not evidenced. The component is a **single-node, single-process hash-chained tamper-evident log**, with no consensus, replication, signatures, authenticated membership, or Byzantine resistance. Obtain supervisor clarification of what “enables” means before claiming this requirement is fully met. |
| 5 | Implement HE or DP+HE for privacy. | **SATISFIED WITH LIMITATIONS (simulation integration)** | The preserved 20-round combined run executes the existing `aggregate_he_dp_v2` path: fixed public weights, per-client delta clipping, CKKS encryption and weighted sum, trusted decryption, then one aggregate Gaussian perturbation. The ledger records client IDs, update hashes, public weights, protocol metadata, and model hashes; it records no actual client sample counts and the off-chain archive contains only global states. The recorded GDP result is `epsilon = 7.979543723560064`, `delta = 1e-5`, sensitivity `0.2`, aggregate noise standard deviation `0.538`, 20 full-participation rounds; the RDP cross-check is `9.359535227141418`. This is conditional on the fixed roster/weights, add/remove-one-client-slot adjacency, ideal independent Gaussian draws, and excluding the decryptor that sees the clean aggregate. A seeded PRNG is used; process isolation is not enforced; CKKS approximation and finite-precision effects are not covered by a proved bound in the accountant. Protocol approval and end-to-end privacy assurance remain open. |
| 6 | Provide results supporting model performance and privacy. | **PARTIAL** | Actual combined-run held-out metrics are accuracy `0.6553763`, macro-F1 `0.0527877`, weighted-F1 `0.5189371`, and macro AUROC `0.4244314` (8/15 classes supported). Validation loss was `2.6343226` at round 0 and larger at every trained round; the pre-existing minimum-validation-loss selection therefore selected the initial checkpoint. This result does not demonstrate useful trained-model utility. Privacy accounting and costs are recorded, but the guarantee is idealized/conditional as above. The utility criterion remains an **OPEN DECISION**. |
| 7 | Propose a new algorithm. | **PARTIAL / OPEN DECISION** | [`proposed_method.md`](proposed_method.md) describes a proposed candidate built from established clipping, similarity/norm screening, and robust-statistics components. Existing ablation and fresh-seed evidence is limited to one dataset/split and a small fixed attack setting; Variant 5 was selected after the Day 19 ablation, and Day 20 reused the same split. The literature matrix is incomplete. There is no supported novelty or general superiority claim, and the researcher/supervisor has not confirmed that this candidate fulfills the requirement. |

The detailed status matrix is maintained in [`requirement_evidence_matrix.md`](requirement_evidence_matrix.md). “Satisfied” for Requirement 2 describes the two-dataset/external-evaluation evidence only; it does not imply that every research question or framework objective is complete.

## Combined HE+DP v2 and lifecycle-gate evidence

The combined experiment completed on 2026-10-16, using the runtime override to the accessible MIT-BIH split copy and the separate output root [`results/he_dp_v2_blockchain_lifecycle_recovered_20261016/`](../results/he_dp_v2_blockchain_lifecycle_recovered_20261016/). Its existing HE+DP v2 configuration, three-seed historical results, and round-0 selection result were not replaced or tuned. The synthetic integration test uses one training round; the 20-round run is preserved evidence, not a test fixture.

The recorded protocol is:

1. Five fixed roster slots with public weights `[0.2, 0.2, 0.2, 0.2, 0.2]`; complete participation.
2. Each client's floating-point model delta is clipped to L2 norm `1.0`, then CKKS-encrypted using the public context.
3. The server performs the public-weight homomorphic sum without the secret key.
4. The trusted decryptor owns the secret key, decrypts the clean aggregate, and adds one Gaussian vector with standard deviation `0.538`.
5. The resulting noised global state becomes the next round state and is used for validation. The held-out test set is evaluated after validation-based selection.

The accounting models fixed-weight add/remove-one-client-slot adjacency, `C max(p_k) = 0.2` sensitivity, multiplier `2.69`, `T=20`, `q=1`, GDP composition and analytic Gaussian conversion at `delta=1e-5`; an RDP upper-bound cross-check is also recorded. The private diagnostics report a maximum absolute CKKS clean-aggregate error of at most `8.016845931546612e-8` and per-round RMS values from `3.0737774066477107e-9` to `3.170211994490261e-9`. These are empirical run measurements, not a worst-case error bound, and are not incorporated into the idealized accountant. DP does not encrypt; CKKS does not provide DP.

The trusted decryptor sees the clean aggregate and is excluded from the conditional DP statement. It is assumed honest and non-colluding; the simulation does not enforce process isolation, distributed key custody, or secure production noise generation. Raw ECG and private keys are not on-chain. The ledger contains no sample counts; no client state is in the off-chain archive. The chain contains 200 client-commitment entries (the 100 round/client update hashes appear once in each prepared/completed block pair). Because those update tensors are intentionally not archived, post-run payload replay is unavailable; the test suite verifies commitments before aggregation and rejects tampering at that boundary.

The historical config's `release_set` wording mentions validation-selected checkpoints but does not explicitly describe each intermediate noisy global model consumed each round. The actual accountant composes all 20 rounds, and that full adaptive transcript—not only the selected checkpoint—is the scope used here. The frozen config remains unchanged; clarify this release set in an approved versioned protocol before further HE+DP experiments.

The chain validates against its saved anchor, contains 41 blocks, and is 48,553 bytes. The selected round-0 checkpoint matches the genesis model hash. All 41 ledger-to-archive global-state hash checks match across the 21 global-only archive files. The anchor and ledger live in the same result tree, so this is internal consistency against a co-located anchor, not independently anchored adversarial protection. Recorded mean timings per round, in this one uncontrolled CPU run:

| Component | Mean | Measurement scope |
|---|---:|---|
| CKKS client encryption | 79.705 ms | All 5 clients |
| CKKS server weighted aggregation | 18.516 ms | Homomorphic sum |
| Trusted decryption | 3.348 ms | Aggregate decryption |
| Aggregate Gaussian noise | 0.265 ms | Noise generation/addition |
| Ledger prepare hashing | 2.913 ms | Client update hashes |
| Ledger prepare block append | 0.114 ms | Prepared block |
| Ledger post-append verification | 2.861 ms | Chain and commitments |
| Ledger completion verification/hash | 3.338 ms | Before completing block |
| Ledger completion block append | 1.753 ms | Completed block |
| Global-only archive save | 1.805 ms | Off-chain result state |

Final full-chain validation took 3.434 ms. These HE and ledger phase measurements are kept separate; they are single-machine observations, not network latencies or controlled performance estimates. The implementation remains a single-node hash chain.

The historical run summary has a known metadata inconsistency: `config.output_root` names `results/he_dp_v2_blockchain_lifecycle`, while the actual `model_path` and artifacts are under `results/he_dp_v2_blockchain_lifecycle_recovered_20261016`. The historical summary is preserved unchanged. The runner now records the effective output root for future runs, covered by the one-round integration test.

## Round-zero selection and utility diagnosis

The saved history supports the minimum-validation-loss selection; it does not indicate a selection-code defect. Round 0 validation loss is `2.6343226451`; round 1 is `294.1776845644`, and every later round is also worse than round 0. The round-1 validation accuracy is `0.00446` and macro-F1 is `0.00122`. The held-out test metrics come from the round-0 initial checkpoint, evaluated once after selection.

The saved first-round global state has a trainable-parameter displacement of `54.63575` L2 from initialization across 10,127 trainable values. The actual seeded Gaussian vector reconstructed only for this audit from the private local seed has trainable L2 `54.63932`; the full 10,351-coordinate noise vector has L2 `55.29228`. Under the configured `C=1`, maximum-weight `0.2`, and Gaussian standard deviation `0.538`, the idealized Gaussian noise is therefore the dominant first-round change; the clipped weighted client signal is bounded by `1.0` L2. The private seed and vector were not copied into public artifacts or this report. This quantitatively explains the collapse and round-0 selection; it does not justify tuning or replacing the preserved result. No evidence of a label-map or validation-selection implementation bug was found.

**Proposed utility gate for approval, not an approved protocol:** keep the existing validation-loss selection rule; use three predeclared matched seeds on the same approved split; require mean validation macro-F1 of the combined protocol to be within 0.01 absolute of matched FedAvg, with no seed more than 0.03 absolute below FedAvg. Report per-seed macro-F1 and supported-class recall; only evaluate test after selection is frozen, once per run, and never use test results for threshold or configuration tuning. These are proposed operational tolerances, not clinically established thresholds. A researcher/supervisor must approve or replace them before any utility sweep.

## Privacy, leakage, and threat-model review

| Concern | Observed evidence | Limitation / mitigation |
|---|---|---|
| Raw ECG leakage | No ECG arrays are written to the ledger or sent by the combined protocol; the synthetic one-round test split stays local to the test fixture. | This is a local simulation. No network capture or independently isolated client/server processes were used. |
| Client-update exposure | Combined ledger participants contain client IDs, SHA-256 update commitments, and fixed public weights; the test asserts no `num_samples`. The off-chain update archive files contain exactly `round` and `global_state`; they contain no `client_states`. | Update commitments cannot be payload-replayed after the run because client tensors are intentionally not archived. Commitments do not authenticate their submitters. |
| Noise/key exposure | The deterministic noise seed and clean-aggregate-dependent CKKS diagnostics are under `private/`; the integration test scans public JSON/JSONL/CSV outputs to ensure the exact seed is absent. Those paths are Git-ignored. | A seeded PRNG is reproducible, not a production CSPRNG. CKKS role separation is API-level in one process, not enforced isolation. The trusted decryptor sees the clean aggregate and is excluded from the DP claim. |
| Ledger/global-model exposure | Ledger reveals roster/participation, public weights, update hashes, metadata, and global hashes; the off-chain archive and selected checkpoint contain global model weights only. | Global models may still encode information. The log has no consensus, replication, authentication, or availability guarantee; the historical anchor is co-located. |

`CKKS` confidentiality under key assumptions is not differential privacy; `DP` accounting does not encrypt client updates or protect the trusted decryptor. The accountant does not incorporate a proven CKKS approximation-error bound or a cryptographic randomness guarantee. These limits are explicit in [`he_dp_v2_protocol.md`](he_dp_v2_protocol.md). The CLI [`anchor_tool.py`](../src/blockchain/anchor_tool.py) can export a new anchor and verify a ledger against it, but independent custody remains an operator-controlled step; the preserved 2026 run's anchor remains co-located.

## Proposed-defense edge cases and related work

Focused tests now reject non-finite model inputs/counts/configuration; a zero update is not rejected solely for an undefined cosine; an extreme finite trainable update is screened without producing non-finite output. A constructed same-norm orthogonal update is flagged by the candidate's cosine rule, demonstrating a false-positive path under a benign directional shift. A BatchNorm-buffer-only poisoning example is not screened when the robust-BatchNorm component is disabled and can produce a negative averaged variance. These are synthetic unit-test characterizations, not rates measured on institutional/non-IID ECG clients. Existing run evidence covers sign-flip, scaling, and norm-matched noise attacks in a limited single-partition setting; natural heterogeneity and adaptive attacks remain outstanding.

The candidate uses established ideas also represented by Krum, coordinate-wise median/trimmed mean, and Bulyan; it has not been compared directly to those robust-aggregation baselines. A targeted, source-verified initial review is in [`literature_matrix.md`](literature_matrix.md), including an ECG weighted-FL article identified by DOI. It is explicitly not a systematic review: full appraisal of ECG-specific FL/privacy work, robust baselines, and the blockchain enabling role remains incomplete. No novelty, general robustness, clinical-utility, or superiority claim is supported.

## Mitigation and execution plan

1. **Researcher/supervisor approval:** record the privacy unit, adversary/role capabilities, key ownership/decryption, adjacency, release set, accountant assumptions, utility criterion, dataset protocol, blockchain enabling role, and candidate scope in a versioned design. Unanswered items remain **OPEN DECISION**.
2. **Reproducibility preflight:** confirm exact dataset/split/config identities and hashes, validate group/record separation and shard provenance, confirm output paths do not exist, and verify private paths are ignored. Preserve all frozen configs/results.
3. **Comparison protocol:** after approval, preregister matched seeds/configs and appropriate centralized/local/FedAvg, HE, DP, integrated HE+DP and blockchain comparisons. Keep test data untouched until all validation selection is complete.
4. **Robustness review:** only after approval, choose realistic benign heterogeneity and attack assumptions; report false positives, per-class utility, and limitations alongside relevant robust baselines. The newly added tests are not substitutes for these experiments.
5. **Blockchain scope:** retain the current single-node ledger unless the supervisor explicitly requires a distributed capability. If the role is accepted, export the final anchor to a separately controlled store and verify it there; this does not add consensus.
6. **Regression/integrity:** run focused tests, the full project suite, split-provenance checks, baseline regression and artifact-hash comparisons. Do not weaken tests or overwrite historical results.

The exact command for a future combined run is recorded for reproducibility but is **NOT AUTHORIZED TO RUN until Step 1 is complete**; it uses a new output directory and never the preserved result root:

```powershell
$env:PYTHONPATH = "$PWD\tests;$PWD"
.\.venv\Scripts\python.exe -c "from src.federated.run_fedavg import run_experiment; run_experiment('configs/blockchain/he_dp_fedavg_v2_lifecycle.json', evaluate_test=True, output_root='results/he_dp_v2_blockchain_lifecycle_reproduction_20261016', overrides={'data': {'split_root': 'data/splits/mitbih_v1'}})"
```

## SVDB recovery evidence

No ACL or ownership settings were changed. The locked historical SVDB processed/split folders were left untouched. New outputs are at [`data/processed/svdb_v1_recovered_20261016/`](../data/processed/svdb_v1_recovered_20261016/) and [`data/splits/svdb_v1_recovered_20261016/`](../data/splits/svdb_v1_recovered_20261016/); these exact data roots and recovered prediction arrays are Git-ignored. Processing checked the source release manifest and generated 184,530 segments across 78 records. The recovered processed-manifest SHA-256 is `b6968cc886278f6f698e535174b97208271984d1a0360ecf36ea2774f27321cf`; split metadata SHA-256 is `8113dd48408cf485130dedc99c43b587b019326f17ef6a8ab74e4389210e6cb2`. The recovered evaluation metrics JSON SHA-256 is `d376c8658013791bb5af625b033278d44b6aa9a470f47d88ea8a588b1f17fcf1`. The recovered held-out test record IDs and every metric match the historical result. The test support is sparse (only five of 15 classes supported); no patient-level independence is claimed.

This audit rechecked each raw SVDB record triplet against the local checksum manifest and validated deterministic split membership plus the hashes of every recovered split shard against its processed source via `validate_svdb_splits`. The recovered split has 54/12/12 records and 130,544/26,742/27,244 segments for train/validation/test. The MIT-BIH source list has 48 records; its split covers all 48 (33/8/7) and validates with no record/group overlap or content/provenance duplicates. Exact read-only verification commands are in [`svdb_integration.md`](svdb_integration.md#current-worktree-provenance-recheck-2026-10-16).

Recovery and evaluation commands, from the repository root:

```powershell
.\.venv\Scripts\python.exe -c "from src.preprocessing.svdb import process_svdb, build_svdb_splits; raw='data/raw/svdb/mit-bih-supraventricular-arrhythmia-database-1.0.0'; process_svdb(raw, 'data/processed/svdb_v1_recovered_20261016'); build_svdb_splits('data/processed/svdb_v1_recovered_20261016', 'data/splits/svdb_v1_recovered_20261016')"
.\.venv\Scripts\python.exe -c "from src.models.evaluate_cnn import evaluate_checkpoint; evaluate_checkpoint('results/models/centralized_cnn_v1.pt', 'data/splits/svdb_v1_recovered_20261016/test', 'configs/preprocessing/mitbih_v1.json', 'results/svdb_external_v1_recovered_20261016', batch_size=512, device='cpu')"
```

Both commands refuse to overwrite existing outputs. The report's recovered metrics were produced by the second command; no test tuning or checkpoint retraining was done.

## Regression coverage and validation

Lifecycle coverage now includes successful plain and combined paths, roster omissions and duplicates, altered/stale updates, wrong prior-model commitments, tampered chains, malformed metadata, invalid/missing phase transitions, round sequencing, checkpoint/ledger hash equality, privacy-safe fixed-weight commitments, and a temporary-split one-round training integration. Legacy plain ledgers are validated separately.

Validation used the project virtual environment:

```powershell
$env:PYTHONPATH = "$PWD\tests;$PWD"
.\.venv\Scripts\python.exe -m pytest tests\test_blockchain.py tests\test_blockchain_lifecycle_v2.py tests\test_he_dp_v2.py tests\test_he_dp_integration.py tests\test_proposed_defense.py tests\test_svdb_integration.py -q
.\.venv\Scripts\python.exe -m pytest -q tests
```

**Results:** focused audit/privacy/blockchain/defense/split tests: **55 passed, 4 existing Matplotlib `set_bad` deprecation warnings** (3 configuration subtests). Full suite: **140 passed, 10 existing Matplotlib `set_bad` deprecation warnings**. `git diff --check` passed. The combined experiment's one-round synthetic integration test passed; no 20-round experiment was launched or rerun as part of this audit.

## Approval and Execution Gate

No signed or dated approval record was located in the repository. Evidence that an experiment already ran is not evidence that its protocol was approved. The detailed requirement-by-requirement checklist, exact verification commands, statuses, and owners are in [`requirement_evidence_matrix.md`](requirement_evidence_matrix.md).

| Open decision | Responsible approver | Evidence required |
|---|---|---|
| HE+DP protocol/trust approval | Researcher; supervisor if required | Written protocol decision covering privacy unit, adversary capabilities/collusion, key ownership and decryption access, adjacency, accountant assumptions, numerical-error boundary, and complete release set. |
| Utility criterion and comparisons | Researcher; supervisor if required | Approve or replace the explicitly proposed validation macro-F1 non-inferiority margins and matched comparison plan; preserve test data for final evaluation only. |
| Dataset identity/version and role | Researcher; supervisor if required | Written acceptance of SVDB v1.0.0 as external evaluation, its processing/split controls, and record-/patient-identity limitations. |
| Blockchain meaning | Supervisor for interpretation; researcher for scope | Written clarification of measurable “enables” acceptance. No consensus, signatures, replication, or other capability is presumed. |
| Candidate sufficiency and novelty | Researcher; supervisor if required | Scope decision; complete source review and appropriate comparisons before any novelty/superiority claim. |
| Independent anchor custody | Researcher / project operator | Retain the exported anchor through a separately controlled channel/store and verify it there. The preserved run is still co-located. |

Until approvals are recorded, do not start new HE+DP tuning or utility sweeps, select another dataset protocol, or make stronger blockchain/privacy/novelty claims. The future combined-run command above is documented only for reproducibility; it is not approval to execute. No commit or push was made.

## Files and worktree scope for this continuation

Files edited during the prior implementation continuation (this task adds the files listed below):

- `.gitignore`, `README.md`
- `docs/blockchain.md`, `docs/blockchain_fl_integration.md`, `docs/final_limitations.md`, `docs/final_methodology.md`, `docs/final_results.md`, `docs/final_submission_report.md`, `docs/he_dp_v2_protocol.md`, `docs/reproducibility.md`, `docs/requirement_evidence_matrix.md`, `docs/svdb_integration.md`
- `src/blockchain/fl_recorder.py`, `src/blockchain/ledger.py`, `src/federated/run_fedavg.py`
- `tests/test_blockchain.py`, `tests/test_blockchain_lifecycle_v2.py`

Files added or edited by this readiness continuation:

- `configs/blockchain/he_dp_fedavg_v2_lifecycle.json`
- `docs/final_submission_report.md`, `docs/requirement_evidence_matrix.md`, `docs/he_dp_v2_protocol.md`, `docs/blockchain_fl_integration.md`, `docs/proposed_method.md`, `docs/literature_matrix.md`, `docs/svdb_integration.md`
- `src/blockchain/anchor_tool.py`, `src/blockchain/fl_recorder.py`, `src/federated/privacy.py`, `src/federated/proposed_defense.py`, `src/federated/run_fedavg.py`, `src/preprocessing/svdb.py`
- `tests/test_blockchain.py`, `tests/test_blockchain_lifecycle_v2.py`, `tests/test_he_dp_v2.py`, `tests/test_proposed_defense.py`, `tests/test_svdb_integration.py`

New SVDB artifacts were written only to the separate `data/processed/svdb_v1_recovered_20261016/`, `data/splits/svdb_v1_recovered_20261016/`, and `results/svdb_external_v1_recovered_20261016/` roots. The raw release and historical result roots were read-only. The worktree already contained unrelated/pre-existing modifications and untracked files before this continuation; those were not discarded or represented as changes made here. Nothing was staged or committed.
