# Final limitations (research freeze, Day 20)

This document consolidates limitations recorded on Days 2–20. Its numbered list is the historical Day 20 record; subsequent extension evidence and status changes are appended under “Post-freeze extension status” below. The conclusions in [`final_results.md`](final_results.md) must be read with all of them.

## A. Data, labels and evaluation

1. **Single public dataset.** MIT-BIH v1.0.0 only: 48 records from 47 subjects, recorded decades ago, with 25 of 48 records selected for uncommon arrhythmias. Class frequencies are not population prevalence, and there is no external or multi-institution validation.
2. **Subject separation is inferred.** The 47 subject-equivalence groups rest on PhysioNet's subject count and the documented 201/202 pair. Explicit patient IDs are unavailable, so this is not verified patient-level separation.
3. **Provisional taxonomy.** The 15 beat symbols are identity-mapped, rhythm and event annotations are out of scope, and no literature-justified task taxonomy has been adopted. The Day 8 taxonomy questions remain open.
4. **Weak test support.** The test set is 7 records with only **8 of 15 classes** positively supported. `S`, `e` and `E` occur in 1, 1 and 2 groups respectively and cannot appear in every split. Several per-class metrics are undefined or rest on fewer than 15 samples.
5. **Severe class imbalance and a weak baseline.** `N` is 68.5% of segments and 65.5% of the test set. The centralized baseline reaches macro-F1 0.1399, and `A`, `F`, `L`, `Q` and `a` have zero F1. Accuracy alone overstates performance.
6. **No repeated splits.** There is one fixed split and no cross-validation, and no confidence intervals are computed.
7. **Lead heterogeneity.** The second channel varies across records (V1, V2, V4, V5), but the model treats channels by position only.
8. **Preprocessing not sensitivity-tested.** The filter band, window length and normalisation were fixed a priori. Whole-record zero-phase filtering is non-causal.

## B. Models and training

9. **Overfitting in the centralized run.** The checkpoint comes from epoch 1, and validation loss worsened afterwards. There was no class weighting, augmentation or hyperparameter search.
10. **Unequal training budgets.** The centralized and federated budgets are not matched. Centralized ran 1 epoch at the selected checkpoint with continuous Adam state; FedAvg's selected round varied from 2 to 20 and reset the optimizer each round.
11. **Selection by validation loss can mask failure.** The minimum-validation-loss rule picked untrained models (DP: round 0) and pre-breakdown rounds (sign flip). It also diverges from validation macro-F1.
12. **Few seeds.** Three seeds per condition (plus three fresh seeds on Day 20) give descriptive variability only. No statistical significance is claimed anywhere.
13. **The centralized baseline has one seed.** Its variability is unknown.

## C. Federated learning

14. **Simulation only.** Five clients are partitions of one database on one machine. There is no real network, heterogeneous hardware, client dropout or partial participation (q = 1 throughout).
15. **The N/V collapse is unexplained.** It is reproducible across seeds, partitions, FedBN and HE (Days 10–14). `/` and `L` sit in only 2 training records each, so no whole-group partition can spread them over more than 2 of 5 clients. The cause is untested, and candidates such as optimizer state, client drift, budget and selection criterion were not ablated.
16. **No local-only training baseline.** RQ2's "local training" comparison was never run.
17. **Narrow FedBN result.** With 47–74 local batches per round, BatchNorm running statistics are re-estimated locally in both arms, so effectively only BatchNorm affine locality was tested.

## D. Privacy and confidentiality

18. **One DP configuration.** Client-level DP at ε = 7.98 and δ = 10⁻⁵ with a trusted server. There was no ε sweep, no record-level or local DP, and no DP-compatible normalisation. The accountant covers full participation only, and public sample counts are assumed.
19. **DP destroyed utility.** At this privacy unit, with 5 clients, every DP run selected the untrained round 0. DP-trained models were never test-evaluated, only assessed on validation.
20. **Gaps in the DP guarantee as implemented.** Seeded PRNG noise is not cryptographically secure. Floating-point effects are not modelled. The logged unclipped-norm diagnostics fall outside the guarantee.
21. **HE protects individual updates from an honest-but-curious server only.** The aggregate and the global model are visible to all key holders. There is one shared secret key, no threshold or multi-key HE, no key distribution, and no protection against malicious parties.
22. **HE runs are not bit-reproducible.** SEAL's encryption randomness is unseeded. Paired trajectories drift (up to 0.15 at seed 2024) through training sensitivity.
23. **DP, HE, the defense and the blockchain were never combined.** Robust screening conflicts with HE confidentiality, because the server needs similarities, and it interacts with DP noise.

## E. Integrity and auditability (blockchain)

24. **Not a blockchain network.** It is a single-node hash chain with no consensus, replication, permissioned membership, digital signatures or Byzantine tolerance.
25. **Integrity depends on an external anchor.** A consistent full rewrite or truncation is undetectable without an independently held anchor, and the anchor itself is not protected.
26. **The server computes the hashes.** A malicious server could record hashes of states other than those it aggregated. Client-side signed commitments are not implemented, and client IDs are self-declared.
27. **Timestamps are writer-clock only,** with no trusted time source.
28. **Only plain FedAvg rounds were recorded.** DP, HE and defended rounds were not recorded on-chain.

## F. Robustness and the proposed defense

29. **Narrow threat model.** One non-adaptive, non-colluding attacker; three attack settings (sign flip, ×10, norm-matched noise); true sample counts. No data, label or backdoor poisoning, no adaptive attacker tuned to the thresholds, and no multiple attackers.
30. **Attacker share is confounded with seed.** It ranges from 15.4% to 21.9% (Days 17–19) and is drawn afresh for Day 20.
31. **Single-seed calibration.** Thresholds come from one clean calibration seed (7). Honest cosines (minimum 0.20) sat close to τ_cos = 0.18, so more heterogeneous clients could produce false positives.
32. **Self-inclusive median reference.** It biases a noise attacker's cosine upwards, which explains the 5–20% noise detection on Day 18. It is unfixed.
33. **Variant 5 was chosen after seeing results.** It was selected from the Day 19 ablation on the evaluation seeds. Day 20 confirms it on fresh seeds of the **same** dataset and split only.
34. **Clean-utility cost not fully isolated.** Clipping costs a little clean utility (Day 19). The six requested ablation variants did not isolate the BatchNorm median without filters.
35. **Hard rejection drops data.** Excluding one client per round removes 15–22% of the data, which could harm minority classes held by few clients.

## G. Measurement

36. **Timing is uncontrolled.** Timings come from one CPU machine without load control. Communication sizes are analytical (FedAvg, DP) or serialised sizes (HE, ledger), not network measurements.

## H. Scope of claims

37. Nothing in this study demonstrates clinical usefulness, deployment readiness, diagnostic validity, generalization beyond MIT-BIH, or statistical significance. The "proposed" defense is a candidate built from established ideas. No novelty or superiority claim is made.
38. The literature review that AGENTS.md requires to justify design choices and novelty is not documented as complete; `docs/literature_matrix.md` is a template-level matrix.

## Post-freeze extension status (2026-10-15)

- The HE+DP extension has measured MIT-BIH metrics and cost results, but every seed selected the initial round-0 checkpoint. Its measured utility is poor and does not demonstrate a useful trained arrhythmia detector; see `docs/he_dp_integration.md` for results and measurements.
- The frozen MIT-BIH split files are missing from the current workspace (they were already reported deleted before this extension). Split validation and a real-dataset run cannot be completed without restoring the exact approved split artifacts or regenerating them under an explicitly approved procedure.
- The integrated local-noise protocol uses an entropy-seeded PyTorch generator and a private, Git-ignored seed artifact for repeatability. Formal Gaussian accounting assumes independent ideal Gaussian noise and secret seeds; finite-precision PRNG behavior and seed disclosure are not covered.
- HE key distribution is simulated with one shared client secret key in one process. There is no threshold decryption, transport security, or independent client process. Any key holder can decrypt ciphertexts it obtains.
- The original supervisor blockchain note was not found in the repository files inspected. Whether consensus, signatures, authentication, or replicated ledger state are mandatory remains unresolved. The existing single-node hash chain does not establish those properties, and this extension does not claim them.
- The HE+DP v2 + CKKS lifecycle-gate combination completed a 20-round run in `results/he_dp_v2_blockchain_lifecycle_recovered_20261016/` and passed the one-round synthetic integration test. Validation selected round 0; held-out test accuracy was 0.6554 and macro-F1 0.0528, so these results do not demonstrate useful trained-model utility. The configured split path remained inaccessible; the run used the readable split copy whose metadata SHA-256 matches the documented frozen digest and whose shards pass split validation. No ACL or split files were changed. The blockchain remains a single-node hash-chained log, and the supervisor's interpretation of Requirement 4 remains open; no consensus-based distributed-blockchain claim is made.
- SVDB was evaluated as a second, external dataset using a frozen MIT-BIH checkpoint. The inaccessible historical processed/split directories were not modified. A new raw-checksum-verified preprocessing and record-split recovery under `*_recovered_20261016` reproduced the 12 test records, test size, checkpoint hash, and all historical evaluation metrics. This is external evaluation only, not federated training on a second dataset; no patient IDs are available, so patient-disjoint or cross-dataset subject independence cannot be asserted.
- The combined run's CKKS clean-aggregate numerical error was measured empirically (maximum absolute error no greater than `8.016845931546612e-8` over its 20 rounds). This does not bound approximation error for all inputs and is not included in the idealized GDP/RDP calculation. Its saved run summary's `config.output_root` disagrees with the actual `model_path` and output directory; the historical summary remains unchanged, and the runner now records the effective root for future executions.
- The combined ledger's anchor is co-located with its ledger in the same result tree. The saved chain validates against it, but there is no evidence that the anchor was independently exported or held; a party able to rewrite both can defeat this check. The combined operation proves the recorded single-process gate behavior, not consensus, replication, authenticated membership, or Byzantine resilience.
- Lifecycle validation now rejects stale prepared-block replay, invalid phase transitions, out-of-order prepared rounds, malformed metadata, and a prepared block whose previous-model hash differs from its predecessor's global hash. Legacy one-block plain-FedAvg ledger behavior remains covered by regression tests. This is an implementation correction; it does not change historical ledgers or results.
- The proposed robust aggregator remains a candidate built from established components. The ablation supports attack-specific filter effects in the tested simulation but does not establish novelty or general superiority. Variant 5 was selected after examining the Day 19 ablation and its Day 20 confirmation reused the same dataset and split; the literature matrix is still a template, and the supervisor has not confirmed that this candidate fulfills Requirement 7.
- The accessible SVDB recovery outputs and recovered evaluation predictions are Git-ignored. Check ignore coverage before any future artifact-generation path changes; do not commit large or private reproducible artifacts.

The current seven-requirement evidence status and open decisions are summarized in [`final_submission_report.md`](final_submission_report.md) and [`requirement_evidence_matrix.md`](requirement_evidence_matrix.md).
