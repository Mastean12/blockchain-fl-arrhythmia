# Literature review matrix

## Initial source-verified scoping review (2026-10-16)

This is a targeted narrative review of foundational work and one directly related ECG/federated-learning article, not a systematic review or a novelty search. Bibliographic facts and source abstracts below were checked against publisher proceedings, the IACR ePrint, Crossref, and official PhysioNet pages on 2026-10-16. The review does not reproduce all experimental tables from every full paper. Claims about the papers are limited to their verified abstracts/titles; the project still needs a broader ECG-FL, privacy, blockchain, and clinical-validation search before any novelty claim or method selection.

| Work and verified source | Method/reported scope | Relevance and appraisal for this project | Evidence still needed |
|---|---|---|---|
| McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS/PMLR 54, 1273–1282 (2017), [publisher page](https://proceedings.mlr.press/v54/mcmahan17a.html). | Introduces iterative model averaging/Federated Averaging and studies decentralized, non-IID data and communication. | Foundational baseline; validates the need to compare FL with centralized/local training, not an ECG-specific result. | ECG-specific partitioning, label skew, client-size weighting, and clinical data-leakage controls. |
| Blanchard et al., “Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent,” NeurIPS (2017), [publisher page](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html). | Proposes Krum and formalizes resilience requirements under a bounded Byzantine-worker model. | A materially different robust-aggregation family; current experiments do not compare to Krum. The paper's assumptions are not evidence that this project’s fixed cosine/norm rule is Byzantine resilient. | Compare only after documenting compatibility with the model, client count, non-IID data, and attacker assumptions. |
| Yin et al., “Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates,” ICML/PMLR 80, 5650–5659 (2018), [publisher page](https://proceedings.mlr.press/v80/yin18a.html). | Analyzes coordinate-wise median and trimmed-mean robust distributed learning under stated assumptions. | Coordinate-wise medians are established prior art; this directly rules out novelty language for the candidate’s median component. Statistical guarantees do not transfer automatically to deep non-convex ECG FL or arbitrary client heterogeneity. | Review theorem assumptions and test matched median/trimmed-mean baselines before a performance claim. |
| El-Mhamdi, Guerraoui, and Rouault, “The Hidden Vulnerability of Distributed Learning in Byzantium,” ICML/PMLR 80, 3521–3530 (2018), [publisher page](https://proceedings.mlr.press/v80/mhamdi18a.html). | Shows that convergence of Byzantine-robust methods may coexist with ineffective high-dimensional models and proposes Bulyan. | Supports measuring utility and attack outcomes, not just convergence/detection. Current project evidence does not evaluate Bulyan or prove the candidate resistant to adaptive/high-dimensional attacks. | Include a comparable robust baseline and attack objective before drawing a robustness conclusion. |
| Asif et al., “Detecting Electrocardiogram Arrhythmia Empowered With Weighted Federated Learning,” IEEE Access 12, 1909–1926 (2024), DOI [10.1109/ACCESS.2023.3347610](https://doi.org/10.1109/ACCESS.2023.3347610). | The bibliographic record confirms a directly relevant ECG-arrhythmia/weighted-FL paper. | A relevant primary-study candidate for the ECG-FL comparison. Its complete method, dataset, split, and results were not independently appraised here; no performance claim is borrowed from it. | Obtain and review the full article, reconcile dataset/preprocessing and split provenance, and extract its baselines and metrics before comparative claims. |
| Cheon et al., “Homomorphic Encryption for Arithmetic of Approximate Numbers,” ASIACRYPT 2017, LNCS 10624, DOI [10.1007/978-3-319-70694-8_15](https://eprint.iacr.org/2016/421). | Describes approximate arithmetic over encrypted values and precision loss in CKKS-style computation. | Supports treating CKKS output as approximate and measuring numerical error. It does not provide DP or validate this project’s particular parameters, implementation, or threat model. | Review scheme security/parameter guidance and quantify implementation error under the exact released protocol; do not treat empirical max error as a worst-case bound. |
| Dong, Roth, and Su, “Gaussian Differential Privacy,” arXiv:1905.02383 (2019), [arXiv record](https://arxiv.org/abs/1905.02383); Balle and Wang, “Improving the Gaussian Mechanism for Differential Privacy,” ICML/PMLR 80, 394–403 (2018), [publisher page](https://proceedings.mlr.press/v80/balle18a.html). | Provide the GDP composition framework and analytical Gaussian calibration/conversion used by the project’s accountant. | These results justify the mathematical accounting only when the implemented mechanism, sensitivity, release transcript, and ideal Gaussian assumptions match. They do not account for the trusted decryptor’s clean view, seeded-PRNG deviations, or unbounded/uncertified CKKS approximation error. | Obtain approval of privacy unit, adjacency, threat model, release set, and accountant assumptions; validate the accountant against an approved end-to-end mechanism before new experiments. |
| Moody and Mark, “MIT-BIH Arrhythmia Database,” PhysioNet v1.0.0 (2005), DOI [10.13026/C2F305](https://physionet.org/content/mitdb/1.0.0/); PhysioNet, “MIT-BIH Supraventricular Arrhythmia Database,” v1.0.0, [dataset page](https://physionet.org/content/svdb/1.0.0/). | Dataset records describe the underlying ECG resources and their released versions. | Supports dataset identity/version and citation. It does not establish the project’s split validity; that is checked separately through saved metadata, group/record overlap, shard hashes, and evaluation artifacts. The recovered SVDB use is external evaluation, not FL training. | Preserve source-release checksums, preprocessing/config version, record membership, and exact outcome measures; patient-level cross-dataset independence remains unverified. |

### Synthesis and decision boundary

1. The literature supports FedAvg as a baseline and establishes multiple robust-aggregation families. The proposed coordinate-median/cosine/norm screening combination is not shown to be novel by this review; novelty and superiority remain unclaimed.
2. Existing attacks and synthetic edge cases do not establish robust operation under natural institutional non-IID shifts. The constructed equal-norm orthogonal update is flagged, demonstrating a plausible false-positive mechanism; no empirical ECG-site false-positive rate is available.
3. HE and DP address distinct properties. CKKS approximate arithmetic does not supply differential privacy, while a Gaussian accountant does not supply ciphertext confidentiality, protect a trusted decryptor, or validate implementation randomness/approximation by itself.
4. No blockchain protocol or healthcare ledger literature was reviewed sufficiently to select a distributed blockchain design. The supervisor's seven requirements do not specify consensus, signatures, or replication; Requirement 4's intended enabling role remains an **OPEN DECISION**.

### Review completion checklist

- [x] Verify and record foundational FedAvg, robust-aggregation, CKKS, GDP/Gaussian-mechanism, and official dataset sources.
- [x] Separate source claims from project interpretations and identify assumptions that do not transfer automatically.
- [ ] Read and appraise the complete ECG weighted-FL paper and additional ECG-specific FL/DP/HE studies.
- [ ] Search systematically for directly comparable robust aggregators, non-IID false-positive analysis, and healthcare-specific threat models.
- [ ] Review distributed-ledger literature only after the supervisor clarifies the required enabling role.
- [ ] Record search databases, exact search strings/date ranges, inclusion/exclusion rules, and screening decisions before describing the review as systematic or making novelty claims.

Use one row per paper or other primary research source. Record claims as reported by the source and distinguish them from our own interpretation. Do not infer unreported experimental details.

| Field | What to record |
|---|---|
| Reference | Full citation, DOI, and stable URL |
| Publication details | Year, venue, publication type, and peer-review status |
| Research aim | Problem and research questions addressed |
| Research gap | Limitation or gap identified by the authors |
| Method | Model, learning, privacy, security, or coordination methods |
| Dataset | Dataset name/version, population, ECG leads, and access conditions |
| Preprocessing | Filtering, segmentation, labeling, normalization, and exclusions |
| Evaluation design | Train/validation/test split, cross-validation, leakage controls, and seeds |
| Federated setup | Client count, partitioning, IID/non-IID assumptions, rounds, and participation |
| Privacy/security | Threat model, adversary assumptions, DP parameters or encryption scheme |
| Blockchain | Platform/consensus, on-chain/off-chain data, and role in coordination |
| Baselines | Methods used for comparison |
| Metrics | Predictive, privacy, computational, communication, blockchain, and robustness measures |
| Main findings | Results as reported, with table/figure/page references |
| Limitations | Limitations acknowledged by authors and issues noted in our appraisal |
| Reproducibility | Code/data availability, configuration, hardware, and missing details |
| Relevance | Connection to project objectives and research questions |
| Appraisal notes | Strengths, validity risks, assumptions, and open questions |

## Empty-entry template

Copy this template for each source:

- **Reference:**
- **Publication details:**
- **Research aim:**
- **Research gap:**
- **Method:**
- **Dataset:**
- **Preprocessing:**
- **Evaluation design:**
- **Federated setup:**
- **Privacy/security:**
- **Blockchain:**
- **Baselines:**
- **Metrics:**
- **Main findings:**
- **Limitations:**
- **Reproducibility:**
- **Relevance:**
- **Appraisal notes:**
