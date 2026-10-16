# AGENTS.md

## Project Identity

This repository contains the research implementation for the project:

**A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection**

This is an academic research project. The objective is not simply to build a software application, but to design, implement, experimentally evaluate, and document a scientifically defensible framework.

All development work must preserve the research purpose of the project.

---

# 1. Research Aim

The overall aim is:

> To develop and evaluate a blockchain-enabled federated learning framework for privacy-preserving ECG-based arrhythmia detection.

The project investigates how distributed ECG data can be used collaboratively for arrhythmia detection without requiring participating institutions to share raw ECG data.

The framework will investigate the combined roles of:

- Deep learning for ECG-based arrhythmia detection
- Federated learning for distributed model training
- Differential privacy and/or homomorphic encryption for privacy preservation
- Permissioned blockchain for trusted coordination, verification, integrity, and auditability
- A proposed aggregation and/or coordination algorithm
- Experimental evaluation under heterogeneous and potentially adversarial federated environments

---

# 2. Core Research Problem

Healthcare institutions may possess valuable ECG data but may be unable or unwilling to share raw patient data because of privacy, security, regulatory, ownership, and operational concerns.

Federated learning provides a mechanism for collaborative model training without directly exchanging raw ECG data.

However, federated learning alone does not necessarily guarantee strong privacy, trustworthy coordination, update integrity, or resistance to malicious participants.

This project therefore investigates a framework combining:

1. ECG-based arrhythmia detection
2. Federated learning
3. Privacy-preserving mechanisms
4. Blockchain-based coordination and verification
5. A proposed aggregation/coordination mechanism

The project must evaluate the benefits and costs of each component rather than assuming that combining technologies automatically improves the system.

---

# 3. Research Principles

The following principles are mandatory.

## 3.1 Research Before Implementation

Do not implement advanced components merely because they are technically interesting.

Major architectural and algorithmic decisions must be justified by:

- Literature
- Research objectives
- Research questions
- Experimental requirements
- Measurable system requirements

## 3.2 No Unsupported Novelty Claims

Do not describe an algorithm, architecture, or technique as "novel", "new", "first", "state-of-the-art", or "better" unless this has been established through a documented literature review and experimental evidence.

Potential research contributions must initially be described as:

- Proposed
- Candidate
- Investigated
- Experimental

until supported by evidence.

## 3.3 No Fabricated Research Results

Never invent:

- Accuracy
- Precision
- Recall
- F1 scores
- AUROC
- Privacy budgets
- Training times
- Communication costs
- Blockchain latency
- Encryption times
- Attack success rates
- Statistical significance
- Experimental conclusions

Results must always come from actual experiments.

## 3.4 Reproducibility

Experiments must be reproducible.

Where appropriate:

- Set random seeds
- Record configuration parameters
- Record dataset versions
- Record model versions
- Record software versions
- Save experiment configurations
- Save evaluation metrics
- Save generated figures and tables

---

# 4. Research Questions

The project should investigate the following broad research questions.

### RQ1 — Arrhythmia Detection

How accurately can ECG-based deep learning models detect and classify arrhythmias under centralized and federated learning settings?

### RQ2 — Federated Learning

How does federated learning perform compared with centralized and local training under heterogeneous/non-IID ECG data distributions?

### RQ3 — Privacy

How do differential privacy and homomorphic encryption affect model utility, privacy, computational cost, and communication overhead?

### RQ4 — Blockchain

How can permissioned blockchain improve the integrity, traceability, verification, and auditability of federated model-update coordination?

### RQ5 — Proposed Method

Can a proposed privacy-aware and/or trust-aware aggregation or coordination mechanism improve an identified limitation of existing approaches while maintaining acceptable model performance, privacy, and computational/communication overhead?

These research questions may be refined after the literature review, but changes must be documented.

---

# 5. Expected Research Objectives

The working objectives are:

1. Develop an ECG-based deep learning model for arrhythmia detection using publicly available ECG datasets.

2. Design a federated learning architecture that enables collaborative arrhythmia model training without exchanging raw ECG data.

3. Investigate differential privacy and homomorphic encryption as privacy-preserving mechanisms within the federated learning process.

4. Design a permissioned blockchain-based coordination and verification mechanism for federated model updates.

5. Develop and evaluate a proposed aggregation and/or coordination algorithm addressing an identified limitation in existing methods.

6. Compare the proposed framework against appropriate centralized, federated, privacy-preserving, and blockchain-enabled baselines.

7. Evaluate the framework in terms of predictive performance, privacy, computational cost, communication overhead, blockchain overhead, convergence, and robustness to relevant attacks.

---

# 6. High-Level System Architecture

The intended architecture is:

```text
                Distributed ECG Data
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   Client/Hospital  Client/Hospital  Client/Hospital
        │               │                │
   Local ECG        Local ECG        Local ECG
        │               │                │
   Local Model      Local Model      Local Model
        │               │                │
        └───────────────┼────────────────┘
                        │
                Privacy Mechanism
                 DP and/or HE
                        │
                        ▼
             Blockchain Coordination
                        │
                        ▼
              Federated Aggregator
                        │
                Proposed Method
                        │
                        ▼
                  Global Model
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Client 1   Client 2   Client 3
---

# 7. Controlled Completion Requirements

This section governs work on the outstanding integrated privacy pipeline and supervisor-specified blockchain architecture. It supplements the research principles above; it does not authorize changes to historical experiments or unrelated project scope.

## 7.1 Authoritative Supervisor Requirements and Unresolved Specification

The researcher supplied seven supervisor requirements, recorded verbatim in Section 8.1. Treat those seven statements as authoritative acceptance criteria. Map each requirement to implementation evidence and an acceptance test before implementation.

The blockchain requirement says to demonstrate how blockchain enables the experiment or algorithm, but it does not specify the required blockchain capabilities or what “enables” means. Project documentation describes a single-node hash-chained ledger integrated as a recorder with plain FedAvg; it has no consensus, replication, permissioned membership, digital signatures, or client authentication. These capabilities remain open decisions, not supervisor requirements. Do not assume that the existing recorder satisfies the supervisor requirement, and do not add consensus, signatures, replication, smart contracts, or other major features unless separately approved. Record ambiguity as an open decision; never fill gaps with invented requirements.

## 7.2 Integrated Homomorphic Encryption and Differential Privacy

The project must contain one demonstrably integrated HE+DP federated-learning pipeline. Separate HE-only and DP-only experiments do not satisfy this requirement. Reuse existing HE, DP, and federated-learning modules where appropriate, while preserving their existing behavior and historical experiment records.

Before selecting or implementing a protocol, explain why it meets the research objective and threat model; implementation convenience alone is not justification. Document and validate the complete protocol, including:

- Per-round data flow and the exact point at which client updates are formed and clipped
- Which party generates DP noise, how the noise is generated, and whether it is added before or after encryption
- What is encrypted, what homomorphic operations are performed, who decrypts, and how the aggregate/global model is distributed
- Key ownership, key distribution assumptions, and which parties can observe individual updates, encrypted updates, aggregates, and released models
- The adversary capabilities and trust assumptions, including server, client, key-holder, and collusion assumptions as applicable
- Privacy unit and adjacency relation; clipping norm; noise parameters; epsilon and delta; composition/accounting method; and all accountant assumptions and limitations

Validate the privacy accountant against the **actual integrated protocol**, including its sensitivity, weighting, sampling, release, and composition assumptions. Do not reuse an accountant result from a different protocol without demonstrating that its assumptions still hold. State clearly that HE provides confidentiality for specified values/operations under its key assumptions, while DP provides a quantified privacy guarantee under its stated mechanism and assumptions; neither property implies the other.

Add tests and a reproducible experiment proving that HE and DP execute together in the same training run. Measure and report model utility, privacy accounting, numerical error, communication overhead, and computation time from actual runs. Do not claim completion if the integrated path is merely configured, mocked, or separately exercised.

## 7.3 Data, Security, and Historical-Artifact Boundaries

- Raw ECG data must remain local to each participating client. Do not transmit it to an aggregator, ledger, or other client.
- Keep raw ECG signals, model parameters, private keys, and sensitive payloads off-chain. Store only justified metadata and cryptographic hashes on-chain; document the exact schema and its privacy implications.
- Do not make unverified security, privacy, integrity, authentication, consensus, or Byzantine-resilience claims. Describe what was tested and under which assumptions.
- Preserve existing datasets, preprocessing, subject/group splits, baselines, frozen configurations, and historical results. Do not silently overwrite, relabel, reinterpret, or regenerate them as though they were produced by a new protocol. Any approved correction must be explicit, traceable, and documented.
- Use new versioned configurations and separate output directories for new experiments. Keep generated checkpoints, predictions, and other large reproducible artifacts out of Git unless specifically required.
- Do not push to GitHub or commit changes unless explicitly authorized.
- Prefer correctness, reproducibility, scientific validity, security, and maintainability over implementation speed.

## 7.4 Mandatory Implementation Controls

Before implementing either outstanding requirement, produce a concise, reviewable plan that identifies reusable code, required changes, dependencies, risks, and acceptance tests. Map every mandatory supervisor requirement to a specific implementation and test. Identify protocol incompatibilities and unresolved security assumptions before coding.

Implement in small, testable stages. Avoid unrelated refactoring, new attacks, additional datasets, or new defense mechanisms unless explicitly required. Never weaken tests or alter old results to make new experiments pass. If a requirement cannot be implemented safely or validated, report the blocker and evidence; do not simulate success or claim completion.

## 7.5 Acceptance and Completion

Do not declare the outstanding work complete until all of the following are evidenced:

1. The integrated HE+DP pipeline runs and its combined behavior is tested.
2. Every blockchain capability explicitly required by the authoritative supervisor specification is implemented and demonstrated. If that specification remains unavailable or ambiguous, report that blocker; do not claim the blockchain requirement is satisfied.
3. New experiments can be reproduced from documented, versioned configurations.
4. The full test suite, split validation, baseline regression checks, and relevant integrity checks pass without weakening tests or changing frozen baselines.
5. Results, protocol assumptions, threat model, limitations, and known failures are documented from actual evidence.
6. Historical results remain unchanged unless an explicitly approved correction is necessary and documented.

---

# 8. Confirmed Supervisor Requirements and Open Decisions

This section records the seven supervisor notes supplied by the researcher and the audit decisions still needed. It does not convert audit recommendations into supervisor mandates or imply that any open decision has been approved.

## 8.1 Confirmed Supervisor Requirements

The following requirements are reproduced exactly as supplied:

1. Prepare the project structure well.
2. Use at least two datasets; one dataset is insufficient for validation.
3. Conduct effective experiments with multiple configurations and comparisons.
4. Demonstrate how blockchain enables the experiment or algorithm.
5. Implement HE or DP+HE for privacy.
6. Provide results supporting model performance and privacy.
7. Propose a new algorithm.

Do not expand these statements into unprovided technical requirements. In particular, Requirement 4 does not itself require consensus, signatures, replication, or a named blockchain protocol. Requirement 7 does not by itself establish that a proposed candidate is novel.

## 8.2 HE+DP Preservation and Experiment Gate

Preserve the existing three-seed integrated HE+DP experiment, its configurations, outputs, and poor-utility result. All three seeds selected round 0; do not tune parameters, replace, reinterpret, or overwrite this result.

Do not start HE+DP parameter tuning or new HE+DP experiments until the researcher has approved the privacy unit, threat model, key ownership and decryption access, adjacency relation, privacy-accounting assumptions, and complete release set. Record the approved protocol and rationale in a versioned design/configuration before experiments. Do not state a privacy guarantee beyond what the approved mechanism, accountant, and evidence support.

## 8.3 Second Dataset Gate

Requirement 2 requires at least two datasets. Before implementing or evaluating on another dataset, obtain approval for its identity/version and a documented protocol stating whether it is used for training, external evaluation, or both, including preprocessing, split and leakage controls, and comparable outcome measures. Do not select a dataset or imply approval on the researcher's behalf.

## 8.4 Blockchain Scope

Requirement 4 requires evidence that blockchain enables the experiment or algorithm. Before implementation, obtain clarification of the intended enabling role and measurable evidence of it. Treat consensus, signatures, replication, membership controls, smart contracts, and specific protocols as open—not mandatory—unless separately approved. Do not claim the current audit recorder meets Requirement 4 solely because it records hashes during FedAvg.

## 8.5 Proposed Algorithm and Research Integrity

Describe the existing method as a proposed candidate unless and until appropriate literature review and comparative evidence support a novelty claim. Evaluate it with documented configurations and relevant baselines; do not claim novelty, superiority, or performance without evidence.

Preserve historical datasets, preprocessing, splits, baselines, configurations, results, and integrity records. Use new versioned configurations and separate output directories for approved new experiments. Never tune on test data. Make no unsupported performance, privacy, security, or novelty claims. Do not commit or push unless explicitly authorized.

## 8.6 Open Decisions

Record unresolved items as **OPEN DECISION** in plans and reports until approval and evidence are available. Do not treat an unanswered item as approval.

| Open decision | Responsible approver | Evidence needed to close |
|---|---|---|
| HE+DP privacy unit, threat model, key ownership/decryption access, adjacency, accountant assumptions, and release set | Researcher; supervisor approval if required by the project | Written protocol decision covering each item and its rationale; accountant assumptions mapped to the implemented mechanism and outputs |
| Acceptable model-utility criteria and experiment comparisons | Researcher; supervisor approval if required | Predeclared metrics, thresholds or interpretation criteria, configurations, seeds, and comparison plan |
| Second dataset and its use | Researcher; supervisor approval if required | Approved dataset/version and documented training or external-evaluation protocol, including preprocessing and split controls |
| What blockchain must do to “enable the experiment or algorithm” | Supervisor for interpretation of Requirement 4; researcher for project implementation scope | Written clarification and measurable acceptance evidence; no specific consensus/signature/replication design is presumed |
| Whether the proposed candidate is sufficient for Requirement 7 and what supports a novelty claim | Researcher; supervisor approval if required | Written scope decision; relevant literature review and appropriate comparative evidence before any novelty claim |

The supplied notes do not name individual decision owners or provide numerical utility thresholds, a second-dataset choice, a privacy protocol, or a blockchain capability specification. Record those details as unresolved until the responsible people provide them.
