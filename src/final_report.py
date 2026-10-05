"""Day 20 consolidation: read every frozen result table (read-only) and write results/final/.

Produces per-dimension CSV tables (utility, privacy/confidentiality, integrity/auditability,
robustness, communication, computation), the Day 20 confirmation tables, a machine-readable
final_summary.json (including research-question status), an experiment manifest with
artifact hashes, and final figures. Values are copied from saved outputs, never recomputed
from models, and no previous result file is written.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.federated.final_validation import collect as collect_final, load_final_config

OUT = Path("results/final")
MET = ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "macro_auroc"]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(values):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return {"mean": float(v.mean()) if len(v) else None, "std": float(v.std(ddof=1)) if len(v) > 1 else None,
            "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None, "n": int(len(v))}


def _utility_row(day, experiment, arm, frame, cols, seeds, source, note=""):
    row = {"day": day, "experiment": experiment, "arm": arm, "n_runs": len(frame), "seeds": seeds, "source": source,
           "note": note}
    for metric, col in zip(MET, cols):
        s = _stats(frame[col]) if col in frame else _stats([])
        row[f"{metric}_mean"], row[f"{metric}_std"] = s["mean"], s["std"]
        row[f"{metric}_min"], row[f"{metric}_max"] = s["min"], s["max"]
    return row


def utility_table(final_runs):
    rows = []
    c = _json("results/metrics/baseline_metrics.json")["metrics"]
    rows.append({"day": 6, "experiment": "centralized_cnn_v1", "arm": "centralized", "n_runs": 1, "seeds": "42",
                 "source": "results/metrics/baseline_metrics.json", "note": "frozen centralized baseline",
                 **{f"{m}_mean": v for m, v in zip(MET, [c["accuracy"], c["macro_f1"], c["weighted_f1"],
                                                       c["macro_recall_all_15_labels_zero_for_no_support"],
                                                       c["macro_auroc_ovr_defined_classes"]])}})
    o = _json("results/metrics/federated/fedavg_v1_test_metrics.json")["metrics"]
    rows.append({"day": 10, "experiment": "fedavg_v1 (official)", "arm": "FedAvg", "n_runs": 1, "seeds": "42",
                 "source": "results/metrics/federated/fedavg_v1_test_metrics.json", "note": "official FedAvg run",
                 **{f"{m}_mean": v for m, v in zip(MET, [o["accuracy"], o["macro_f1"], o["weighted_f1"],
                                                       o["macro_recall_all_15_labels_zero_for_no_support"],
                                                       o["macro_auroc_ovr_defined_classes"]])}})
    std_cols = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_recall", "test_macro_auroc"]
    rob = pd.read_csv("results/robustness/tables/robustness_runs.csv")
    for part in ("natural", "controlled"):
        rows.append(_utility_row(11, "FedAvg robustness", f"{part} partition", rob[rob.partition == part], std_cols,
                                 "42,123,2024", "results/robustness/tables/robustness_runs.csv"))
    fedbn = pd.read_csv("results/diagnostics/fedbn/tables/robustness_runs.csv")
    rows.append(_utility_row(12, "FedBN-style diagnostic", "local BatchNorm", fedbn, std_cols, "42,123,2024",
                             "results/diagnostics/fedbn/tables/robustness_runs.csv"))
    dp = pd.read_csv("results/privacy/tables/dp_per_seed_privacy_utility.csv")
    rows.append(_utility_row(13, "DP-FedAvg (client-level)", "epsilon=7.98", dp, std_cols, "42,123,2024",
                             "results/privacy/tables/dp_per_seed_privacy_utility.csv",
                             "round 0 (untrained model) selected in all seeds"))
    he = pd.read_csv("results/he/tables/he_per_seed_utility_overhead.csv")
    rows.append(_utility_row(14, "HE-FedAvg (CKKS)", "encrypted aggregation", he, std_cols, "42,123,2024",
                             "results/he/tables/he_per_seed_utility_overhead.csv"))
    bc = pd.read_csv("results/blockchain_integration/tables/blockchain_integration_per_seed.csv")
    rows.append(_utility_row(16, "FedAvg + blockchain recording", "recorded", bc,
                             ["test_accuracy", "test_macro_f1", "test_weighted_f1", "x", "x"], "42,123,2024",
                             "results/blockchain_integration/tables/blockchain_integration_per_seed.csv",
                             "bit-identical to FedAvg without recording"))
    atk = pd.read_csv("results/security/tables/attack_runs.csv")
    for name, g in atk.groupby("attack", sort=False):
        rows.append(_utility_row(17, "FedAvg under attack (no defense)", name, g, std_cols, "42,123,2024",
                                 "results/security/tables/attack_runs.csv"))
    prop = pd.read_csv("results/proposed/tables/proposed_runs.csv")
    dcols = [f"defended_{m}" for m in MET]
    for name, g in prop.groupby("condition", sort=False):
        rows.append(_utility_row(18, "Proposed defense v1 (full)", name, g, dcols, "42,123,2024",
                                 "results/proposed/tables/proposed_runs.csv"))
    abl = pd.read_csv("results/ablation/tables/ablation_runs.csv")
    for (variant, cond), g in abl.groupby(["variant", "condition"], sort=False):
        rows.append(_utility_row(19, f"Ablation: {variant}", cond, g, std_cols, "42,123,2024",
                                 "results/ablation/tables/ablation_runs.csv"))
    for (arm, cond), g in final_runs.groupby(["arm", "condition"], sort=False):
        rows.append(_utility_row(20, f"Final confirmation: {arm}", cond, g, std_cols, "101,202,303",
                                 "results/final/final_confirmation_runs.csv", "fresh seeds"))
    return pd.DataFrame(rows)


def dimension_tables(final_runs):
    dp = pd.read_csv("results/privacy/tables/dp_per_seed_privacy_utility.csv")
    he = pd.read_csv("results/he/tables/he_per_seed_utility_overhead.csv")
    bcb = _json("results/blockchain/metrics/blockchain_benchmark.json")
    bci = pd.read_csv("results/blockchain_integration/tables/blockchain_integration_per_seed.csv")
    ver = pd.read_csv("results/blockchain_integration/tables/blockchain_integration_verification.csv")
    tam = pd.read_csv("results/blockchain_integration/tables/blockchain_integration_tamper_detection.csv")
    comm = _json("results/metrics/federated/fedavg_v1_communication_cost.json")
    privacy = pd.DataFrame([
        {"mechanism": "Federated learning (FedAvg)", "property": "raw ECG stays on simulated clients",
         "evidence": "design (simulation)", "formal_guarantee": "none", "value": None, "day": 10},
        {"mechanism": "DP-FedAvg", "property": "client-level (epsilon, delta)-DP, trusted server, full participation",
         "evidence": "GDP accountant (Dong-Roth-Su; Balle-Wang conversion)", "formal_guarantee": "yes (as stated)",
         "value": f"epsilon={dp['epsilon'].iloc[0]:.4f}, delta={dp['delta'].iloc[0]}, RDP bound {dp['epsilon_rdp_upper_bound'].iloc[0]:.2f}",
         "day": 13},
        {"mechanism": "HE-FedAvg (CKKS)", "property": "honest-but-curious server sees only ciphertexts of individual updates",
         "evidence": "server context has no secret key (tested); 128-bit parameter set", "formal_guarantee":
         "computational confidentiality of individual updates only; none for the aggregate/model", "value": "N=8192, 200-bit modulus, scale 2^40",
         "day": 14},
        {"mechanism": "Blockchain ledger", "property": "no confidentiality; stores hashes/IDs/counts only", "evidence":
         "schema + integration tests (no floats on-chain)", "formal_guarantee": "none", "value": None, "day": 16},
    ])
    integrity = pd.DataFrame([
        {"component": "ledger (synthetic)", "metric": "tamper detection with anchor (13 scenarios x 200)", "value": 1.0, "day": 15},
        {"component": "ledger (synthetic)", "metric": "tamper detection without anchor, full rewrite / truncation", "value": 0.0, "day": 15},
        {"component": "ledger (real FedAvg runs)", "metric": "on-chain hashes verified against off-chain artifacts",
         "value": f"{int(ver['hashes_matching'].sum())}/{int(ver['hashes_checked'].sum())}", "day": 16},
        {"component": "ledger (real FedAvg runs)", "metric": "tamper detection with anchor (14 scenarios, all trials)",
         "value": float((tam["detected_with_anchor"].sum()) / tam["trials"].sum()), "day": 16},
        {"component": "ledger (real FedAvg runs)", "metric": "recording changes training", "value": "no (bit-identical)", "day": 16},
    ])
    v5 = final_runs[final_runs.arm == "variant5"]
    ctrl = final_runs[final_runs.arm == "fedavg_control"]
    prop = pd.read_csv("results/proposed/tables/proposed_runs.csv")
    abl = pd.read_csv("results/ablation/tables/ablation_runs.csv")
    atk = pd.read_csv("results/security/tables/attack_runs.csv")
    rob_rows = []
    for cond in ("sign_flip", "scaled", "random_noise"):
        a = atk[atk.attack == cond]
        rob_rows.append({"day": 17, "method": "FedAvg (no defense)", "condition": cond, "seeds": "42,123,2024",
                         "detection_rate": None, "honest_fpr": None,
                         "nan_rounds_mean": float(a["nan_validation_rounds"].mean()),
                         "test_macro_f1_mean": float(a["test_macro_f1"].mean())})
        p = prop[prop.condition == cond]
        rob_rows.append({"day": 18, "method": "Defense v1 (full)", "condition": cond, "seeds": "42,123,2024",
                         "detection_rate": float(p["attack_detection_rate"].mean()),
                         "honest_fpr": float(p["honest_false_positive_rate"].mean()),
                         "nan_rounds_mean": float(p["nan_validation_rounds"].mean()),
                         "test_macro_f1_mean": float(p["defended_macro_f1"].mean())})
        b = abl[(abl.variant == "clip_cos_norm") & (abl.condition == cond)]
        rob_rows.append({"day": 19, "method": "Variant 5 (ablation)", "condition": cond, "seeds": "42,123,2024",
                         "detection_rate": float(b["attack_detection_rate"].mean()),
                         "honest_fpr": float(b["honest_false_positive_rate"].mean()),
                         "nan_rounds_mean": float(b["nan_validation_rounds"].mean()),
                         "test_macro_f1_mean": float(b["test_macro_f1"].mean())})
        for arm, frame, label in (("fedavg_control", ctrl, "FedAvg control"), ("variant5", v5, "Variant 5 (final)")):
            f = frame[frame.condition == cond]
            rob_rows.append({"day": 20, "method": label, "condition": cond, "seeds": "101,202,303",
                             "detection_rate": float(f["attack_detection_rate"].mean()) if arm == "variant5" else None,
                             "honest_fpr": float(f["honest_false_positive_rate"].mean()) if arm == "variant5" else None,
                             "nan_rounds_mean": float(f["nan_validation_rounds"].mean()),
                             "test_macro_f1_mean": float(f["test_macro_f1"].mean())})
    robustness = pd.DataFrame(rob_rows)
    communication = pd.DataFrame([
        {"setting": "Plaintext FedAvg", "bytes_per_model_transfer": comm["payload_bytes_per_model"],
         "total_bytes_20_rounds_5_clients": comm["total_bytes"], "ratio_vs_fedavg": 1.0, "measured": "analytical", "day": 10},
        {"setting": "DP-FedAvg", "bytes_per_model_transfer": comm["payload_bytes_per_model"],
         "total_bytes_20_rounds_5_clients": comm["total_bytes"], "ratio_vs_fedavg": 1.0, "measured": "analytical (0 extra bytes)", "day": 13},
        {"setting": "HE-FedAvg (all ciphertext traffic)", "bytes_per_model_transfer": float(he["uplink_ciphertext_bytes_per_client"].mean()),
         "total_bytes_20_rounds_5_clients": float(he["he_total_bytes_20_rounds"].mean()),
         "ratio_vs_fedavg": float(he["total_communication_overhead_ratio"].mean()), "measured": "serialized ciphertext sizes", "day": 14},
        {"setting": "Blockchain ledger (per run)", "bytes_per_model_transfer": None,
         "total_bytes_20_rounds_5_clients": float(bci["ledger_bytes"].mean()),
         "ratio_vs_fedavg": float(bci["ledger_bytes"].mean()) / comm["total_bytes"], "measured": "ledger file size", "day": 16},
        {"setting": "Proposed defense (v1 / Variant 5)", "bytes_per_model_transfer": comm["payload_bytes_per_model"],
         "total_bytes_20_rounds_5_clients": comm["total_bytes"], "ratio_vs_fedavg": 1.0, "measured": "0 extra bytes (server-side)", "day": 18},
    ])
    computation = pd.DataFrame([
        {"component": "HE: encryption per client per round", "seconds": float(he["encrypt_seconds_per_client_per_round"].mean()), "day": 14},
        {"component": "HE: server aggregation per round", "seconds": float(he["aggregate_seconds_per_round_server"].mean()), "day": 14},
        {"component": "HE: total over 20 rounds", "seconds": float(he["he_seconds_total_20_rounds"].mean()), "day": 14},
        {"component": "Blockchain: hashing + blocks per run (real FedAvg)",
         "seconds": float(bci["blockchain_seconds_total_hash_plus_blocks"].mean()), "day": 16},
        {"component": "Blockchain: create per block (synthetic)",
         "seconds": bcb["scaling"][-1]["create_ms_per_block"] / 1e3, "day": 15},
        {"component": "Defense v1: server time per round", "seconds": float(prop["defense_ms_per_round"].mean()) / 1e3, "day": 18},
        {"component": "Variant 5: server time per round (fresh seeds)", "seconds": float(v5["defense_ms_per_round"].mean()) / 1e3, "day": 20},
        {"component": "FedAvg training run (20 rounds, CPU)", "seconds": float(ctrl["training_seconds"].mean()), "day": 20},
    ])
    return privacy, integrity, robustness, communication, computation


def research_questions(final_runs):
    v5, ctrl = final_runs[final_runs.arm == "variant5"], final_runs[final_runs.arm == "fedavg_control"]

    def m(frame, cond, col="test_macro_f1"):
        return round(float(frame[frame.condition == cond][col].mean()), 4)
    return {
        "RQ1": {"question": "How accurately can ECG deep learning detect/classify arrhythmias centrally and federated?",
                "status": "partially answered",
                "evidence": ["Day 6 centralized test: accuracy 0.7688, macro-F1 0.1399", "Days 10-11 FedAvg macro-F1 0.085 +/- 0.013",
                             "Day 8 label audit: only 8 of 15 classes have test support"],
                "why_not_conclusive": "single dataset, provisional 15-symbol taxonomy, weak minority-class detection (N/V collapse), 7-record test set"},
        "RQ2": {"question": "FL vs centralized and local training under heterogeneous/non-IID data?",
                "status": "partially answered",
                "evidence": ["Days 10-11: FedAvg below centralized on accuracy/macro-F1/weighted-F1 in 6/6 runs",
                             "Day 12: BatchNorm locality does not explain the gap", "Day 11: label skew of / and L cannot be changed by whole-group partitions"],
                "why_not_conclusive": "local-only training baseline not run; cause of the collapse unresolved; 3 seeds, no significance testing"},
        "RQ3": {"question": "Effect of DP and HE on utility, privacy, computation and communication?",
                "status": "partially answered",
                "evidence": ["Day 13: client-level DP at epsilon=7.98 destroys utility (round 0 selected; noise-to-signal ~65)",
                             "Day 14: CKKS HE preserves utility within 1.3e-7 error; ~20.5x communication; ~2% compute"],
                "why_not_conclusive": "single epsilon, client-level unit only (no record-level DP-SGD, no epsilon sweep); DP+HE never combined"},
        "RQ4": {"question": "Can permissioned blockchain improve integrity, traceability, verification, auditability of FL coordination?",
                "status": "partially answered",
                "evidence": ["Day 15: hash-chained ledger detects all 13 tamper scenarios with an external anchor",
                             "Day 16: 363/363 recorded hashes verified; recording bit-identical to FedAvg; ~0.025% overhead"],
                "why_not_conclusive": "single-node prototype: no permissioned network, consensus, replication, signatures or client authentication"},
        "RQ5": {"question": "Can a proposed aggregation/coordination mechanism improve an identified limitation at acceptable cost?",
                "status": "partially answered",
                "evidence": ["Day 17: plain FedAvg breaks under one sign-flip or x10 client",
                             "Days 18-19: cosine filter stops sign flip, norm filter stops scaling; BatchNorm median unnecessary",
                             f"Day 20 fresh seeds: Variant 5 macro-F1 clean {m(v5, 'clean')} vs control {m(ctrl, 'clean')}; "
                             f"scaled {m(v5, 'scaled')} vs {m(ctrl, 'scaled')}; sign flip {m(v5, 'sign_flip')} vs {m(ctrl, 'sign_flip')}"],
                "why_not_conclusive": "one non-adaptive attacker, 3 attack settings, 3+3 seeds; Variant 5 selected after Day 19 results "
                                      "(confirmed on fresh seeds but not on a new dataset); not integrated with DP/HE/blockchain"},
    }


def manifest():
    commits = subprocess.run(["git", "log", "--format=%h %s"], capture_output=True, text=True, check=True).stdout.splitlines()
    experiments = [
        {"day": 6, "id": "centralized_cnn_v1", "configs": ["configs/training/centralized_cnn_v1.json"], "seeds": [42],
         "outputs": ["results/metrics", "results/tables", "results/figures"], "doc": "docs/evaluation_baseline.md"},
        {"day": 9, "id": "baseline_freeze", "configs": [], "seeds": [], "outputs": [], "doc": "docs/baseline_freeze.md"},
        {"day": 10, "id": "fedavg_v1", "configs": ["configs/federated/fedavg_v1.json"], "seeds": [42],
         "outputs": ["results/metrics/federated", "results/tables/federated", "results/figures/federated"],
         "doc": "docs/federated_learning_baseline.md"},
        {"day": 11, "id": "fedavg_v1_robustness", "configs": ["configs/federated/fedavg_v1_robustness.json"],
         "seeds": [42, 123, 2024], "outputs": ["results/robustness"], "doc": "docs/fedavg_robustness.md"},
        {"day": 12, "id": "fedbn_diagnostic_v1", "configs": ["configs/federated/fedbn_diagnostic_v1.json"],
         "seeds": [42, 123, 2024], "outputs": ["results/diagnostics/fedbn"], "doc": "docs/fedbn_diagnostic.md"},
        {"day": 13, "id": "dp_fedavg_v1", "configs": ["configs/privacy/dp_fedavg_v1.json"], "seeds": [42, 123, 2024],
         "outputs": ["results/privacy"], "doc": "docs/differential_privacy.md"},
        {"day": 14, "id": "he_fedavg_v1", "configs": ["configs/he/he_fedavg_v1.json"], "seeds": [42, 123, 2024],
         "outputs": ["results/he"], "doc": "docs/homomorphic_encryption.md"},
        {"day": 15, "id": "ledger_v1", "configs": ["configs/blockchain/ledger_v1.json"], "seeds": [42],
         "outputs": ["results/blockchain"], "doc": "docs/blockchain.md"},
        {"day": 16, "id": "fedavg_blockchain_v1", "configs": ["configs/blockchain/fedavg_blockchain_v1.json"],
         "seeds": [42, 123, 2024], "outputs": ["results/blockchain_integration"], "doc": "docs/blockchain_fl_integration.md"},
        {"day": 17, "id": "malicious_clients_v1", "configs": ["configs/security/attacks_v1.json"], "seeds": [42, 123, 2024],
         "outputs": ["results/security"], "doc": "docs/malicious_clients.md"},
        {"day": 18, "id": "proposed_robust_fedavg_v1",
         "configs": ["configs/proposed/calibration_v1.json", "configs/proposed/robust_defense_v1.json"],
         "seeds": [7, 42, 123, 2024], "outputs": ["results/proposed"], "doc": "docs/proposed_method.md"},
        {"day": 19, "id": "proposed_defense_ablation_v1", "configs": ["configs/ablation/ablation_v1.json"],
         "seeds": [42, 123, 2024], "outputs": ["results/ablation"], "doc": "docs/ablation.md"},
        {"day": 20, "id": "final_independent_confirmation_v1", "configs": ["configs/final/final_validation_v1.json"],
         "seeds": [101, 202, 303], "outputs": ["results/final"], "doc": "docs/final_results.md"},
    ]
    key = ["results/models/centralized_cnn_v1.pt", "results/metrics/baseline_metrics.json",
           "results/metrics/baseline_test_predictions.npz", "results/models/federated/fedavg_v1_global.pt",
           "results/metrics/federated/fedavg_v1_test_predictions.npz", "results/metrics/federated/fedavg_v1_test_metrics.json",
           "data/splits/mitbih_v1/split_metadata.json", "configs/preprocessing/mitbih_v1.json",
           "configs/preprocessing/mitbih_split_v1.json", "configs/training/centralized_cnn_v1.json",
           "configs/federated/fedavg_v1.json", "configs/proposed/robust_defense_v1.json", "configs/final/final_validation_v1.json"]
    return {"study": "A Blockchain-Enabled Federated Learning Framework for Privacy-Preserving Arrhythmia Detection",
            "frozen_on": "2026-10-05", "dataset": "MIT-BIH Arrhythmia Database v1.0.0 (PhysioNet), 48 records",
            "experiments": experiments, "key_artifact_sha256": {p: _sha(p) for p in key},
            "git_history_at_freeze": commits[:25],
            "regenerable_untracked": ["data/raw", "data/processed", "data/splits/*/*/*.npz", "*/models/ (per-run checkpoints)",
                                      "*/metrics/federated/*.npz (per-run predictions/trajectories)",
                                      "results/blockchain_integration/offchain", "results/ablation/figures/federated/*.png",
                                      "results/final/runs/figures/federated/*.png"]}


def write_all():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    final_runs = collect_final()
    final_runs.to_csv(OUT / "final_confirmation_runs.csv", index=False)
    summary_rows = []
    for (arm, cond), g in final_runs.groupby(["arm", "condition"], sort=False):
        for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_recall", "test_macro_auroc",
                       "selected_round", "final_round_validation_macro_f1", "nan_validation_rounds",
                       "rounds_with_negative_bn_running_var", "attack_detection_rate", "honest_false_positive_rate",
                       "relative_deviation_from_clean_fedavg_final", "defense_ms_per_round", "training_seconds",
                       "n_predicted_classes"]:
            summary_rows.append({"arm": arm, "condition": cond, "metric": metric, **_stats(g[metric])})
    confirmation = pd.DataFrame(summary_rows)
    confirmation.to_csv(OUT / "final_confirmation_summary.csv", index=False)
    utility = utility_table(final_runs)
    privacy, integrity, robustness, communication, computation = dimension_tables(final_runs)
    for name, frame in (("utility", utility), ("privacy_confidentiality", privacy), ("integrity_auditability", integrity),
                        ("robustness", robustness), ("communication", communication), ("computation", computation)):
        frame.to_csv(OUT / f"final_{name}.csv", index=False)
    rqs = research_questions(final_runs)
    summary = {
        "frozen_on": "2026-10-05", "scope": "simulated 5-client federation on public MIT-BIH data; one CPU machine",
        "utility": utility.to_dict("records"), "privacy_confidentiality": privacy.to_dict("records"),
        "integrity_auditability": integrity.to_dict("records"), "robustness": robustness.to_dict("records"),
        "communication": communication.to_dict("records"), "computation": computation.to_dict("records"),
        "final_confirmation": confirmation.to_dict("records"), "research_questions": rqs,
        "not_demonstrated": ["clinical usefulness or diagnostic validity", "record-level or local DP",
                             "DP, HE, blockchain and defense combined in one system", "distributed/permissioned blockchain consensus",
                             "robustness to adaptive, colluding or data-poisoning attackers", "generalization beyond MIT-BIH",
                             "statistical significance of any comparison"],
    }
    (OUT / "final_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest(), indent=2) + "\n", encoding="utf-8")
    from src.federated import plots
    plots.plot_final_overview(utility, OUT / "figures" / "final_utility_overview.png")
    plots.plot_final_confirmation(final_runs, OUT / "figures" / "final_confirmation.png")
    plots.plot_final_costs(communication, computation, OUT / "figures" / "final_costs.png")
    return summary


if __name__ == "__main__":
    write_all()
