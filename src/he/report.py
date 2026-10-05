"""Day 14 HE-FedAvg reporting: utility vs paired plaintext FedAvg, numerical error, time and size overhead.

Reads saved outputs only (HE runs under results/he, paired Day 11 FedAvg runs
under results/robustness). Nothing is trained or evaluated here.
"""
import json
from pathlib import Path

import pandas as pd

from src.federated.diagnostics import write_diagnostic_reports
from src.federated.robustness import load_robustness_config, planned_runs, write_reports
from src.federated import plots

UTILITY = ["selected_round", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
           "test_weighted_f1", "test_macro_auroc", "n_predicted_classes"]
CLASSES = ["/", "A", "E", "F", "J", "L", "N", "Q", "R", "S", "V", "a", "e", "f", "j"]


def write_he_reports(config_path="configs/he/he_fedavg_v1.json"):
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    comparator = config["comparator"]
    robustness = write_reports(config_path)
    paired, _ = write_diagnostic_reports(config_path)
    runs = robustness["runs"].set_index("run_id")
    base_runs = pd.read_csv(comparator["runs_table"]).set_index("run_id")
    base_metrics = Path(comparator["runs_table"]).parents[1] / "metrics" / "federated"
    metrics_dir = root / "metrics" / "federated"
    rows, rounds = [], []
    for run_id, _, _, seed in planned_runs(config):
        base_id = comparator["run_id_template"].format(seed=seed)
        he = pd.read_csv(metrics_dir / f"{run_id}_he_round_diagnostics.csv")
        setup = json.loads((metrics_dir / f"{run_id}_he_setup.json").read_text(encoding="utf-8"))
        summary = json.loads((metrics_dir / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        base_summary = json.loads((base_metrics / f"{base_id}_run_summary.json").read_text(encoding="utf-8"))
        comm = json.loads((base_metrics / f"{base_id}_communication_cost.json").read_text(encoding="utf-8"))
        r, b = runs.loc[run_id], base_runs.loc[base_id]
        counts = paired["prediction_counts"]
        dist = counts[(counts["seed"] == seed) & (counts["arm"] == config["arm_label"])].iloc[0]
        plain_payload = comm["payload_bytes_per_model"]
        participants = int(he["participants"].iloc[0])
        up = float(he["uplink_ciphertext_bytes_per_client_mean"].mean())
        down = float(he["downlink_ciphertext_bytes_per_client"].mean())
        he_total = float((he["uplink_ciphertext_bytes_total"] + participants * he["downlink_ciphertext_bytes_per_client"]).sum())
        rows.append({
            "seed": seed, "run_id": run_id, "paired_fedavg_run": base_id,
            **{m: float(r[m]) for m in UTILITY},
            **{f"fedavg_{m}": float(b[m]) for m in UTILITY},
            **{f"delta_{m}": float(r[m]) - float(b[m]) for m in UTILITY},
            "predicted_class_distribution": json.dumps({c: int(dist[c]) for c in CLASSES if int(dist[c]) > 0}),
            "max_aggregate_abs_error_all_rounds": float(he["aggregate_max_abs_error"].max()),
            "mean_aggregate_rms_error": float(he["aggregate_rms_error"].mean()),
            "max_relative_error_all_rounds": float((he["aggregate_max_abs_error"] / he["aggregate_max_abs_value"]).max()),
            "max_state_abs_error_vs_plaintext_fedavg": float(he["state_max_abs_error_vs_plaintext_fedavg"].max()),
            "encrypt_seconds_per_round_all_clients": float(he["encrypt_seconds_all_clients"].mean()),
            "encrypt_seconds_per_client_per_round": float(he["encrypt_seconds_all_clients"].mean()) / participants,
            "aggregate_seconds_per_round_server": float(he["aggregate_seconds_server"].mean()),
            "decrypt_seconds_per_round": float(he["decrypt_seconds"].mean()),
            "he_seconds_total_20_rounds": float((he["encrypt_seconds_all_clients"] + he["aggregate_seconds_server"]
                                                 + he["decrypt_seconds"]).sum()),
            "key_generation_seconds": setup["key_generation_seconds"],
            "training_seconds": summary["training_elapsed_seconds"],
            "fedavg_training_seconds": base_summary["training_elapsed_seconds"],
            "ciphertext_chunks_per_client": setup["ciphertext_chunks_per_client"],
            "plaintext_payload_bytes_per_model": plain_payload,
            "uplink_ciphertext_bytes_per_client": up, "downlink_ciphertext_bytes_per_client": down,
            "uplink_expansion_vs_plaintext": up / plain_payload, "downlink_expansion_vs_plaintext": down / plain_payload,
            "public_context_bytes_one_time": setup["public_context_bytes"],
            "he_total_bytes_20_rounds": he_total, "plaintext_total_bytes_20_rounds": comm["total_bytes"],
            "total_communication_overhead_ratio": he_total / comm["total_bytes"],
        })
        he.insert(0, "seed", seed)
        rounds.append(he)
    per_seed = pd.DataFrame(rows)
    rounds = pd.concat(rounds, ignore_index=True)
    summary_rows = []
    numeric = [c for c in per_seed.columns if pd.api.types.is_numeric_dtype(per_seed[c]) and c != "seed"]
    for column in numeric:
        values = per_seed[column]
        summary_rows.append({"quantity": column, "mean": values.mean(), "std_sample": values.std(ddof=1),
                             "min": values.min(), "max": values.max(), "n_runs": len(values)})
    summary = pd.DataFrame(summary_rows)
    tables = root / "tables"
    per_seed.to_csv(tables / "he_per_seed_utility_overhead.csv", index=False)
    summary.to_csv(tables / "he_vs_fedavg_3seed_summary.csv", index=False)
    rounds.to_csv(tables / "he_round_diagnostics_all_seeds.csv", index=False)
    plots.plot_he_error(rounds, root / "figures" / "he_numerical_error.png")
    return per_seed, summary, rounds
