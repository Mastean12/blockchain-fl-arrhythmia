"""Day 13 privacy reporting: per-seed privacy/utility table, 3-seed summary, communication overhead.

Reads saved outputs only (DP runs under results/privacy, paired Day 11 FedAvg runs
under results/robustness). Nothing is trained or evaluated.
"""
import json
from pathlib import Path

import pandas as pd

from .diagnostics import write_diagnostic_reports
from .robustness import load_robustness_config, planned_runs, write_reports
from . import plots

SUMMARY_METRICS = ["selected_round", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
                   "test_weighted_f1", "test_macro_auroc", "n_predicted_classes"]


def write_privacy_reports(config_path="configs/privacy/dp_fedavg_v1.json"):
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    comparator = config["comparator"]
    robustness = write_reports(config_path)  # per-run tables, history, convergence/test figures
    paired, _ = write_diagnostic_reports(config_path)  # seed-paired deltas, prediction counts, per-class
    runs = robustness["runs"].set_index("run_id")
    base_runs = pd.read_csv(comparator["runs_table"]).set_index("run_id")
    metrics_dir = root / "metrics" / "federated"
    rows, diagnostics = [], []
    for run_id, _, _, seed in planned_runs(config):
        base_id = comparator["run_id_template"].format(seed=seed)
        accounting = json.loads((metrics_dir / f"{run_id}_privacy_accounting.json").read_text(encoding="utf-8"))
        summary = json.loads((metrics_dir / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        base_summary = json.loads((Path(comparator["runs_table"]).parents[1] / "metrics" / "federated" /
                                   f"{base_id}_run_summary.json").read_text(encoding="utf-8"))
        comm = json.loads((metrics_dir / f"{run_id}_communication_cost.json").read_text(encoding="utf-8"))
        acc, cfg = accounting["accounting"], accounting["privacy_config"]
        r, b = runs.loc[run_id], base_runs.loc[base_id]
        counts = paired["prediction_counts"]
        dist = counts[(counts["seed"] == seed) & (counts["arm"] == config["arm_label"])].iloc[0]
        predicted = {c: int(dist[c]) for c in ["/", "A", "E", "F", "J", "L", "N", "Q", "R", "S", "V", "a", "e", "f", "j"]
                     if int(dist[c]) > 0}
        rows.append({
            "seed": seed, "run_id": run_id, "paired_fedavg_run": base_id,
            "privacy_unit": cfg["privacy_unit"], "clipping_norm": cfg["clipping_norm"],
            "noise_multiplier": cfg["noise_multiplier"], "client_sampling_rate": cfg["client_sampling_rate"],
            "rounds": cfg["rounds"], "delta": acc["delta"], "epsilon": acc["epsilon"], "mu_gdp": acc["mu_gdp"],
            "epsilon_rdp_upper_bound": acc["epsilon_rdp_upper_bound"],
            **{m: float(r[m]) for m in SUMMARY_METRICS},
            **{f"fedavg_{m}": float(b[m]) for m in SUMMARY_METRICS},
            **{f"delta_{m}": float(r[m]) - float(b[m]) for m in SUMMARY_METRICS if m != "selected_round"},
            "relative_macro_f1_change": (float(r["test_macro_f1"]) - float(b["test_macro_f1"])) / float(b["test_macro_f1"]),
            "predicted_class_distribution": json.dumps(predicted, sort_keys=True),
            "payload_bytes_per_model": comm["payload_bytes_per_model"], "total_bytes_20_rounds": comm["total_bytes"],
            "dp_extra_bytes": 0,
            "training_seconds": summary["training_elapsed_seconds"],
            "fedavg_training_seconds": base_summary["training_elapsed_seconds"],
        })
        history = pd.read_csv(metrics_dir / f"{run_id}_dp_round_diagnostics.csv")
        history.insert(0, "seed", seed)
        diagnostics.append(history)
    per_seed = pd.DataFrame(rows)
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    summary_rows = []
    for metric in SUMMARY_METRICS:
        for arm, column in (("dp_fedavg", metric), ("fedavg", f"fedavg_{metric}")):
            values = per_seed[column]
            summary_rows.append({"metric": metric, "arm": arm, "mean": values.mean(), "std_sample": values.std(ddof=1),
                                 "min": values.min(), "max": values.max(), "n_runs": len(values)})
    summary = pd.DataFrame(summary_rows)
    tables = root / "tables"
    per_seed.to_csv(tables / "dp_per_seed_privacy_utility.csv", index=False)
    summary.to_csv(tables / "dp_vs_fedavg_3seed_summary.csv", index=False)
    diagnostics.to_csv(tables / "dp_round_diagnostics_all_seeds.csv", index=False)
    plots.plot_dp_noise_vs_signal(diagnostics, root / "figures" / "dp_noise_vs_signal.png")
    return per_seed, summary, diagnostics
