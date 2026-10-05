"""Day 11 FedAvg robustness runs: fixed configuration, predefined seeds, group partitions.

Each run reuses `run_experiment` with the unchanged `fedavg_v1` training
configuration. Only `random_seed`, `partition.seed`, `partition.strategy`, and
`experiment_id` are overridden. Outputs go under `results/robustness/` so the
official Day 10 `fedavg_v1` artifacts are never touched. Every run selects its
round on validation loss and evaluates the test split once.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_federated_config
from .evaluation import compare_with_centralized
from .data import load_class_to_index
from .heterogeneity import client_heterogeneity
from .run_fedavg import run_experiment
from . import plots

MAIN_METRICS = ["selected_round", "best_validation_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1",
                "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
                "test_weighted_precision", "test_weighted_recall", "test_weighted_f1",
                "test_macro_auroc", "n_predicted_classes"]


def load_robustness_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be unique")
    return config


def planned_runs(config):
    """(run_id, partition_strategy, partition_label, seed), primary partition first."""
    runs = []
    for strategy in [config["primary_partition"], *config.get("diagnostic_partitions", [])]:
        label = config["partition_labels"][strategy]
        for seed in config["seeds"]:
            run_id = config["run_id_template"].format(partition_label=label, seed=seed)
            runs.append((run_id, strategy, label, int(seed)))
    return runs


def run_overrides(run_id, strategy, seed):
    return {"experiment_id": run_id, "random_seed": seed, "partition": {"seed": seed, "strategy": strategy}}


def run_all(config_path="configs/federated/fedavg_v1_robustness.json", log=print):
    """Execute every planned run once; completed runs are loaded, not re-run or re-tested."""
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    summaries = {}
    for run_id, strategy, _, seed in planned_runs(config):
        summary_path = root / "metrics" / "federated" / f"{run_id}_run_summary.json"
        if summary_path.exists():
            log(f"{run_id}: already complete, loading saved summary")
            summaries[run_id] = json.loads(summary_path.read_text(encoding="utf-8"))
            continue
        log(f"{run_id}: training")
        summaries[run_id] = run_experiment(config["base_config"], evaluate_test=True, output_root=root,
                                           overrides=run_overrides(run_id, strategy, seed), log=lambda *a: None,
                                           comparison_prefix=f"{run_id}_")
    return summaries


def collect_results(config_path="configs/federated/fedavg_v1_robustness.json"):
    """Build run-level, summary, per-class, heterogeneity, and history tables from saved outputs."""
    config = load_robustness_config(config_path)
    base = load_federated_config(config["base_config"])
    class_names = list(load_class_to_index(base["data"]["label_config"]))
    root = Path(config["output_root"])
    metrics_dir = root / "metrics" / "federated"
    run_rows, per_class, hetero_rows, hetero_summary, history_rows = [], [], [], [], []
    for run_id, strategy, label, seed in planned_runs(config):
        summary = json.loads((metrics_dir / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        test = json.loads((metrics_dir / f"{run_id}_test_metrics.json").read_text(encoding="utf-8"))["metrics"]
        partition = json.loads((metrics_dir / f"{run_id}_client_partition.json").read_text(encoding="utf-8"))
        predictions = np.load(metrics_dir / f"{run_id}_test_predictions.npz", allow_pickle=False)
        predicted = sorted({class_names[i] for i in predictions["y_pred"].tolist()})
        sel = summary["selected_round_validation"]
        clients, het = client_heterogeneity(partition, class_names)
        run_rows.append({
            "run_id": run_id, "partition": label, "seed": seed,
            "selected_round": summary["selected_round_by_validation_loss"],
            "best_validation_loss": sel["validation_loss"], "val_accuracy": sel["validation_accuracy"],
            "val_macro_f1": sel["validation_macro_f1"], "val_weighted_f1": sel["validation_weighted_f1"],
            "test_accuracy": test["accuracy"], "test_macro_precision": test["macro_precision"],
            "test_macro_recall": test["macro_recall_all_15_labels_zero_for_no_support"],
            "test_macro_f1": test["macro_f1"], "test_weighted_precision": test["weighted_precision"],
            "test_weighted_recall": test["weighted_recall"], "test_weighted_f1": test["weighted_f1"],
            "test_macro_auroc": test["macro_auroc_ovr_defined_classes"],
            "test_weighted_auroc": test["weighted_auroc_ovr_defined_classes"],
            "n_predicted_classes": len(predicted), "predicted_classes": " ".join(predicted),
            "predicts_only_N_V": set(predicted) <= {"N", "V"},
            "client_segments": " ".join(str(c["segments"]) for c in partition["clients"]),
            "client_class_absences": het["client_class_absences"],
            "mean_js_divergence_vs_pooled_bits": het["mean_js_divergence_vs_pooled_bits"],
        })
        per_class_path = root / "tables" / "federated" / f"{run_id}_per_class_metrics.csv"
        for row in pd.read_csv(per_class_path, keep_default_na=False, na_values=[""]).to_dict("records"):
            per_class.append({"run_id": run_id, "partition": label, "seed": seed, **row})
        for row in clients:
            hetero_rows.append({"run_id": run_id, "partition": label, "seed": seed, **row})
        hetero_summary.append({"run_id": run_id, "partition": label, "seed": seed,
                               **{k: v for k, v in het.items() if k != "clients_per_class"},
                               **{f"clients_with_{c}": n for c, n in het["clients_per_class"].items()}})
        history = pd.read_csv(metrics_dir / f"{run_id}_round_history.csv")
        for row in history.to_dict("records"):
            history_rows.append({"run_id": run_id, "partition": label, "seed": seed, **row})
    runs = pd.DataFrame(run_rows)
    summary_rows = []
    for label, group in runs.groupby("partition", sort=False):
        for metric in MAIN_METRICS:
            values = group[metric].astype(float)
            summary_rows.append({"partition": label, "metric": metric, "n_runs": len(values),
                                 "mean": values.mean(), "std_sample": values.std(ddof=1),
                                 "min": values.min(), "max": values.max()})
    return {"runs": runs, "summary": pd.DataFrame(summary_rows), "per_class": pd.DataFrame(per_class),
            "heterogeneity_clients": pd.DataFrame(hetero_rows), "heterogeneity_summary": pd.DataFrame(hetero_summary),
            "history": pd.DataFrame(history_rows)}


def write_comparison_tables(config_path="configs/federated/fedavg_v1_robustness.json"):
    """Per-run centralized-vs-FedAvg tables rebuilt from each run's saved test outputs.

    Post-processing only: reads `<run>_test_metrics.json` and `<run>_per_class_metrics.csv`;
    no model is trained or evaluated.
    """
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    metrics_dir, tables_dir = root / "metrics" / "federated", root / "tables" / "federated"
    for run_id, _, _, _ in planned_runs(config):
        aggregate = json.loads((metrics_dir / f"{run_id}_test_metrics.json").read_text(encoding="utf-8"))["metrics"]
        per_class = pd.read_csv(tables_dir / f"{run_id}_per_class_metrics.csv", keep_default_na=False, na_values=[""])
        per_class = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
                     for row in per_class.to_dict("records")]
        aggregate_rows, per_class_rows = compare_with_centralized(aggregate, per_class, label=run_id)
        pd.DataFrame(aggregate_rows).to_csv(tables_dir / f"{run_id}_centralized_vs_fedavg_aggregate.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(tables_dir / f"{run_id}_centralized_vs_fedavg_per_class.csv", index=False)


def write_reports(config_path="configs/federated/fedavg_v1_robustness.json",
                  centralized_metrics_path="results/metrics/baseline_metrics.json",
                  centralized_history_path="results/metrics/centralized_cnn_v1_history.json"):
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_comparison_tables(config_path)
    results = collect_results(config_path)
    for name, frame in results.items():
        frame.to_csv(tables / f"robustness_{name}.csv", index=False)
    central = json.loads(Path(centralized_metrics_path).read_text(encoding="utf-8"))["metrics"]
    history = json.loads(Path(centralized_history_path).read_text(encoding="utf-8"))
    epoch = history["history"][history["best_epoch_by_validation_loss"] - 1]
    reference_val = {"validation_accuracy": epoch["validation_accuracy"],
                     "validation_macro_f1": epoch["validation_macro_f1_all_labels"],
                     "validation_loss": epoch["validation_loss"]}
    reference_test = {"test_accuracy": central["accuracy"], "test_macro_f1": central["macro_f1"],
                      "test_weighted_f1": central["weighted_f1"]}
    plots.plot_robustness_convergence(results["history"], reference_val, figures / "robustness_convergence.png")
    plots.plot_robustness_test_metrics(results["runs"], reference_test, figures / "robustness_test_metrics.png")
    return results
