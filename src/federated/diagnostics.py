"""Paired comparison of a one-factor FL diagnostic (e.g. FedBN-style) against Day 11 FedAvg.

Pairs are matched by seed: same natural partition, initialization and training
order, so the BatchNorm handling is the only configured difference. Reads saved
outputs only; nothing is trained or evaluated here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_class_to_index
from .config import load_federated_config
from .fedavg import batchnorm_state_keys, get_model_state
from .robustness import collect_results, load_robustness_config, planned_runs
from src.models.cnn1d import ECG1DCNN
from . import plots

PAIRED_METRICS = ["selected_round", "best_validation_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1",
                  "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
                  "test_weighted_precision", "test_weighted_recall", "test_weighted_f1",
                  "test_macro_auroc", "test_weighted_auroc", "n_predicted_classes"]


def _predicted_counts(confusion_csv, class_names):
    cm = pd.read_csv(confusion_csv, index_col=0)
    return {name: int(cm[name].sum()) for name in class_names}


def batchnorm_payload():
    """Bytes of BatchNorm vs shared (non-BatchNorm) state entries for the frozen architecture."""
    model = ECG1DCNN()
    state = get_model_state(model)
    keys = set(batchnorm_state_keys(model))
    size = lambda ks: int(sum(state[k].numel() * state[k].element_size() for k in ks))  # noqa: E731
    return {"batchnorm_bytes": size(keys), "shared_bytes": size(set(state) - keys), "full_bytes": size(state)}


def paired_comparison(config_path="configs/federated/fedbn_diagnostic_v1.json"):
    config = load_robustness_config(config_path)
    comparator = config["comparator"]
    class_names = list(load_class_to_index(load_federated_config(config["base_config"])["data"]["label_config"]))
    root = Path(config["output_root"])
    arm = config.get("arm_label", "fedbn")
    diag = collect_results(config_path)
    base_runs = pd.read_csv(comparator["runs_table"])
    paired, counts, per_class, personalised = [], [], [], []
    for run_id, _, _, seed in planned_runs(config):
        base_id = comparator["run_id_template"].format(seed=seed)
        d = diag["runs"].set_index("run_id").loc[run_id]
        b = base_runs.set_index("run_id").loc[base_id]
        for metric in PAIRED_METRICS:
            paired.append({"seed": seed, "metric": metric, "fedavg": float(b[metric]), arm: float(d[metric]),
                           f"delta_{arm}_minus_fedavg": float(d[metric]) - float(b[metric])})
        for arm_name, rid, tables in (("fedavg", base_id, Path(comparator["tables_dir"])),
                                      (arm, run_id, root / "tables" / "federated")):
            pc = _predicted_counts(tables / f"{rid}_confusion_matrix.csv", class_names)
            total = sum(pc.values())
            counts.append({"seed": seed, "arm": arm_name, "run_id": rid, **pc,
                           "share_N_or_V": (pc["N"] + pc["V"]) / total,
                           "classes_predicted": sum(1 for v in pc.values() if v > 0)})
            metrics = pd.read_csv(tables / f"{rid}_per_class_metrics.csv", keep_default_na=False, na_values=[""])
            for row in metrics.to_dict("records"):
                if row["support"] > 0:
                    per_class.append({"seed": seed, "arm": arm_name, "label": row["label"], "support": int(row["support"]),
                                      "precision": row["precision"], "recall": row["recall"], "f1_score": row["f1_score"],
                                      "specificity": row["specificity"], "auroc_ovr": row["auroc_ovr"]})
        history_path = root / "metrics" / "federated" / f"{run_id}_client_validation_history.csv"
        selected = int(d["selected_round"])
        clients = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame(columns=["round"])
        for row in clients[clients["round"] == selected].to_dict("records"):
            personalised.append({"seed": seed, "selected_round": selected, **row,
                                 "evaluation_model_val_macro_f1": float(d["val_macro_f1"]),
                                 "evaluation_model_val_accuracy": float(d["val_accuracy"])})
    paired = pd.DataFrame(paired)
    summary = (paired.groupby("metric", sort=False)[f"delta_{arm}_minus_fedavg"]
               .agg(mean_delta="mean", min_delta="min", max_delta="max",
                    **{f"seeds_{arm}_higher": lambda s: int((s > 0).sum()),
                       f"seeds_{arm}_lower": lambda s: int((s < 0).sum())})
               .reset_index())
    return {"paired": paired, "paired_summary": summary, "prediction_counts": pd.DataFrame(counts),
            "per_class": pd.DataFrame(per_class), "personalised_validation": pd.DataFrame(personalised)}


def write_diagnostic_reports(config_path="configs/federated/fedbn_diagnostic_v1.json",
                             centralized_metrics_path="results/metrics/baseline_metrics.json"):
    config = load_robustness_config(config_path)
    root = Path(config["output_root"])
    tables, figures = root / "tables", root / "figures"
    results = paired_comparison(config_path)
    prefix = config.get("comparison_prefix", "fedbn_vs_fedavg")
    for name, frame in results.items():
        if not frame.empty:
            frame.to_csv(tables / f"{prefix}_{name}.csv", index=False)
    payload = batchnorm_payload()
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "metrics" / "fedbn_payload.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    central = json.loads(Path(centralized_metrics_path).read_text(encoding="utf-8"))["metrics"]
    reference = {"test_accuracy": central["accuracy"], "test_macro_f1": central["macro_f1"],
                 "test_weighted_f1": central["weighted_f1"],
                 "test_macro_recall": central["macro_recall_all_15_labels_zero_for_no_support"]}
    plots.plot_paired_slopes(results["paired"], reference, figures / f"{prefix}_paired.png",
                             arm=config.get("arm_label", "fedbn"),
                             arm_title=config.get("arm_title", "FedBN-style\n(local BatchNorm)"),
                             suptitle=config.get("paired_title", "Seed-paired comparison: same partition, initialization "
                                                 "and training order; only BatchNorm handling differs"))
    return results, payload
