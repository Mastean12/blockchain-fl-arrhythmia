"""Day 19 component ablation of the frozen Day 18 defense.

New runs are needed only for variants 2-5; variant 1 (plain FedAvg) reuses the Day 11
clean and Day 17 attacked runs, and variant 6 (full method) reuses the Day 18 runs.
Every new run is the Day 18 run configuration with an added ``components`` switch;
thresholds are never changed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .proposed_study import load_proposed_config, run_overrides as defended_overrides
from .run_fedavg import run_experiment
from .security import _min_running_var, _prediction_profile, load_security_config
from . import plots


def load_ablation_config(path="configs/ablation/ablation_v1.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_runs(config):
    return [(config["run_id_template"].format(variant=v, condition=c, seed=s), v, c, int(s))
            for v, spec in config["variants"].items() if spec["source"] == "new"
            for c in config["conditions"] for s in config["seeds"]]


def ablation_overrides(config, run_id, variant, condition, seed):
    frozen = load_proposed_config(config["frozen_defense_config"])
    attacks = load_security_config(config["attack_config"])
    overrides = defended_overrides(frozen, attacks, run_id, condition, seed)
    overrides["defense"]["components"] = dict(config["variants"][variant]["components"])
    overrides["defense"]["name"] = f"{frozen['defense']['name']}__{variant}"
    return overrides


def run_all(config_path="configs/ablation/ablation_v1.json", log=print):
    config = load_ablation_config(config_path)
    root = Path(config["output_root"])
    out = {}
    for run_id, variant, condition, seed in new_runs(config):
        path = root / "metrics" / "federated" / f"{run_id}_run_summary.json"
        if path.exists():
            log(f"{run_id}: already complete, loading")
            out[run_id] = json.loads(path.read_text(encoding="utf-8"))
            continue
        log(f"{run_id}: training")
        out[run_id] = run_experiment(load_proposed_config(config["frozen_defense_config"])["base_config"],
                                     evaluate_test=True, output_root=root,
                                     overrides=ablation_overrides(config, run_id, variant, condition, seed),
                                     comparison_prefix=f"{run_id}_", log=lambda *a: None)
    return out


def _locate(config, variant, condition, seed):
    """(metrics_dir, tables_dir, run_id) for any variant/condition/seed, new or reused."""
    attacks = load_security_config(config["attack_config"])
    if variant == "plain_fedavg":
        if condition == "clean":
            comp = attacks["comparator"]
            run_id = comp["run_id_template"].format(seed=seed)
            return Path(comp["metrics_dir"]), Path(comp["tables_dir"]), run_id
        root = Path(attacks["output_root"])
        return (root / "metrics/federated", root / "tables/federated",
                attacks["run_id_template"].format(attack=condition, seed=seed))
    if variant == "full":
        frozen = load_proposed_config(config["frozen_defense_config"])
        root = Path(frozen["output_root"])
        return (root / "metrics/federated", root / "tables/federated",
                frozen["run_id_template"].format(condition=condition, seed=seed))
    root = Path(config["output_root"])
    return (root / "metrics/federated", root / "tables/federated",
            config["run_id_template"].format(variant=variant, condition=condition, seed=seed))


def write_reports(config_path="configs/ablation/ablation_v1.json"):
    config = load_ablation_config(config_path)
    rows = []
    for variant, spec in config["variants"].items():
        for condition in config["conditions"]:
            for seed in config["seeds"]:
                metrics, tables, run_id = _locate(config, variant, condition, seed)
                summary = json.loads((metrics / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
                test = summary["test_metrics"]
                history = pd.read_csv(metrics / f"{run_id}_round_history.csv")
                profile = _prediction_profile(tables / f"{run_id}_confusion_matrix.csv")[1]
                defense_path = metrics / f"{run_id}_defense_client_diagnostics.csv"
                traj_path = metrics / f"{run_id}_global_trajectory.npz"
                row = {"variant": variant, "label": spec["label"], "condition": condition, "seed": seed, "run_id": run_id,
                       "selected_round": summary["selected_round_by_validation_loss"],
                       "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
                       "test_weighted_f1": test["weighted_f1"],
                       "test_macro_recall": test["macro_recall_all_15_labels_zero_for_no_support"],
                       "test_macro_auroc": test["macro_auroc_ovr_defined_classes"],
                       "n_predicted_classes": profile["n_predicted_classes"],
                       "majority_predicted_share": profile["majority_predicted_share"],
                       "nan_validation_rounds": int(history["validation_loss"].isna().sum()),
                       "final_validation_macro_f1": float(history["validation_macro_f1"].iloc[-1]),
                       "final_validation_loss": float(history["validation_loss"].iloc[-1]),
                       "attack_detection_rate": None, "honest_false_positive_rate": None,
                       "honest_clipped_rate": None, "attacker_clipped_rate": None, "defense_ms_per_round": 0.0,
                       "rounds_with_negative_bn_running_var": None}
                if traj_path.exists():
                    row["rounds_with_negative_bn_running_var"] = int((_min_running_var(np.load(traj_path)["states"]) < 0).sum())
                if defense_path.exists():
                    d = pd.read_csv(defense_path)
                    rounds = pd.read_csv(metrics / f"{run_id}_defense_round_diagnostics.csv")
                    honest, attacker = d[~d["is_attacker"]], d[d["is_attacker"]]
                    row["honest_false_positive_rate"] = float(honest["flagged"].mean())
                    row["honest_clipped_rate"] = float(honest["clipped"].mean())
                    row["defense_ms_per_round"] = 1e3 * float(rounds["defense_seconds"].mean())
                    row["bn_variance_entries_floored"] = int(rounds["bn_variance_entries_floored"].sum())
                    if len(attacker):
                        filters = spec.get("components", {})
                        row["attack_detection_rate"] = (float(attacker["flagged"].mean())
                                                        if filters.get("cosine_filter") or filters.get("norm_filter") else None)
                        row["attacker_clipped_rate"] = float(attacker["clipped"].mean())
                        row["attacker_mean_weight"] = float(attacker["defended_weight"].mean())
                rows.append(row)
    runs = pd.DataFrame(rows)
    order = list(config["variants"])
    summary_rows = []
    for (variant, condition), group in runs.groupby(["variant", "condition"], sort=False):
        out = {"variant": variant, "label": config["variants"][variant]["label"], "condition": condition}
        for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_recall", "test_macro_auroc",
                       "attack_detection_rate", "honest_false_positive_rate", "nan_validation_rounds",
                       "rounds_with_negative_bn_running_var", "final_validation_macro_f1", "defense_ms_per_round",
                       "n_predicted_classes"]:
            values = pd.to_numeric(group[metric], errors="coerce")
            out[f"{metric}_mean"] = values.mean() if values.notna().any() else None
            out[f"{metric}_std"] = values.std(ddof=1) if values.notna().sum() > 1 else None
            out[f"{metric}_min"] = values.min() if values.notna().any() else None
            out[f"{metric}_max"] = values.max() if values.notna().any() else None
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)
    summary["variant"] = pd.Categorical(summary["variant"], order, ordered=True)
    summary = summary.sort_values(["condition", "variant"])
    root = Path(config["output_root"])
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    runs.to_csv(root / "tables" / "ablation_runs.csv", index=False)
    summary.to_csv(root / "tables" / "ablation_summary.csv", index=False)
    plots.plot_ablation(runs, order, {v: s["label"] for v, s in config["variants"].items()},
                        root / "figures" / "ablation_overview.png")
    return runs, summary
