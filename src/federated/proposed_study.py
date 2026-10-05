"""Day 18 evaluation of the proposed defense: 4 conditions (clean + 3 Day 17 attacks) x 3 seeds.

Each run is the unchanged fedavg_v1 configuration plus the frozen `defense` section
(configs/proposed/robust_defense_v1.json) and, for attacked conditions, the exact
Day 17 attack section. Comparators are read-only: clean FedAvg (Day 11) and
attacked FedAvg without defense (Day 17).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_fedavg import run_experiment
from .security import (_clean_trajectory, _min_running_var, _prediction_profile, attack_overrides,
                       load_security_config)
from . import plots


def load_proposed_config(path="configs/proposed/robust_defense_v1.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def planned_runs(config):
    return [(config["run_id_template"].format(condition=c, seed=s), c, int(s))
            for c in config["conditions"] for s in config["seeds"]]


def run_overrides(config, attack_config, run_id, condition, seed):
    overrides = {"experiment_id": run_id, "random_seed": seed,
                 "partition": {"seed": seed, "strategy": config["partition_strategy"]},
                 "defense": dict(config["defense"])}
    if condition != "clean":
        overrides["attack"] = attack_overrides(attack_config, run_id, condition, seed)["attack"]
    return overrides


def run_all(config_path="configs/proposed/robust_defense_v1.json", log=print):
    config = load_proposed_config(config_path)
    attack_config = load_security_config(config["attack_config"])
    root = Path(config["output_root"])
    out = {}
    for run_id, condition, seed in planned_runs(config):
        path = root / "metrics" / "federated" / f"{run_id}_run_summary.json"
        if path.exists():
            log(f"{run_id}: already complete, loading")
            out[run_id] = json.loads(path.read_text(encoding="utf-8"))
            continue
        log(f"{run_id}: training")
        out[run_id] = run_experiment(config["base_config"], evaluate_test=True, output_root=root,
                                     overrides=run_overrides(config, attack_config, run_id, condition, seed),
                                     comparison_prefix=f"{run_id}_", log=lambda *a: None)
    return out


def _metrics(test):
    return {"accuracy": test["accuracy"], "macro_f1": test["macro_f1"], "weighted_f1": test["weighted_f1"],
            "macro_recall": test["macro_recall_all_15_labels_zero_for_no_support"],
            "macro_auroc": test["macro_auroc_ovr_defined_classes"]}


def write_reports(config_path="configs/proposed/robust_defense_v1.json"):
    config = load_proposed_config(config_path)
    attack_config = load_security_config(config["attack_config"])
    comp = attack_config["comparator"]
    root = Path(config["output_root"])
    sec_root = Path(attack_config["output_root"])
    rows, weights_rows, deviation_rows = [], [], []
    for run_id, condition, seed in planned_runs(config):
        metrics, tables = root / "metrics" / "federated", root / "tables" / "federated"
        summary = json.loads((metrics / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        history = pd.read_csv(metrics / f"{run_id}_round_history.csv")
        clients = pd.read_csv(metrics / f"{run_id}_defense_client_diagnostics.csv")
        rounds = pd.read_csv(metrics / f"{run_id}_defense_round_diagnostics.csv")
        trajectory = np.load(metrics / f"{run_id}_global_trajectory.npz")["states"]
        counts, profile = _prediction_profile(tables / f"{run_id}_confusion_matrix.csv")
        clean_id = comp["run_id_template"].format(seed=seed)
        clean_summary = json.loads((Path(comp["metrics_dir"]) / f"{clean_id}_run_summary.json").read_text(encoding="utf-8"))
        clean_profile = _prediction_profile(Path(comp["tables_dir"]) / f"{clean_id}_confusion_matrix.csv")[1]
        undefended = None
        if condition != "clean":
            undef_id = attack_config["run_id_template"].format(attack=condition, seed=seed)
            undefended = json.loads((sec_root / "metrics/federated" / f"{undef_id}_run_summary.json").read_text(encoding="utf-8"))
            undef_profile = _prediction_profile(sec_root / "tables/federated" / f"{undef_id}_confusion_matrix.csv")[1]
        attacker_rows = clients[clients["is_attacker"]]
        honest_rows = clients[~clients["is_attacker"]]
        clean_traj = _clean_trajectory(comp["clean_trajectory_archive"], seed, trajectory.shape[0] - 1)
        dev_final = None
        if clean_traj is not None:
            distance = np.linalg.norm(trajectory - clean_traj, axis=1)
            reference = np.linalg.norm(clean_traj, axis=1)
            dev_final = float(distance[-1] / reference[-1])
            for r in range(len(distance)):
                deviation_rows.append({"condition": condition, "seed": seed, "round": r,
                                       "relative_deviation": float(distance[r] / reference[r])})
        for _, row in clients.iterrows():
            weights_rows.append({"condition": condition, "seed": seed, **row.to_dict()})
        min_var = _min_running_var(trajectory)
        row = {"condition": condition, "seed": seed, "run_id": run_id,
               "attacker_client": summary.get("attack", {}).get("attacker_client_id"),
               "selected_round": summary["selected_round_by_validation_loss"],
               **{f"defended_{k}": v for k, v in _metrics(summary["test_metrics"]).items()},
               **{f"clean_{k}": v for k, v in _metrics(clean_summary["test_metrics"]).items()},
               **({f"undefended_{k}": v for k, v in _metrics(undefended["test_metrics"]).items()} if undefended else {}),
               "defended_n_predicted_classes": profile["n_predicted_classes"],
               "defended_majority_predicted_share": profile["majority_predicted_share"],
               "defended_predicted_counts": json.dumps({c: v for c, v in counts.items() if v}),
               "clean_n_predicted_classes": clean_profile["n_predicted_classes"],
               "undefended_n_predicted_classes": undef_profile["n_predicted_classes"] if undefended else None,
               "undefended_selected_round": undefended["selected_round_by_validation_loss"] if undefended else None,
               "attack_detection_rate": float(attacker_rows["flagged"].mean()) if len(attacker_rows) else None,
               "attacker_mean_defended_weight": float(attacker_rows["defended_weight"].mean()) if len(attacker_rows) else None,
               "attacker_mean_prior_weight": float(attacker_rows["prior_weight"].mean()) if len(attacker_rows) else None,
               "honest_false_positive_rate": float(honest_rows["flagged"].mean()),
               "honest_clipped_rate": float(honest_rows["clipped"].mean()),
               "honest_cosine_min": float(honest_rows["cosine_to_median"].min()),
               "honest_norm_ratio_max": float(honest_rows["norm_ratio_to_median"].max()),
               "fallback_rounds": int(rounds["fallback_median"].sum()),
               "bn_variance_entries_floored": int(rounds["bn_variance_entries_floored"].sum()),
               "nan_validation_rounds": int(history["validation_loss"].isna().sum()),
               "rounds_with_negative_bn_running_var": int((min_var < 0).sum()),
               "final_validation_loss": float(history["validation_loss"].iloc[-1]),
               "final_validation_macro_f1": float(history["validation_macro_f1"].iloc[-1]),
               "final_validation_accuracy": float(history["validation_accuracy"].iloc[-1]),
               "relative_deviation_final_round": dev_final,
               "defense_ms_per_round": 1e3 * float(rounds["defense_seconds"].mean()),
               "defense_seconds_total": float(rounds["defense_seconds"].sum()),
               "training_seconds": summary["training_elapsed_seconds"],
               "extra_communication_bytes": 0}
        if undefended is not None:
            undef_hist = pd.read_csv(sec_root / "metrics/federated" /
                                     f"{attack_config['run_id_template'].format(attack=condition, seed=seed)}_round_history.csv")
            row["undefended_final_validation_loss"] = float(undef_hist["validation_loss"].iloc[-1])
            row["undefended_final_validation_macro_f1"] = float(undef_hist["validation_macro_f1"].iloc[-1])
            row["undefended_nan_validation_rounds"] = int(undef_hist["validation_loss"].isna().sum())
        rows.append(row)
    runs = pd.DataFrame(rows)
    summary_rows = []
    metric_cols = [c for c in runs.columns if c.startswith(("defended_", "clean_", "undefended_")) and
                   pd.api.types.is_numeric_dtype(runs[c])] + [
        "attack_detection_rate", "honest_false_positive_rate", "attacker_mean_defended_weight",
        "nan_validation_rounds", "final_validation_macro_f1", "relative_deviation_final_round", "defense_ms_per_round"]
    for condition, group in runs.groupby("condition", sort=False):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.notna().any():
                summary_rows.append({"condition": condition, "metric": metric, "mean": values.mean(),
                                     "std_sample": values.std(ddof=1), "min": values.min(), "max": values.max(),
                                     "n_runs": int(values.notna().sum())})
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "proposed_runs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "proposed_summary.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(out / "proposed_client_weights.csv", index=False)
    deviation = pd.DataFrame(deviation_rows)
    deviation.to_csv(out / "proposed_global_deviation_by_round.csv", index=False)
    plots.plot_defense_comparison(runs, root / "figures" / "proposed_vs_attacks_test_metrics.png")
    plots.plot_defense_weights(pd.DataFrame(weights_rows), root / "figures" / "proposed_client_weights.png")
    return runs, pd.DataFrame(summary_rows)
