"""Day 17 malicious-client study: run each predeclared attack x seed once, then report vulnerability.

Each run uses the unchanged fedavg_v1 configuration; only `experiment_id`,
`random_seed`, `partition.seed` and an `attack` section are overridden. Outputs go
to results/security/. The clean comparator is the Day 11 natural FedAvg run with
the same seed. Global-model deviation is measured against the clean per-round
global states archived by the bit-identical Day 16 recorded runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attacks import flat_float_state
from .run_fedavg import run_experiment
from . import plots

CLASSES = ["/", "A", "E", "F", "J", "L", "N", "Q", "R", "S", "V", "a", "e", "f", "j"]


def load_security_config(path="configs/security/attacks_v1.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be unique")
    return config


def planned_attack_runs(config):
    return [(config["run_id_template"].format(attack=name, seed=seed), name, int(seed))
            for name in config["attacks"] for seed in config["seeds"]]


def attack_overrides(config, run_id, name, seed):
    attack = {"enabled": True, "name": name, "selection_seed_offset": config["selection_seed_offset"],
              **config["attacks"][name]}
    attack.pop("description", None)
    return {"experiment_id": run_id, "random_seed": seed,
            "partition": {"seed": seed, "strategy": config["partition_strategy"]}, "attack": attack}


def run_all(config_path="configs/security/attacks_v1.json", log=print):
    config = load_security_config(config_path)
    root = Path(config["output_root"])
    summaries = {}
    for run_id, name, seed in planned_attack_runs(config):
        path = root / "metrics" / "federated" / f"{run_id}_run_summary.json"
        if path.exists():
            log(f"{run_id}: already complete, loading saved summary")
            summaries[run_id] = json.loads(path.read_text(encoding="utf-8"))
            continue
        log(f"{run_id}: training")
        summaries[run_id] = run_experiment(config["base_config"], evaluate_test=True, output_root=root,
                                           overrides=attack_overrides(config, run_id, name, seed),
                                           comparison_prefix=f"{run_id}_", log=lambda *a: None)
    return summaries


def _prediction_profile(confusion_csv):
    cm = pd.read_csv(confusion_csv, index_col=0)
    counts = {c: int(cm[c].sum()) for c in CLASSES}
    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values() if v > 0])
    top = max(counts, key=counts.get)
    return counts, {"n_predicted_classes": int(sum(v > 0 for v in counts.values())),
                    "share_N_or_V": (counts["N"] + counts["V"]) / total,
                    "majority_predicted_class": top, "majority_predicted_share": counts[top] / total,
                    "prediction_entropy_bits": float(-(probs * np.log2(probs)).sum())}


def _running_var_mask():
    """Positions of BatchNorm running-variance entries in the flat floating-point state vector."""
    from src.models.cnn1d import ECG1DCNN
    mask = []
    for key, value in ECG1DCNN().state_dict().items():
        if value.is_floating_point():
            mask.extend([key.endswith("running_var")] * value.numel())
    return np.asarray(mask)


def _min_running_var(trajectory):
    """Minimum BatchNorm running variance of each round's global model (negative => NaN outputs)."""
    return trajectory[:, _running_var_mask()].min(axis=1)


def _clean_trajectory(template, seed, rounds):
    directory = Path(template.format(seed=seed))
    if not directory.exists():
        return None
    return np.stack([flat_float_state(torch.load(directory / f"round_{r:03d}.pt", map_location="cpu",
                                                 weights_only=False)["global_state"]) for r in range(rounds + 1)])


def write_security_reports(config_path="configs/security/attacks_v1.json"):
    config = load_security_config(config_path)
    root = Path(config["output_root"])
    comp = config["comparator"]
    base_runs = pd.read_csv(comp["runs_table"]).set_index("run_id")
    rows, deviation_rows, per_class_rows = [], [], []
    clean_seen = set()
    for run_id, name, seed in planned_attack_runs(config):
        metrics = root / "metrics" / "federated"
        tables = root / "tables" / "federated"
        summary = json.loads((metrics / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        test = summary["test_metrics"]
        base_id = comp["run_id_template"].format(seed=seed)
        base = base_runs.loc[base_id]
        counts, profile = _prediction_profile(tables / f"{run_id}_confusion_matrix.csv")
        base_counts, base_profile = _prediction_profile(Path(comp["tables_dir"]) / f"{base_id}_confusion_matrix.csv")
        diag = pd.read_csv(metrics / f"{run_id}_attack_round_diagnostics.csv")
        history = pd.read_csv(metrics / f"{run_id}_round_history.csv")
        attacked = np.load(metrics / f"{run_id}_global_trajectory.npz")["states"]
        clean = _clean_trajectory(comp["clean_trajectory_archive"], seed, attacked.shape[0] - 1)
        selected = summary["selected_round_by_validation_loss"]
        final_dev = selected_dev = rel_final = None
        if clean is not None:
            with np.errstate(invalid="ignore", over="ignore"):
                distance = np.linalg.norm(attacked - clean, axis=1)
                reference = np.linalg.norm(clean, axis=1)
            for r, (d, ref) in enumerate(zip(distance, reference)):
                deviation_rows.append({"attack": name, "seed": seed, "round": r, "l2_deviation_from_clean": d,
                                       "relative_deviation": d / ref, "clean_global_norm": ref,
                                       "validation_loss": history.loc[r, "validation_loss"],
                                       "validation_macro_f1": history.loc[r, "validation_macro_f1"]})
            final_dev, selected_dev, rel_final = float(distance[-1]), float(distance[selected]), float(distance[-1] / reference[-1])
        for row in pd.read_csv(tables / f"{run_id}_per_class_metrics.csv", keep_default_na=False, na_values=[""]).to_dict("records"):
            if row["support"] > 0:
                per_class_rows.append({"attack": name, "seed": seed, **row})
        if seed not in clean_seen:
            clean_seen.add(seed)
            for row in pd.read_csv(Path(comp["tables_dir"]) / f"{base_id}_per_class_metrics.csv",
                                   keep_default_na=False, na_values=[""]).to_dict("records"):
                if row["support"] > 0:
                    per_class_rows.append({"attack": "clean", "seed": seed, **row})
        rows.append({
            "attack": name, "seed": seed, "run_id": run_id, "clean_run": base_id,
            "attacker_client": summary["attack"]["attacker_client_id"],
            "attacker_weight": float(diag["attacker_weight"].iloc[0]),
            "selected_round": selected, "clean_selected_round": int(base["selected_round"]),
            "non_finite_global_rounds": summary["attack"]["non_finite_global_rounds"],
            "first_non_finite_round": int(diag.loc[~diag["global_state_finite"], "round"].min())
            if (~diag["global_state_finite"]).any() else None,
            "test_accuracy": test["accuracy"], "clean_test_accuracy": float(base["test_accuracy"]),
            "test_macro_f1": test["macro_f1"], "clean_test_macro_f1": float(base["test_macro_f1"]),
            "test_weighted_f1": test["weighted_f1"], "clean_test_weighted_f1": float(base["test_weighted_f1"]),
            "test_macro_recall": test["macro_recall_all_15_labels_zero_for_no_support"],
            "clean_test_macro_recall": float(base["test_macro_recall"]),
            "test_macro_auroc": test["macro_auroc_ovr_defined_classes"],
            "clean_test_macro_auroc": float(base["test_macro_auroc"]),
            **{k: v for k, v in profile.items()},
            **{f"clean_{k}": v for k, v in base_profile.items()},
            "predicted_counts": json.dumps({c: v for c, v in counts.items() if v}),
            "nan_validation_rounds": int(history["validation_loss"].isna().sum()),
            "first_nan_validation_round": int(history.loc[history["validation_loss"].isna(), "round"].min())
            if history["validation_loss"].isna().any() else None,
            "rounds_with_negative_bn_running_var": int((_min_running_var(attacked) < 0).sum()),
            "first_negative_bn_running_var_round": int(np.argmax(_min_running_var(attacked) < 0))
            if (_min_running_var(attacked) < 0).any() else None,
            "min_validation_loss": float(history["validation_loss"].min()),
            "final_validation_loss": float(history["validation_loss"].iloc[-1]),
            "final_validation_macro_f1": float(history["validation_macro_f1"].iloc[-1]),
            "honest_update_norm_mean": float(diag["honest_update_norm"].mean()),
            "malicious_update_norm_mean": float(diag["malicious_update_norm"].mean()),
            "cosine_malicious_vs_honest_mean": float(diag["cosine_malicious_vs_honest"].mean()),
            "deviation_final_round": final_dev, "deviation_selected_round": selected_dev,
            "relative_deviation_final_round": rel_final,
        })
    runs = pd.DataFrame(rows)
    for metric in ("accuracy", "macro_f1", "weighted_f1", "macro_recall", "macro_auroc"):
        runs[f"delta_{metric}"] = runs[f"test_{metric}"] - runs[f"clean_test_{metric}"]
    summary_rows = []
    for name, group in runs.groupby("attack", sort=False):
        for metric in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_recall", "test_macro_auroc",
                       "delta_accuracy", "delta_macro_f1", "delta_weighted_f1", "delta_macro_recall",
                       "n_predicted_classes", "majority_predicted_share", "selected_round", "non_finite_global_rounds",
                       "relative_deviation_final_round"]:
            values = group[metric].astype(float)
            summary_rows.append({"attack": name, "metric": metric, "mean": values.mean(),
                                 "std_sample": values.std(ddof=1), "min": values.min(), "max": values.max(),
                                 "n_runs": len(values)})
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "attack_runs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "attack_summary.csv", index=False)
    deviation = pd.DataFrame(deviation_rows)
    deviation.to_csv(out / "attack_global_deviation_by_round.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "attack_per_class.csv", index=False)
    plots.plot_attack_test_metrics(runs, root / "figures" / "attack_test_metrics.png")
    if not deviation.empty:
        plots.plot_attack_deviation(deviation, root / "figures" / "attack_global_deviation.png")
    return runs, pd.DataFrame(summary_rows), deviation
