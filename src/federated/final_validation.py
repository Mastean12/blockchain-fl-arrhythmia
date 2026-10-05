"""Day 20 independent confirmation: Variant 5 defense vs FedAvg controls on fresh seeds.

Both arms use the frozen Day 18 thresholds; the control runs the defense in observe
mode, which aggregates with plain FedAvg (bit-identical) while logging would-be
screening statistics and the global trajectory. Nothing here is tuned.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .proposed_study import load_proposed_config
from .run_fedavg import run_experiment
from .security import _min_running_var, _prediction_profile, attack_overrides, load_security_config

USED_SEEDS = {42, 123, 2024, 7}


def load_final_config(path="configs/final/final_validation_v1.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config["seeds"]) & USED_SEEDS:
        raise ValueError("Final confirmation seeds must be fresh")
    return config


def planned_runs(config):
    return [(config["run_id_template"].format(arm=a, condition=c, seed=s), a, c, int(s))
            for a in config["arms"] for c in config["conditions"] for s in config["seeds"]]


def final_overrides(config, run_id, arm, condition, seed):
    frozen = load_proposed_config(config["frozen_defense_config"])
    attacks = load_security_config(config["attack_config"])
    defense = dict(frozen["defense"])
    spec = config["arms"][arm]
    defense["mode"] = spec["defense_mode"]
    defense["name"] = f"{frozen['defense']['name']}__{arm}"
    if "components" in spec:
        defense["components"] = dict(spec["components"])
    overrides = {"experiment_id": run_id, "random_seed": seed,
                 "partition": {"seed": seed, "strategy": frozen["partition_strategy"]}, "defense": defense}
    if condition != "clean":
        overrides["attack"] = attack_overrides(attacks, run_id, condition, seed)["attack"]
    return overrides


def run_all(config_path="configs/final/final_validation_v1.json", log=print):
    config = load_final_config(config_path)
    base = load_proposed_config(config["frozen_defense_config"])["base_config"]
    root = Path(config["output_root"])
    out = {}
    for run_id, arm, condition, seed in planned_runs(config):
        path = root / "metrics" / "federated" / f"{run_id}_run_summary.json"
        if path.exists():
            log(f"{run_id}: already complete, loading")
            out[run_id] = json.loads(path.read_text(encoding="utf-8"))
            continue
        log(f"{run_id}: training")
        out[run_id] = run_experiment(base, evaluate_test=True, output_root=root,
                                     overrides=final_overrides(config, run_id, arm, condition, seed),
                                     comparison_prefix=f"{run_id}_", log=lambda *a: None)
    return out


def collect(config_path="configs/final/final_validation_v1.json"):
    config = load_final_config(config_path)
    root = Path(config["output_root"])
    metrics, tables = root / "metrics" / "federated", root / "tables" / "federated"
    trajectories, rows = {}, []
    for run_id, arm, condition, seed in planned_runs(config):
        trajectories[(arm, condition, seed)] = np.load(metrics / f"{run_id}_global_trajectory.npz")["states"]
    for run_id, arm, condition, seed in planned_runs(config):
        summary = json.loads((metrics / f"{run_id}_run_summary.json").read_text(encoding="utf-8"))
        test = summary["test_metrics"]
        history = pd.read_csv(metrics / f"{run_id}_round_history.csv")
        clients = pd.read_csv(metrics / f"{run_id}_defense_client_diagnostics.csv")
        rounds = pd.read_csv(metrics / f"{run_id}_defense_round_diagnostics.csv")
        profile = _prediction_profile(tables / f"{run_id}_confusion_matrix.csv")[1]
        traj = trajectories[(arm, condition, seed)]
        clean = trajectories[("fedavg_control", "clean", seed)]
        distance = np.linalg.norm(traj - clean, axis=1)
        honest, attacker = clients[~clients["is_attacker"]], clients[clients["is_attacker"]]
        selected = summary["selected_round_by_validation_loss"]
        rows.append({
            "arm": arm, "label": config["arms"][arm]["label"], "condition": condition, "seed": seed, "run_id": run_id,
            "attacker_client": summary.get("attack", {}).get("attacker_client_id"),
            "attacker_prior_weight": float(attacker["prior_weight"].mean()) if len(attacker) else None,
            "selected_round": selected,
            "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "test_weighted_f1": test["weighted_f1"],
            "test_macro_recall": test["macro_recall_all_15_labels_zero_for_no_support"],
            "test_macro_auroc": test["macro_auroc_ovr_defined_classes"],
            "n_predicted_classes": profile["n_predicted_classes"],
            "majority_predicted_share": profile["majority_predicted_share"],
            "final_round_validation_loss": float(history["validation_loss"].iloc[-1]),
            "final_round_validation_accuracy": float(history["validation_accuracy"].iloc[-1]),
            "final_round_validation_macro_f1": float(history["validation_macro_f1"].iloc[-1]),
            "nan_validation_rounds": int(history["validation_loss"].isna().sum()),
            "rounds_with_negative_bn_running_var": int((_min_running_var(traj) < 0).sum()),
            # control: would-be flags (aggregation unaffected); variant 5: applied flags
            "attack_detection_rate": float(attacker["flagged"].mean()) if len(attacker) else None,
            "honest_false_positive_rate": float(honest["flagged"].mean()),
            "honest_clipped_rate": float(honest["clipped"].mean()) if arm == "variant5" else None,
            "relative_deviation_from_clean_fedavg_selected": float(distance[selected] / np.linalg.norm(clean[selected])),
            "relative_deviation_from_clean_fedavg_final": float(distance[-1] / np.linalg.norm(clean[-1])),
            "defense_ms_per_round": 1e3 * float(rounds["defense_seconds"].mean()),
            "training_seconds": summary["training_elapsed_seconds"],
        })
    return pd.DataFrame(rows)
