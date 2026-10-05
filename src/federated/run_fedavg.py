"""Orchestrate the simulated FedAvg baseline: partition, train, validate, test once.

global model -> broadcast -> local client training -> collect client states
-> sample-weighted FedAvg -> new global model -> validation -> next round.

Validation is the only split used during training and for selecting the final
round (minimum validation loss, the same rule as centralized_cnn_v1). The test
split is opened once, after selection, by `evaluate_test_once`.
"""
import copy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.models.cnn1d import ECG1DCNN, count_trainable_parameters, model_architecture
from src.models.train_cnn import ECGSegmentDataset

from .config import load_federated_config
from .data import ClientECGDataset, load_class_to_index
from .evaluation import (communication_cost, compare_with_centralized, evaluate_test_once,
                         validation_metrics)
from .fedavg import (batchnorm_state_keys, client_seed, client_start_state, client_update, fedavg_aggregate,
                     get_model_state, select_clients, set_model_state)
from .partition import build_client_partition, client_distribution_tables
from .privacy import dp_fedavg_aggregate, noise_generator, validate_privacy_config
from . import plots


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def apply_overrides(config, overrides):
    """Merge `{section: {key: value}}` or `{key: value}` overrides into a copy of the config."""
    config = copy.deepcopy(config)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    return config


def prepare_partition(config, output_root=None, write=True):
    """Build the client partition and (optionally) write the distribution report."""
    part = config["partition"]
    class_to_index = load_class_to_index(config["data"]["label_config"])
    class_names = list(class_to_index)
    partition = build_client_partition(config["data"]["split_root"], int(part["num_clients"]), int(part["seed"]),
                                       part["strategy"], int(part.get("min_client_segments", 1)))
    summary, counts, proportions = client_distribution_tables(partition, class_names)
    if write:
        root = Path(output_root or config["output_root"])
        exp = config["experiment_id"]
        tables, metrics, figures = (root / d / "federated" for d in ("tables", "metrics", "figures"))
        for d in (tables, metrics, figures):
            d.mkdir(parents=True, exist_ok=True)
        _write_json(metrics / f"{exp}_client_partition.json", partition)
        _write_csv(tables / f"{exp}_client_summary.csv", summary)
        _write_csv(tables / f"{exp}_client_class_counts.csv", counts)
        _write_csv(tables / f"{exp}_client_class_proportions.csv", proportions)
        plots.plot_client_distribution(counts, class_names, figures / f"{exp}_client_distribution.png")
    return partition, summary, counts, proportions


def run_experiment(config_path="configs/federated/fedavg_v1.json", *, evaluate_test=True, output_root=None,
                   overrides=None, overwrite=False, log=print, comparison_prefix=""):
    config = apply_overrides(load_federated_config(config_path), overrides)
    privacy = config.get("privacy")
    accounting = None
    if privacy is not None:
        if config["aggregation"].get("batchnorm", "aggregate") != "aggregate":
            raise ValueError("Differential privacy is a separate one-factor experiment; do not combine with local BatchNorm")
        accounting = validate_privacy_config(privacy, config["training"]["rounds"], config["participation"]["fraction"])
    root = Path(output_root or config["output_root"])
    exp = config["experiment_id"]
    model_dir = root / "models" / "federated"
    metrics_dir, tables_dir, figures_dir = (root / d / "federated" for d in ("metrics", "tables", "figures"))
    model_path = model_dir / f"{exp}_global.pt"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"{model_path} exists; pass overwrite=True to replace this experiment")
    for d in (model_dir, metrics_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    seed = int(config["random_seed"])
    train_cfg, runtime = config["training"], config["runtime"]
    seed_everything(seed)
    torch.set_num_threads(int(runtime["cpu_threads"]))
    device = runtime["device"]
    eval_bs = int(runtime["evaluation_batch_size"])

    class_to_index = load_class_to_index(config["data"]["label_config"])
    class_names = list(class_to_index)
    partition, summary, _, _ = prepare_partition(config, root)
    split_root = Path(config["data"]["split_root"])
    client_data = [ClientECGDataset(split_root, c["records"], class_to_index) for c in partition["clients"]]
    for client, data in zip(partition["clients"], client_data):
        if len(data) != client["segments"]:
            raise RuntimeError(f"{client['client_id']}: loaded {len(data)} segments, expected {client['segments']}")
    validation = ECGSegmentDataset(split_root / "validation", class_to_index)
    log(f"Clients: {[len(d) for d in client_data]} segments; validation: {len(validation)}")

    # Same initialization procedure as centralized_cnn_v1: seed, then construct the model.
    seed_everything(seed)
    global_model = ECG1DCNN(num_classes=len(class_names), dropout=float(config["model"]["dropout"])).to(device)
    worker = copy.deepcopy(global_model)
    global_state = get_model_state(global_model)
    trainable = count_trainable_parameters(global_model)
    # aggregation.batchnorm: "aggregate" = standard FedAvg (default); "local" = FedBN-style, where each
    # client keeps its own BatchNorm entries across rounds. In both modes the evaluated global model is
    # the sample-weighted average of all client states; in "local" mode its BatchNorm part is used only
    # for evaluation and is never broadcast back to clients.
    bn_mode = config["aggregation"].get("batchnorm", "aggregate")
    bn_keys = batchnorm_state_keys(global_model) if bn_mode == "local" else []
    local_bn = {}
    client_validation = []
    dp_rounds, dp_clients = [], []

    history = [{"round": 0, "participants": 0, "client_train_loss_weighted": None,
                "client_train_accuracy_weighted": None, "elapsed_seconds": 0.0,
                **validation_metrics(global_model, validation, len(class_names), eval_bs, device)}]
    client_history = []
    best = {"round": 0, "loss": history[0]["validation_loss"], "state": global_state}
    participants_per_round = []
    started = time.perf_counter()
    num_clients = len(client_data)
    for round_index in range(1, int(train_cfg["rounds"]) + 1):
        selected = select_clients(num_clients, float(config["participation"]["fraction"]), seed, round_index)
        participants_per_round.append(len(selected))
        updates = []
        for k in selected:
            start_state = client_start_state(global_state, local_bn.get(k)) if bn_mode == "local" else global_state
            update = client_update(worker, start_state, client_data[k], local_epochs=int(train_cfg["local_epochs"]),
                                   batch_size=int(train_cfg["batch_size"]),
                                   learning_rate=float(train_cfg["learning_rate"]),
                                   optimizer_name=train_cfg["optimizer"], seed=client_seed(seed, round_index, k),
                                   device=device, num_workers=int(runtime["num_workers"]),
                                   shuffle=bool(train_cfg["shuffle_training"]))
            updates.append(update)
            if bn_mode == "local":
                local_bn[k] = {key: update["state"][key] for key in bn_keys}
            client_history.append({"round": round_index, "client_id": partition["clients"][k]["client_id"],
                                   "num_samples": update["num_samples"], "train_loss": update["train_loss"],
                                   "train_accuracy": update["train_accuracy"]})
        counts = [u["num_samples"] for u in updates]
        if privacy is None:
            global_state = fedavg_aggregate([u["state"] for u in updates], counts)
        else:
            global_state, dp = dp_fedavg_aggregate(
                global_state, [u["state"] for u in updates], counts, float(privacy["clipping_norm"]),
                float(privacy["noise_multiplier"]), noise_generator(seed, privacy["noise_seed_offset"], round_index))
            # Diagnostics below use unclipped norms: they are NOT covered by the DP guarantee and would not
            # be released in a deployment. They are kept only to interpret this simulation.
            dp_rounds.append({"round": round_index, "sensitivity": dp["sensitivity"], "noise_std": dp["noise_std"],
                              "noise_l2_norm": dp["noise_l2_norm"],
                              "aggregate_signal_l2_norm": dp["aggregate_signal_l2_norm"],
                              "noise_to_signal_ratio": dp["noise_l2_norm"] / max(dp["aggregate_signal_l2_norm"], 1e-12),
                              "clients_clipped": int(sum(dp["clipped"])), "median_update_norm": float(np.median(dp["update_norms"])),
                              "max_update_norm": float(max(dp["update_norms"])), "dimension": dp["dimension"]})
            for k, norm_value, was_clipped in zip(selected, dp["update_norms"], dp["clipped"]):
                dp_clients.append({"round": round_index, "client_id": partition["clients"][k]["client_id"],
                                   "update_l2_norm_unclipped": norm_value, "clipped": was_clipped})
        set_model_state(global_model, global_state)
        weights = np.asarray(counts, dtype=float) / sum(counts)
        row = {"round": round_index, "participants": len(selected),
               "client_train_loss_weighted": float(np.dot(weights, [u["train_loss"] for u in updates])),
               "client_train_accuracy_weighted": float(np.dot(weights, [u["train_accuracy"] for u in updates])),
               "elapsed_seconds": time.perf_counter() - started,
               **validation_metrics(global_model, validation, len(class_names), eval_bs, device)}
        history.append(row)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if bn_mode == "local":
            # Validation-only diagnostic: each client's personalised model (shared weights + own BatchNorm).
            for k in sorted(local_bn):
                set_model_state(worker, client_start_state(global_state, local_bn[k]))
                client_validation.append({"round": round_index, "client_id": partition["clients"][k]["client_id"],
                                          **validation_metrics(worker, validation, len(class_names), eval_bs, device)})
        if row["validation_loss"] < best["loss"]:
            best = {"round": round_index, "loss": row["validation_loss"], "state": global_state,
                    "local_bn": {k: dict(v) for k, v in local_bn.items()}}
    elapsed = time.perf_counter() - started

    set_model_state(global_model, best["state"])
    artifact = {"model_state_dict": best["state"], "architecture": model_architecture(), "input_shape": [2, 216],
                "class_to_index": class_to_index, "selected_round_by_validation_loss": best["round"],
                "federated_config": config, "partition": partition}
    if bn_mode == "local":
        artifact["batchnorm_mode"] = bn_mode
        artifact["client_batchnorm_states"] = {partition["clients"][k]["client_id"]: v
                                               for k, v in best.get("local_bn", {}).items()}
    torch.save(artifact, model_path)
    if client_validation:
        _write_csv(metrics_dir / f"{exp}_client_validation_history.csv", client_validation)
    if privacy is not None:
        _write_csv(metrics_dir / f"{exp}_dp_round_diagnostics.csv", dp_rounds)
        _write_csv(metrics_dir / f"{exp}_dp_client_update_norms.csv", dp_clients)
        _write_json(metrics_dir / f"{exp}_privacy_accounting.json",
                    {"privacy_config": privacy, "accounting": accounting, "training_seed": seed,
                     "partition_seed": config["partition"]["seed"],
                     "noise_seed_rule": "seed * 1000003 + noise_seed_offset + round",
                     "selection_note": "Round selection uses only the validation split, which is not client data; "
                                       "selecting among released models is post-processing."})
    _write_csv(metrics_dir / f"{exp}_round_history.csv", history)
    _write_csv(metrics_dir / f"{exp}_client_round_history.csv", client_history)
    comm = communication_cost(best["state"], participants_per_round, trainable)
    _write_json(metrics_dir / f"{exp}_communication_cost.json", comm)

    central_history = json.loads(Path("results/metrics/centralized_cnn_v1_history.json").read_text(encoding="utf-8"))
    central_epoch = central_history["history"][central_history["best_epoch_by_validation_loss"] - 1]
    reference = {"validation_accuracy": central_epoch["validation_accuracy"],
                 "validation_macro_f1": central_epoch["validation_macro_f1_all_labels"]}
    plots.plot_convergence(history, client_history, best["round"], reference, figures_dir / f"{exp}_convergence.png")
    plots.plot_client_losses(client_history, figures_dir / f"{exp}_client_losses.png")

    selected_row = history[best["round"]]
    run_summary = {
        "experiment_id": exp, "config": config, "trainable_parameters": trainable,
        "num_clients": num_clients, "client_segments": [len(d) for d in client_data],
        "validation_segments": len(validation), "rounds": int(train_cfg["rounds"]),
        "selected_round_by_validation_loss": best["round"], "selected_round_validation": selected_row,
        "final_round_validation": history[-1], "training_elapsed_seconds": elapsed,
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "model_path": str(model_path).replace("\\", "/"), "test_evaluated": False,
        "centralized_validation_reference_epoch1": reference,
    }
    if privacy is not None:
        run_summary["privacy"] = {"config": privacy, "accounting": accounting}
    if bn_mode == "local":
        run_summary["batchnorm_mode"] = bn_mode
        run_summary["batchnorm_state_keys"] = bn_keys
    if evaluate_test:
        result, test_summary = evaluate_test_once(global_model, split_root / "test", class_names, root, exp,
                                                  device, eval_bs, overwrite)
        plots.plot_confusion_matrix(result["confusion_matrix"], class_names,
                                    ("FedBN-style evaluation model" if bn_mode == "local"
                                     else "DP-FedAvg global model" if privacy is not None else "FedAvg global model")
                                    + f" (round {best['round']}) — held-out test confusion matrix",
                                    figures_dir / f"{exp}_confusion_matrix.png")
        aggregate_rows, per_class_rows = compare_with_centralized(result["aggregate"], result["per_class"], label=exp)
        _write_csv(tables_dir / f"{comparison_prefix}centralized_vs_fedavg_aggregate.csv", aggregate_rows)
        _write_csv(tables_dir / f"{comparison_prefix}centralized_vs_fedavg_per_class.csv", per_class_rows)
        run_summary.update({"test_evaluated": True, "test_metrics": test_summary["metrics"]})
    _write_json(metrics_dir / f"{exp}_run_summary.json", run_summary)
    return run_summary


if __name__ == "__main__":
    summary = run_experiment()
    print(json.dumps({k: summary[k] for k in ("selected_round_by_validation_loss", "test_evaluated")}, indent=2))
