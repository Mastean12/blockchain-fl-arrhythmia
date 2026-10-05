"""Validation monitoring, one-time test evaluation, and baseline comparison for FedAvg.

The test metrics reuse `calculate_metrics` and `ECGTestDataset` from the frozen
centralized evaluator, so the evaluation methodology is identical. Outputs are
written under `federated/` subdirectories with `fedavg_v1_*` names and never
touch the centralized baseline artifacts.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.evaluate_cnn import ECGTestDataset, calculate_metrics

MATCH_TOLERANCE = 0.01  # descriptive |delta| band for "matches"; not a statistical test


def predict(model, dataset, batch_size=512, device="cpu"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred, probs, loss_sum = [], [], [], 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            targets = torch.as_tensor(targets)
            logits = model(inputs.to(device))
            loss_sum += criterion(logits, targets.to(device)).item()
            y_true.append(targets.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(y_true).astype(np.int64)
    return y_true, np.concatenate(y_pred).astype(np.int64), np.concatenate(probs), loss_sum / len(y_true)


def validation_metrics(model, validation_dataset, num_classes, batch_size=512, device="cpu"):
    """Per-round validation metrics over all configured labels (zero_division=0)."""
    y_true, y_pred, _, loss = predict(model, validation_dataset, batch_size, device)
    labels = list(range(num_classes))
    return {
        "validation_loss": float(loss),
        "validation_accuracy": float(np.mean(y_true == y_pred)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "validation_macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "validation_macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def communication_cost(state, participants_per_round, trainable_parameters):
    """Analytical payload sizes for broadcasting/uploading the full model state.

    Assumptions: every participating client downloads the full global state and
    uploads its full local state once per round; tensors are sent uncompressed in
    their native dtype; protocol, encryption, and serialization overhead are
    excluded from `payload_bytes`. `torch_save_bytes` is the measured size of the
    state serialized in memory with `torch.save` (a local measurement, not a
    network measurement).
    """
    payload = int(sum(t.numel() * t.element_size() for t in state.values()))
    elements = int(sum(t.numel() for t in state.values()))
    buffer = io.BytesIO()
    torch.save(state, buffer)
    per_round = [{"round": r, "participants": k, "downlink_bytes": k * payload, "uplink_bytes": k * payload,
                  "total_bytes": 2 * k * payload} for r, k in enumerate(participants_per_round, start=1)]
    total = sum(row["total_bytes"] for row in per_round)
    return {
        "trainable_parameters": int(trainable_parameters),
        "state_dict_elements": elements,
        "state_dict_entries": len(state),
        "payload_bytes_per_model": payload,
        "torch_save_bytes_per_model": len(buffer.getvalue()),
        "rounds": len(participants_per_round),
        "participants_per_round": list(participants_per_round),
        "total_downlink_bytes": sum(r["downlink_bytes"] for r in per_round),
        "total_uplink_bytes": sum(r["uplink_bytes"] for r in per_round),
        "total_bytes": total,
        "total_megabytes": total / 1e6,
        "per_round": per_round,
        "assumptions": [
            "Full state_dict (weights, biases, BatchNorm running statistics and counters) sent each way.",
            "Each participating client receives its own copy of the global model (no multicast saving).",
            "Uncompressed native dtypes (float32 parameters/statistics, int64 counters).",
            "No protocol, transport, encryption, or serialization overhead in payload_bytes.",
            "Analytical calculation for a simulated federation; no network traffic was measured.",
        ],
    }


def evaluate_test_once(model, test_directory, class_names, output_root, experiment_id, device="cpu",
                       batch_size=512, overwrite=False):
    """Evaluate the finalized global model on the held-out test split and save outputs."""
    root = Path(output_root)
    metrics_dir, tables_dir = root / "metrics" / "federated", root / "tables" / "federated"
    metrics_path = metrics_dir / f"{experiment_id}_test_metrics.json"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; the test set is evaluated once per experiment")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    dataset = ECGTestDataset(test_directory, class_to_index)
    y_true, y_pred, probabilities, _ = predict(model, dataset, batch_size, device)
    result = calculate_metrics(y_true, y_pred, probabilities, class_names)

    cm = pd.DataFrame(result["confusion_matrix"], index=class_names, columns=class_names)
    cm.index.name = "true_label"
    cm.to_csv(tables_dir / f"{experiment_id}_confusion_matrix.csv")
    pd.DataFrame(result["per_class"]).to_csv(tables_dir / f"{experiment_id}_per_class_metrics.csv", index=False)
    (metrics_dir / f"{experiment_id}_classification_report.txt").write_text(
        result["classification_report_text"], encoding="utf-8")
    roc_rows = [{"label": f"{k}_aggregate", "auroc": v, "support": None, "defined": v is not None}
                for k, v in result["aggregate"].items() if "auroc" in k and "count" not in k]
    roc_rows += [{"label": r["label"], "auroc": r["auroc_ovr"], "support": r["support"],
                  "defined": r["auroc_defined"]} for r in result["per_class"]]
    pd.DataFrame(roc_rows).to_csv(metrics_dir / f"{experiment_id}_roc_metrics.csv", index=False)
    np.savez_compressed(metrics_dir / f"{experiment_id}_test_predictions.npz", y_true=y_true, y_pred=y_pred,
                        probabilities=probabilities, class_names=np.asarray(class_names, dtype=str),
                        record_ids=dataset.record_ids, patient_groups=dataset.patient_groups,
                        source_symbols=dataset.source_symbols, annotation_samples=dataset.annotation_samples)
    summary = {
        "experiment_id": experiment_id,
        "evaluation": "held-out test only, evaluated once after model selection on validation",
        "test_size": len(dataset),
        "test_record_ids": sorted(set(dataset.record_ids.tolist())),
        "class_names": class_names,
        "class_support": {name: int(result["support"][i]) for i, name in enumerate(class_names)},
        "metrics": result["aggregate"],
        "test_data_used_for_training_or_model_selection": False,
    }
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, summary


def _verdict(delta, higher_is_better=True):
    if delta is None:
        return "not comparable"
    if abs(delta) < MATCH_TOLERANCE:
        return "matches"
    return "improves" if (delta > 0) == higher_is_better else "degrades"


def compare_with_centralized(fedavg_aggregate, fedavg_per_class,
                             baseline_metrics_path="results/metrics/baseline_metrics.json",
                             baseline_per_class_path="results/tables/baseline_per_class_metrics.csv",
                             label="fedavg_v1"):
    """Read the frozen centralized artifacts (read-only) and build comparison tables."""
    central = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))["metrics"]
    keys = ["accuracy", "macro_precision", "macro_recall_all_15_labels_zero_for_no_support",
            "macro_sensitivity_supported_classes", "macro_f1", "weighted_precision", "weighted_recall",
            "weighted_f1", "macro_specificity_all_classes", "weighted_specificity",
            "macro_auroc_ovr_defined_classes", "weighted_auroc_ovr_defined_classes", "micro_auroc_ovr_flattened"]
    aggregate_rows = []
    for key in keys:
        c, f = central.get(key), fedavg_aggregate.get(key)
        delta = None if c is None or f is None else f - c
        aggregate_rows.append({"metric": key, "centralized_cnn_v1": c, label: f, "delta_fedavg_minus_central": delta,
                               "verdict": _verdict(delta)})
    central_pc = pd.read_csv(baseline_per_class_path, keep_default_na=False, na_values=[""])
    central_pc = central_pc.set_index("label")
    per_class_rows = []
    for row in fedavg_per_class:
        label = row["label"]
        c = central_pc.loc[label]
        support = int(row["support"])
        out = {"label": label, "test_support": support}
        for metric in ("precision", "sensitivity", "specificity", "f1_score", "auroc_ovr"):
            cv = None if pd.isna(c[metric]) else float(c[metric])
            fv = row[metric]
            out[f"central_{metric}"] = cv
            out[f"fedavg_{metric}"] = fv
        delta = out["fedavg_f1_score"] - out["central_f1_score"]
        out["delta_f1"] = delta
        out["f1_verdict"] = _verdict(delta) if support > 0 else "not evaluable (no test support)"
        per_class_rows.append(out)
    return aggregate_rows, per_class_rows
