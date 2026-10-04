"""Evaluate a saved centralized CNN using only one explicit test directory."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset

from .cnn1d import ECG1DCNN, count_trainable_parameters


class ECGTestDataset(Dataset):
    """Load per-record shards from the supplied held-out test directory only."""

    REQUIRED = {"segments", "labels", "source_symbols", "record_ids", "patient_groups",
                "annotation_samples", "starts", "ends", "sampling_frequency_hz", "channel_names"}

    def __init__(self, test_directory, class_to_index, segment_length=216, channels=2):
        root = Path(test_directory)
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No test record shards found in {root}")
        xs, ys, record_ids, groups, symbols, ann_samples = [], [], [], [], [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED - set(data.files)
                if missing:
                    raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
                x = data["segments"]
                labels = data["labels"].astype(str)
                n = len(labels)
                if x.shape != (n, segment_length, channels):
                    raise ValueError(f"{path.name}: incorrect ECG shape {x.shape}")
                if not np.isfinite(x).all():
                    raise ValueError(f"{path.name}: NaN/Inf in input")
                if set(map(str, data["record_ids"])) != {path.stem}:
                    raise ValueError(f"{path.name}: record ID metadata mismatch")
                for key in ("source_symbols", "patient_groups", "annotation_samples", "starts", "ends"):
                    if len(data[key]) != n:
                        raise ValueError(f"{path.name}: metadata length mismatch for {key}")
                source = data["source_symbols"].astype(str)
                if not np.array_equal(labels, source):
                    raise ValueError(f"{path.name}: label/source annotation mismatch")
                if set(labels) - set(class_to_index):
                    raise ValueError(f"{path.name}: unknown target label")
                xs.append(x)
                ys.append(np.fromiter((class_to_index[label] for label in labels), dtype=np.int64, count=n))
                record_ids.extend(map(str, data["record_ids"]))
                groups.extend(map(str, data["patient_groups"]))
                symbols.extend(source.tolist())
                ann_samples.extend(data["annotation_samples"].astype(np.int64).tolist())
        self.features = np.ascontiguousarray(np.concatenate(xs).transpose(0, 2, 1), dtype=np.float32)
        self.targets = np.concatenate(ys)
        self.record_ids = np.asarray(record_ids, dtype=str)
        self.patient_groups = np.asarray(groups, dtype=str)
        self.source_symbols = np.asarray(symbols, dtype=str)
        self.annotation_samples = np.asarray(ann_samples, dtype=np.int64)
        self.record_shard_count = len(paths)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index]), int(self.targets[index])


def calculate_metrics(y_true, y_pred, probabilities, class_names):
    """Calculate fixed-label classification metrics and one-vs-rest specificity/AUROC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be nonempty one-dimensional arrays of equal length")
    if probabilities.shape != (len(y_true), len(class_names)):
        raise ValueError("probabilities must have shape [samples, classes]")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to 1")
    if np.any(y_true < 0) or np.any(y_true >= len(class_names)) or np.any(y_pred < 0) or np.any(y_pred >= len(class_names)):
        raise ValueError("class index outside configured range")
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    support = cm.sum(axis=1).astype(int)
    predicted_support = cm.sum(axis=0).astype(int)
    tp = np.diag(cm).astype(int)
    fn = support - tp
    fp = predicted_support - tp
    tn = total - tp - fn - fp

    sensitivity = np.divide(tp, tp + fn, out=np.full(len(labels), np.nan), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.full(len(labels), np.nan), where=(tn + fp) > 0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                        digits=4, zero_division=0)
    one_hot = np.eye(len(class_names), dtype=np.uint8)[y_true]
    per_class_auc = {}
    roc_curves = {}
    for idx, name in enumerate(class_names):
        positives = int(support[idx])
        if positives == 0 or positives == total:
            per_class_auc[name] = None
            continue
        fpr, tpr, _ = roc_curve(one_hot[:, idx], probabilities[:, idx])
        per_class_auc[name] = float(auc(fpr, tpr))
        roc_curves[name] = (fpr, tpr)
    valid_auc_names = list(roc_curves)
    if not valid_auc_names:
        macro_auc = weighted_auc = None
    else:
        auc_values = np.asarray([per_class_auc[name] for name in valid_auc_names])
        auc_weights = np.asarray([support[class_names.index(name)] for name in valid_auc_names], dtype=float)
        macro_auc = float(auc_values.mean())
        weighted_auc = float(np.average(auc_values, weights=auc_weights))
    micro_auc = float(roc_auc_score(one_hot.ravel(), probabilities.ravel())) if one_hot.size else None

    supported_sens = sensitivity[~np.isnan(sensitivity)]
    supported_weights = support[~np.isnan(sensitivity)]
    report_rows = []
    for name in class_names:
        item = report[name]
        idx = class_names.index(name)
        report_rows.append({"label": name, "precision": float(item["precision"]),
                            "recall": float(item["recall"]), "sensitivity": None if np.isnan(sensitivity[idx]) else float(sensitivity[idx]),
                            "specificity": None if np.isnan(specificity[idx]) else float(specificity[idx]),
                            "f1_score": float(item["f1-score"]), "support": int(item["support"]),
                            "auroc_ovr": per_class_auc[name], "auroc_defined": per_class_auc[name] is not None})
    aggregate = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall_all_15_labels_zero_for_no_support": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_sensitivity_supported_classes": float(supported_sens.mean()) if len(supported_sens) else None,
        "weighted_sensitivity": float(np.average(sensitivity[~np.isnan(sensitivity)], weights=supported_weights)) if len(supported_sens) else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "macro_specificity_all_classes": float(np.nanmean(specificity)),
        "weighted_specificity": float(np.average(specificity, weights=support)) if support.sum() else None,
        "macro_auroc_ovr_defined_classes": macro_auc,
        "weighted_auroc_ovr_defined_classes": weighted_auc,
        "micro_auroc_ovr_flattened": micro_auc,
        "auroc_defined_class_count": len(valid_auc_names),
        "class_count": len(class_names),
    }
    report_text += "\nSensitivity for classes with zero test support is undefined (shown as blank in the detailed CSV).\n"
    report_text += "Specificity uses one-vs-rest TN/(TN+FP). AUROC is undefined for classes with no positives.\n"
    return {"confusion_matrix": cm, "per_class": report_rows, "aggregate": aggregate,
            "classification_report": report, "classification_report_text": report_text,
            "roc_curves": roc_curves, "support": support}


def _save_outputs(result, dataset, y_true, y_pred, probabilities, class_names, output_root):
    root = Path(output_root)
    metrics_dir, figures_dir, tables_dir = root / "metrics", root / "figures", root / "tables"
    for directory in (metrics_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cm = result["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "true_label"
    cm_df.to_csv(tables_dir / "baseline_confusion_matrix.csv")
    pd.DataFrame(result["per_class"]).to_csv(metrics_dir / "baseline_classification_report.csv", index=False)
    pd.DataFrame(result["per_class"]).to_csv(tables_dir / "baseline_per_class_metrics.csv", index=False)
    (metrics_dir / "baseline_classification_report.txt").write_text(result["classification_report_text"], encoding="utf-8")

    roc_rows = []
    for label, value in result["aggregate"].items():
        if "auroc" in label and "count" not in label:
            roc_rows.append({"label": f"{label}_aggregate", "auroc": value, "support": None,
                             "defined": value is not None})
    for row in result["per_class"]:
        roc_rows.append({"label": row["label"], "auroc": row["auroc_ovr"],
                         "support": row["support"], "defined": row["auroc_defined"]})
    pd.DataFrame(roc_rows).to_csv(metrics_dir / "baseline_roc_metrics.csv", index=False)

    matrix_fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    matrix_fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of segments")
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, xlabel="Predicted label",
           ylabel="True label", title="Centralized 1D CNN — held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            ax.text(col_idx, row_idx, f"{cm[row_idx, col_idx]:,}", ha="center", va="center",
                    color="white" if cm[row_idx, col_idx] > threshold else "black", fontsize=7)
    matrix_fig.savefig(figures_dir / "baseline_confusion_matrix.png", dpi=300)
    plt.close(matrix_fig)

    roc_fig, roc_ax = plt.subplots(figsize=(10, 8), layout="constrained")
    # Per-class AUROCs are stored in rows and used to label each curve.
    auc_by_label = {row["label"]: row["auroc_ovr"] for row in result["per_class"]}
    for label, (fpr, tpr) in result["roc_curves"].items():
        roc_ax.plot(fpr, tpr, linewidth=1.5, label=f"{label} (AUC={auc_by_label[label]:.3f})")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
               title="One-vs-rest ROC — held-out test set")
    roc_ax.legend(loc="lower right", fontsize=8, ncol=2)
    roc_fig.savefig(figures_dir / "baseline_roc_curve.png", dpi=300)
    plt.close(roc_fig)

    summary = {
        "evaluation": "held-out test only",
        "test_size": len(dataset),
        "test_record_shards": dataset.record_shard_count,
        "test_record_ids": sorted(set(dataset.record_ids.tolist())),
        "class_names": class_names,
        "class_support": {name: int(result["support"][i]) for i, name in enumerate(class_names)},
        "confusion_matrix_labels_order": class_names,
        "metrics": result["aggregate"],
        "checkpoint_best_validation_loss_epoch": None,
        "test_data_used_for_training_or_model_selection": False,
    }
    metadata_path = Path(root) / "models" / "centralized_cnn_v1.pt"
    summary["checkpoint_path"] = str(metadata_path)
    (metrics_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(metrics_dir / "baseline_test_predictions.npz",
                        y_true=y_true, y_pred=y_pred, probabilities=probabilities,
                        class_names=np.asarray(class_names, dtype=str),
                        record_ids=dataset.record_ids, patient_groups=dataset.patient_groups,
                        source_symbols=dataset.source_symbols, annotation_samples=dataset.annotation_samples)
    return summary


def evaluate_checkpoint(checkpoint_path="results/models/centralized_cnn_v1.pt",
                        test_directory="data/splits/mitbih_v1/test",
                        label_config_path="configs/preprocessing/mitbih_v1.json",
                        output_root="results", batch_size=512, device="cpu"):
    """Load a frozen Day 5 checkpoint and evaluate every sample in one test directory."""
    label_cfg = json.loads(Path(label_config_path).read_text(encoding="utf-8"))
    class_names = sorted(set(label_cfg["labels"]["beat_symbols"].values()))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["class_to_index"] != class_to_index:
        raise ValueError("Checkpoint class mapping differs from configured source labels")
    if checkpoint["input_shape"] != [2, 216]:
        raise ValueError(f"Unexpected checkpoint input shape: {checkpoint['input_shape']}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = ECG1DCNN(num_classes=len(class_names), dropout=checkpoint["training_config"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = ECGTestDataset(test_directory, class_to_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_true, all_pred, all_prob = [], [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_true.extend(targets.numpy().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_prob.append(probs)
    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    probabilities = np.concatenate(all_prob, axis=0)
    result = calculate_metrics(y_true, y_pred, probabilities, class_names)
    summary = _save_outputs(result, dataset, y_true, y_pred, probabilities, class_names, output_root)
    summary["checkpoint_best_validation_loss_epoch"] = checkpoint["best_epoch_by_validation_loss"]
    metrics_path = Path(output_root) / "metrics" / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["per_class"] = result["per_class"]
    summary["confusion_matrix"] = result["confusion_matrix"].tolist()
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/models/centralized_cnn_v1.pt")
    parser.add_argument("--test-dir", default="data/splits/mitbih_v1/test")
    parser.add_argument("--label-config", default="configs/preprocessing/mitbih_v1.json")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.test_dir, args.label_config, args.output_root, args.batch_size)
    print(json.dumps(result["metrics"], indent=2))
