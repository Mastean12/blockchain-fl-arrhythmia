"""Centralized 1D CNN training using only explicit train/validation directories."""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.storage import load_record
from .cnn1d import ECG1DCNN, count_trainable_parameters, model_architecture


class ECGSegmentDataset(Dataset):
    """In-memory dataset assembled from per-record NPZ shards in one split."""

    def __init__(self, split_directory, class_to_index, expected_length=216, expected_channels=2):
        self.directory = Path(split_directory)
        paths = sorted(self.directory.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No record shards found in {self.directory}")
        features, targets = [], []
        self.record_ids = []
        class_counts = Counter()
        for path in paths:
            with np.load(path, allow_pickle=False) as stored:
                required = {"segments", "labels", "record_ids", "patient_groups", "source_symbols",
                            "annotation_samples", "starts", "ends", "sampling_frequency_hz", "channel_names"}
                missing = required - set(stored.files)
                if missing:
                    raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
                x = stored["segments"]
                labels = stored["labels"].astype(str)
                if x.shape != (len(labels), expected_length, expected_channels):
                    raise ValueError(f"{path.name}: invalid segment shape {x.shape}")
                if not np.isfinite(x).all():
                    raise ValueError(f"{path.name}: NaN/Inf in ECG input")
                if set(map(str, stored["record_ids"])) != {path.stem}:
                    raise ValueError(f"{path.name}: record provenance mismatch")
                if len(stored["patient_groups"]) != len(labels):
                    raise ValueError(f"{path.name}: group metadata length mismatch")
                if set(labels) - set(class_to_index):
                    raise ValueError(f"{path.name}: labels not defined by model mapping")
                for key in ("source_symbols", "patient_groups", "annotation_samples", "starts", "ends"):
                    if len(stored[key]) != len(labels):
                        raise ValueError(f"{path.name}: metadata length mismatch for {key}")
                if not np.array_equal(labels, stored["source_symbols"].astype(str)):
                    raise ValueError(f"{path.name}: source labels disagree with identity label mapping")
                features.append(x)
                targets.append(np.fromiter((class_to_index[s] for s in labels), dtype=np.int64, count=len(labels)))
                self.record_ids.append(path.stem)
                class_counts.update(labels.tolist())
        x = np.concatenate(features, axis=0)
        # PyTorch Conv1d expects [batch, channels, samples]. Store this once
        # contiguously rather than copying each segment inside __getitem__.
        self.features = np.ascontiguousarray(x.transpose(0, 2, 1), dtype=np.float32)
        self.targets = np.concatenate(targets)
        self.class_counts = dict(sorted(class_counts.items()))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index]), torch.tensor(self.targets[index], dtype=torch.long)


def load_training_configuration(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = {"random_seed", "batch_size", "learning_rate", "optimizer", "loss", "epochs",
                "num_workers", "cpu_threads", "dropout", "device", "shuffle_training"}
    if required - set(config):
        raise ValueError(f"Training config missing {sorted(required-set(config))}")
    if config["batch_size"] < 1 or config["epochs"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    if config["optimizer"] not in {"Adam", "AdamW", "SGD"}:
        raise ValueError("Supported optimizers: Adam, AdamW, SGD")
    if config["loss"] != "CrossEntropyLoss":
        raise ValueError("Only CrossEntropyLoss is currently supported")
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch environment has no available CUDA device")
    return config


def load_train_validation_datasets(split_root, label_config_path):
    """Load only `train/` and `validation/`; this function never opens `test/`."""
    labels_config = json.loads(Path(label_config_path).read_text(encoding="utf-8"))
    classes = sorted(set(labels_config["labels"]["beat_symbols"].values()))
    class_to_index = {label: i for i, label in enumerate(classes)}
    root = Path(split_root)
    train = ECGSegmentDataset(root / "train", class_to_index)
    validation = ECGSegmentDataset(root / "validation", class_to_index)
    return train, validation, class_to_index


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _optimizer(name, parameters, learning_rate):
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "AdamW":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    return torch.optim.SGD(parameters, lr=learning_rate)


def _evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_n = len(targets)
            total_loss += loss.item() * batch_n
            total += batch_n
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))
    return {"loss": total_loss / total, "accuracy": accuracy, "macro_f1_all_labels": macro_f1}


def train_model(train_dataset, validation_dataset, class_to_index, config, output_root="results"):
    """Train with training batches; validation is used for epoch monitoring only."""
    _seed_everything(int(config["random_seed"]))
    torch.set_num_threads(int(config["cpu_threads"]))
    device = torch.device(config["device"])
    model = ECG1DCNN(num_classes=len(class_to_index), dropout=float(config["dropout"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _optimizer(config["optimizer"], model.parameters(), float(config["learning_rate"]))
    generator = torch.Generator().manual_seed(int(config["random_seed"]))
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]),
                              shuffle=bool(config["shuffle_training"]), num_workers=int(config["num_workers"]),
                              generator=generator)
    val_loader = DataLoader(validation_dataset, batch_size=int(config["batch_size"]),
                            shuffle=False, num_workers=int(config["num_workers"]))

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            n = len(targets)
            train_loss_sum += loss.item() * n
            train_total += n
            train_correct += int((logits.argmax(dim=1) == targets).sum().item())
        validation = _evaluate(model, val_loader, criterion, device, len(class_to_index))
        row = {"epoch": epoch, "train_loss": train_loss_sum / train_total,
               "train_accuracy": train_correct / train_total,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "validation_macro_f1_all_labels": validation["macro_f1_all_labels"]}
        history.append(row)
        print(json.dumps(row))
        if validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)

    root = Path(output_root)
    model_dir, metrics_dir = root / "models", root / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "centralized_cnn_v1.pt"
    history_path = metrics_dir / "centralized_cnn_v1_history.json"
    csv_path = metrics_dir / "centralized_cnn_v1_history.csv"
    artifact = {
        "model_state_dict": best_state,
        "architecture": model_architecture(),
        "input_shape": [2, 216],
        "class_to_index": class_to_index,
        "best_epoch_by_validation_loss": best_epoch,
        "training_config": config,
    }
    torch.save(artifact, model_path)
    result = {
        "training_config": config,
        "input_shape_channel_first": [2, 216],
        "number_of_classes": len(class_to_index),
        "class_to_index": class_to_index,
        "train_segments": len(train_dataset),
        "validation_segments": len(validation_dataset),
        "train_record_ids": train_dataset.record_ids,
        "validation_record_ids": validation_dataset.record_ids,
        "train_class_distribution": train_dataset.class_counts,
        "validation_class_distribution": validation_dataset.class_counts,
        "architecture": model_architecture(),
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "history": history,
        "model_path": str(model_path),
    }
    history_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return model, result


def run_from_config(training_config_path="configs/training/centralized_cnn_v1.json",
                    split_root="data/splits/mitbih_v1",
                    label_config_path="configs/preprocessing/mitbih_v1.json", output_root="results"):
    config = load_training_configuration(training_config_path)
    train, validation, class_to_index = load_train_validation_datasets(split_root, label_config_path)
    return train_model(train, validation, class_to_index, config, output_root)


if __name__ == "__main__":
    _, result = run_from_config()
    print(json.dumps({key: result[key] for key in ("train_segments", "validation_segments", "trainable_parameters", "best_epoch_by_validation_loss", "model_path")}, indent=2))
