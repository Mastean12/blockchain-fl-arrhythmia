"""Client-local datasets built only from training shards assigned to that client."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_FIELDS = {"segments", "labels", "record_ids", "patient_groups", "source_symbols",
                   "annotation_samples", "starts", "ends", "sampling_frequency_hz", "channel_names"}


def load_class_to_index(label_config_path):
    """Same sorted-symbol class order as the centralized baseline."""
    config = json.loads(Path(label_config_path).read_text(encoding="utf-8"))
    classes = sorted(set(config["labels"]["beat_symbols"].values()))
    return {label: index for index, label in enumerate(classes)}


class ClientECGDataset(Dataset):
    """In-memory segments for one simulated client.

    Shards are opened only from `<split_root>/train/`; a record ID that has no
    training shard raises instead of falling back to another split.
    """

    def __init__(self, split_root, record_ids, class_to_index, expected_length=216, expected_channels=2):
        train_dir = Path(split_root) / "train"
        if not record_ids:
            raise ValueError("A client needs at least one record")
        features, targets = [], []
        for record_id in sorted(record_ids):
            path = train_dir / f"{record_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Record {record_id} is not in the training split ({path})")
            with np.load(path, allow_pickle=False) as stored:
                missing = REQUIRED_FIELDS - set(stored.files)
                if missing:
                    raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
                x = stored["segments"]
                labels = stored["labels"].astype(str)
                if x.shape != (len(labels), expected_length, expected_channels):
                    raise ValueError(f"{path.name}: invalid segment shape {x.shape}")
                if not np.isfinite(x).all():
                    raise ValueError(f"{path.name}: NaN/Inf in ECG input")
                if set(map(str, stored["record_ids"])) != {record_id}:
                    raise ValueError(f"{path.name}: record provenance mismatch")
                if not np.array_equal(labels, stored["source_symbols"].astype(str)):
                    raise ValueError(f"{path.name}: labels disagree with source symbols")
                if set(labels) - set(class_to_index):
                    raise ValueError(f"{path.name}: labels not defined by model mapping")
                features.append(x)
                targets.append(np.fromiter((class_to_index[s] for s in labels), dtype=np.int64, count=len(labels)))
        self.record_ids = sorted(record_ids)
        # Channels-first, as in the centralized ECGSegmentDataset.
        self.features = np.ascontiguousarray(np.concatenate(features).transpose(0, 2, 1), dtype=np.float32)
        self.targets = np.concatenate(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index]), torch.tensor(self.targets[index], dtype=torch.long)
