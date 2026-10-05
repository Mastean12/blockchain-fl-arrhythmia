"""Load and validate the federated experiment configuration."""
import json
from pathlib import Path

SUPPORTED_STRATEGIES = {"random_group_equal_count", "controlled_group_partition"}


def load_federated_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for section in ("model", "data", "partition", "training", "participation", "aggregation",
                    "model_selection", "runtime"):
        if section not in config:
            raise ValueError(f"Federated config missing section '{section}'")
    part, train = config["partition"], config["training"]
    if part["strategy"] not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported partition strategy {part['strategy']!r}")
    if int(part["num_clients"]) < 1:
        raise ValueError("num_clients must be at least 1")
    for key in ("rounds", "local_epochs", "batch_size"):
        if int(train[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if float(train["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if train["optimizer"] not in {"Adam", "AdamW", "SGD"}:
        raise ValueError("Supported optimizers: Adam, AdamW, SGD")
    if train["loss"] != "CrossEntropyLoss":
        raise ValueError("Only CrossEntropyLoss is supported")
    if not 0 < float(config["participation"]["fraction"]) <= 1:
        raise ValueError("participation.fraction must be in (0, 1]")
    if config["aggregation"]["method"] != "FedAvg":
        raise ValueError("Only FedAvg aggregation is implemented")
    if config["aggregation"]["weighting"] != "client_training_samples":
        raise ValueError("FedAvg weighting must be 'client_training_samples'")
    if config["aggregation"].get("batchnorm", "aggregate") not in {"aggregate", "local"}:
        raise ValueError("aggregation.batchnorm must be 'aggregate' (FedAvg) or 'local' (FedBN-style)")
    if config["model_selection"]["split"] != "validation":
        raise ValueError("Model selection may only use the validation split")
    return config
