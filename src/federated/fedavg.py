"""FedAvg building blocks: parameter exchange, client update, and weighted aggregation.

FedAvg (McMahan et al., 2017): for participating clients k with n_k local
training samples, the new global state is

    w_{t+1} = sum_k (n_k / sum_j n_j) * w_{t+1}^k

Every floating-point entry of the model `state_dict` (weights, biases, and the
BatchNorm running mean/variance buffers) is averaged this way. The integer
BatchNorm `num_batches_tracked` counters are averaged with the same weights and
rounded; they do not affect computation because BatchNorm uses a fixed momentum.
"""
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_model_state(model):
    """Detached CPU copy of the full state dict (what a client would transmit)."""
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())


def set_model_state(model, state):
    """Load a broadcast global state; shapes and keys must match exactly."""
    model.load_state_dict(state, strict=True)
    return model


def check_state_compatibility(reference, other):
    if list(reference) != list(other):
        raise ValueError("Client state keys differ from the global model")
    for key, value in reference.items():
        if other[key].shape != value.shape or other[key].dtype != value.dtype:
            raise ValueError(f"Client state for {key} has shape/dtype {tuple(other[key].shape)}/"
                             f"{other[key].dtype}, expected {tuple(value.shape)}/{value.dtype}")


def fedavg_aggregate(client_states, client_sample_counts):
    """Sample-count-weighted average of client state dicts (standard FedAvg)."""
    if not client_states or len(client_states) != len(client_sample_counts):
        raise ValueError("Need one sample count per client state")
    counts = np.asarray(client_sample_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("Client sample counts must be positive")
    weights = counts / counts.sum()
    reference = client_states[0]
    for state in client_states[1:]:
        check_state_compatibility(reference, state)
    aggregated = OrderedDict()
    for key, value in reference.items():
        stacked = torch.stack([state[key].to(torch.float64) for state in client_states])
        mean = torch.tensordot(torch.as_tensor(weights, dtype=torch.float64), stacked, dims=1)
        if value.is_floating_point():
            aggregated[key] = mean.to(value.dtype)
        else:
            aggregated[key] = torch.round(mean).to(value.dtype)
    return aggregated


def select_clients(num_clients, fraction, seed, round_index):
    """Indices of participating clients; all clients when fraction == 1."""
    m = max(1, int(round(fraction * num_clients)))
    if m >= num_clients:
        return list(range(num_clients))
    rng = np.random.default_rng([int(seed), int(round_index)])
    return sorted(int(i) for i in rng.choice(num_clients, size=m, replace=False))


def client_seed(seed, round_index, client_index):
    """Deterministic per-round, per-client seed for shuffling and dropout."""
    return int(seed) * 100_000 + int(round_index) * 1_000 + int(client_index)


def _optimizer(name, parameters, learning_rate):
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "AdamW":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    return torch.optim.SGD(parameters, lr=learning_rate)


def client_update(model, global_state, dataset, *, local_epochs, batch_size, learning_rate,
                  optimizer_name, seed, device="cpu", num_workers=0, shuffle=True):
    """Train a fresh copy of the global model on one client's local data.

    A new optimizer is created each round (optimizer state is not transmitted or
    persisted). Returns the updated state, the number of local samples, and the
    mean local training loss over all local epochs.
    """
    torch.manual_seed(seed)
    set_model_state(model, global_state)
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = _optimizer(optimizer_name, model.parameters(), learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, generator=generator)
    loss_sum, seen, correct = 0.0, 0, 0
    for _ in range(local_epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            n = len(targets)
            loss_sum += loss.item() * n
            seen += n
            correct += int((logits.argmax(dim=1) == targets).sum().item())
    return {"state": get_model_state(model), "num_samples": len(dataset),
            "train_loss": loss_sum / seen, "train_accuracy": correct / seen}


def batchnorm_state_keys(model):
    """State-dict keys owned by BatchNorm layers (affine weight/bias, running stats, counter)."""
    bn_prefixes = [name for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    return [key for key in model.state_dict() if any(key.startswith(prefix + ".") for prefix in bn_prefixes)]


def client_start_state(global_state, local_state=None):
    """State a client starts a round from.

    Standard FedAvg (`local_state=None`): the broadcast global state. FedBN-style:
    the global non-BatchNorm entries with the client's own persisted BatchNorm
    entries substituted. Key order is kept identical to the global state.
    """
    if not local_state:
        return global_state
    unknown = set(local_state) - set(global_state)
    if unknown:
        raise ValueError(f"Local state has keys not in the global model: {sorted(unknown)}")
    return OrderedDict((key, local_state.get(key, value)) for key, value in global_state.items())
