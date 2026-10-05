"""Proposed robust aggregation for FedAvg (Day 18): clipping + similarity/norm screening + robust BatchNorm stats.

Per round, given the broadcast global state w_t and client states w_k with sample
counts n_k (prior weights p_k = n_k / sum n):

1. Parameter updates  Delta_k = w_k - w_t over *trainable parameters* only
   (conv/linear weights and biases, BatchNorm affine weight/bias).
2. L2 clipping        Delta_k <- Delta_k * min(1, C / ||Delta_k||).
3. Similarity         cos_k = cos(Delta_k, m), where m is the coordinate-wise median
                      of all client parameter updates (robust reference direction).
4. Screening          norm ratio r_k = ||Delta_k|| (before clipping) / median_j ||Delta_j||.
                      A client is flagged if cos_k < tau_cos or r_k > tau_norm; flagged
                      clients get weight 0 and the prior weights p_k are renormalised
                      over accepted clients.
5. Aggregation        w_{t+1} = w_t + sum_k w_k * clip(Delta_k) on trainable parameters.
6. BatchNorm stats    running mean/var are aggregated separately as the coordinate-wise
                      median of the *accepted* clients' reported statistics, and running
                      variances are floored at `bn_variance_floor`. Integer counters are
                      aggregated as in FedAvg (public function of sample counts).

If no client is accepted, the round falls back to the coordinate-wise median of the
clipped updates (logged). Mode "observe" computes and logs every statistic but returns
plain FedAvg, which is used only for calibration on a non-evaluation seed.

The defense runs entirely on the server, adds no client-side computation and no
communication, and uses only quantities the server already receives in FedAvg.
"""
from collections import OrderedDict
import time

import numpy as np
import torch
from torch import nn

from .fedavg import fedavg_aggregate

REQUIRED = {"enabled", "name", "mode", "clipping_norm", "cosine_threshold", "norm_ratio_threshold",
            "bn_variance_floor", "reference_direction", "bn_running_stats"}


def validate_defense_config(defense):
    missing = REQUIRED - set(defense)
    if missing:
        raise ValueError(f"defense config missing {sorted(missing)}")
    if defense["enabled"] is not True:
        raise ValueError("defense.enabled must be true when a defense section is present")
    if defense["mode"] not in {"defend", "observe"}:
        raise ValueError("defense.mode must be 'defend' or 'observe'")
    if not float(defense["clipping_norm"]) > 0 or not float(defense["norm_ratio_threshold"]) > 1:
        raise ValueError("clipping_norm must be > 0 and norm_ratio_threshold > 1")
    if not -1 <= float(defense["cosine_threshold"]) < 1:
        raise ValueError("cosine_threshold must be in [-1, 1)")
    if not float(defense["bn_variance_floor"]) > 0:
        raise ValueError("bn_variance_floor must be positive")
    if defense["reference_direction"] != "coordinate_wise_median":
        raise ValueError("Only the coordinate-wise median reference direction is implemented")
    if defense["bn_running_stats"] != "coordinate_wise_median_of_accepted":
        raise ValueError("Only median-of-accepted BatchNorm statistics are implemented")
    return defense


def partition_keys(model):
    """(trainable parameter keys, BatchNorm running-statistic keys, integer buffer keys) in state order."""
    params = {name for name, p in model.named_parameters() if p.requires_grad}
    bn_modules = [name for name, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    trainable, running, integer = [], [], []
    for key, value in model.state_dict().items():
        if key in params:
            trainable.append(key)
        elif not value.is_floating_point():
            integer.append(key)
        elif any(key == f"{m}.running_mean" or key == f"{m}.running_var" for m in bn_modules):
            running.append(key)
        else:
            raise ValueError(f"Unclassified state entry {key}")
    return trainable, running, integer


def _flat(state, keys):
    return torch.cat([state[k].reshape(-1).to(torch.float64) for k in keys])


def _cos(a, b):
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def robust_aggregate(global_state, client_states, sample_counts, defense, keys, client_ids=None):
    """Return (new_global_state, per_client_rows, round_summary)."""
    trainable, running, integer = keys
    counts = np.asarray(sample_counts, dtype=np.float64)
    if not client_states or len(client_states) != len(counts) or np.any(counts <= 0):
        raise ValueError("Need one positive sample count per client state")
    t0 = time.perf_counter()
    prior = counts / counts.sum()
    reference = _flat(global_state, trainable)
    updates = torch.stack([_flat(s, trainable) - reference for s in client_states])
    norms = torch.linalg.vector_norm(updates, dim=1)
    C = float(defense["clipping_norm"])
    factors = torch.clamp(C / torch.clamp(norms, min=1e-12), max=1.0)
    clipped = updates * factors[:, None]
    median_direction = updates.median(dim=0).values
    cosines = np.array([_cos(u, median_direction) for u in updates])
    median_norm = float(norms.median())
    ratios = (norms / max(median_norm, 1e-12)).numpy()
    low_cos = cosines < float(defense["cosine_threshold"])
    high_norm = ratios > float(defense["norm_ratio_threshold"])
    flagged = low_cos | high_norm
    accepted = ~flagged
    fallback = not accepted.any()
    if fallback:
        weights = np.zeros_like(prior)
        param_update = clipped.median(dim=0).values
    else:
        weights = np.where(accepted, prior, 0.0)
        weights = weights / weights.sum()
        param_update = torch.tensordot(torch.as_tensor(weights, dtype=torch.float64), clipped, dims=1)

    new_params = reference + param_update
    stats_from = [s for s, ok in zip(client_states, accepted) if ok] if not fallback else client_states
    floor = float(defense["bn_variance_floor"])
    out, offset, floored = OrderedDict(), 0, 0
    fedavg_ints = fedavg_aggregate(client_states, sample_counts) if integer else {}
    for key, value in global_state.items():
        if key in trainable:
            n = value.numel()
            out[key] = new_params[offset:offset + n].reshape(value.shape).to(value.dtype)
            offset += n
        elif key in running:
            stacked = torch.stack([s[key].to(torch.float64) for s in stats_from])
            stat = stacked.median(dim=0).values
            if key.endswith("running_var"):
                floored += int((stat < floor).sum())
                stat = stat.clamp_min(floor)
            out[key] = stat.to(value.dtype)
        else:
            out[key] = fedavg_ints[key].clone()
    elapsed = time.perf_counter() - t0

    if defense["mode"] == "observe":
        out = fedavg_aggregate(client_states, sample_counts)
    ids = client_ids or [f"client_{i}" for i in range(len(client_states))]
    rows = [{"client_id": ids[i], "prior_weight": float(prior[i]), "defended_weight": float(weights[i]),
             "update_norm": float(norms[i]), "norm_ratio_to_median": float(ratios[i]),
             "clip_factor": float(factors[i]), "clipped": bool(norms[i] > C),
             "cosine_to_median": float(cosines[i]), "flag_low_cosine": bool(low_cos[i]),
             "flag_high_norm": bool(high_norm[i]), "flagged": bool(flagged[i])}
            for i in range(len(client_states))]
    summary = {"defense_seconds": elapsed, "accepted_clients": int(accepted.sum()), "fallback_median": fallback,
               "median_update_norm": median_norm, "bn_variance_entries_floored": floored,
               "aggregated_param_update_norm": float(torch.linalg.vector_norm(param_update))}
    return out, rows, summary
