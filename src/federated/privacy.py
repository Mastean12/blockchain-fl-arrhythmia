"""Client-level differential privacy for FedAvg (DP-FedAvg style) and its privacy accountant.

Mechanism (one round, trusted server; McMahan et al., 2018, fixed-weight variant)
-------------------------------------------------------------------------------
For participating clients k with public, fixed aggregation weights p_k = n_k / sum_j n_j:

    delta_k   = w_k - w_t                              (all floating-point state entries)
    clipped_k = delta_k * min(1, C / ||delta_k||_2)    (per-client L2 clipping)
    w_{t+1}   = w_t + sum_k p_k * clipped_k + N(0, (z * C * max_k p_k)^2 I)

Adding or removing one client's contribution (other weights held fixed) changes
sum_k p_k * clipped_k by at most S = C * max_k p_k in L2 norm, so each round is a
Gaussian mechanism with sensitivity S and noise standard deviation z * S, i.e.
noise multiplier z. Integer BatchNorm counters are aggregated exactly as in
FedAvg; they depend only on the public sample counts. Clamping BatchNorm
running variances to be non-negative after noise is post-processing.

Accountant (full participation, q = 1)
--------------------------------------
Each round is a Gaussian mechanism with noise multiplier z, i.e. (1/z)-GDP, and
T adaptive rounds compose exactly to mu-GDP with mu = sqrt(T) / z (Dong, Roth &
Su, 2019). The tight (epsilon, delta) curve of mu-GDP is (Balle & Wang, 2018):

    delta(eps) = Phi(-eps/mu + mu/2) - exp(eps) * Phi(-eps/mu - mu/2).

As a cross-check, the RDP bound (Mironov, 2017) is reported:
eps_RDP = min_alpha [T * alpha / (2 z^2) + log(1/delta) / (alpha - 1)], which is
an upper bound and is never smaller than the GDP value. Client subsampling
(q < 1) is NOT supported by this accountant and is rejected.
"""
from collections import OrderedDict
import math

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm

# --------------------------------------------------------------------------- accountant


def gdp_mu(rounds, noise_multiplier, sampling_rate=1.0):
    if sampling_rate != 1.0:
        raise ValueError("This accountant supports only full participation (client sampling rate 1.0)")
    if rounds < 1 or noise_multiplier <= 0:
        raise ValueError("rounds must be >= 1 and noise_multiplier > 0")
    return math.sqrt(rounds) / noise_multiplier


def gdp_delta(epsilon, mu):
    """Tight delta(epsilon) of mu-GDP (Balle & Wang 2018, Theorem 8), evaluated in log space."""
    a = norm.cdf(-epsilon / mu + mu / 2)
    log_b = epsilon + norm.logcdf(-epsilon / mu - mu / 2)
    return float(a - math.exp(log_b))


def gdp_epsilon(delta, mu):
    """Smallest epsilon with delta(epsilon) <= delta for mu-GDP."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if gdp_delta(0.0, mu) <= delta:
        return 0.0
    upper = 1.0
    while gdp_delta(upper, mu) > delta:
        upper *= 2
        if upper > 1e4:
            raise ValueError("epsilon is unbounded for these parameters")
    return float(brentq(lambda e: gdp_delta(e, mu) - delta, 0.0, upper, xtol=1e-12))


def rdp_epsilon(rounds, noise_multiplier, delta, orders=None):
    """RDP upper bound for T compositions of the Gaussian mechanism (Mironov 2017, Prop. 3)."""
    orders = np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10.1, 512, 5000)]) if orders is None else orders
    values = rounds * orders / (2 * noise_multiplier ** 2) + math.log(1 / delta) / (orders - 1)
    i = int(np.argmin(values))
    return float(values[i]), float(orders[i])


def account(rounds, noise_multiplier, delta, sampling_rate=1.0):
    mu = gdp_mu(rounds, noise_multiplier, sampling_rate)
    eps_rdp, order = rdp_epsilon(rounds, noise_multiplier, delta)
    return {"accountant": "GDP composition (Dong, Roth & Su 2019) + analytic Gaussian conversion (Balle & Wang 2018)",
            "mu_gdp": mu, "epsilon": gdp_epsilon(delta, mu), "delta": delta, "rounds": rounds,
            "noise_multiplier": noise_multiplier, "client_sampling_rate": sampling_rate,
            "epsilon_rdp_upper_bound": eps_rdp, "rdp_optimal_order": order}


def noise_multiplier_for_epsilon(target_epsilon, delta, rounds):
    """Smallest noise multiplier whose GDP epsilon does not exceed the target."""
    return float(brentq(lambda z: gdp_epsilon(delta, gdp_mu(rounds, z)) - target_epsilon, 0.5, 1e3, xtol=1e-10))

# --------------------------------------------------------------------------- configuration


def validate_privacy_config(privacy, training_rounds, participation_fraction):
    required = {"enabled", "privacy_unit", "clipping_norm", "noise_multiplier", "delta", "accountant",
                "client_sampling_rate", "rounds", "noise_seed_offset"}
    missing = required - set(privacy)
    if missing:
        raise ValueError(f"privacy config missing {sorted(missing)}")
    if privacy["enabled"] is not True:
        raise ValueError("privacy.enabled must be true when a privacy section is present")
    if privacy["privacy_unit"] != "client":
        raise ValueError("Only client-level privacy (privacy_unit='client') is implemented")
    if not float(privacy["clipping_norm"]) > 0 or not float(privacy["noise_multiplier"]) > 0:
        raise ValueError("clipping_norm and noise_multiplier must be positive")
    if not 0 < float(privacy["delta"]) < 1:
        raise ValueError("delta must be in (0, 1)")
    if privacy["accountant"] != "gdp_full_participation":
        raise ValueError("Only the 'gdp_full_participation' accountant is implemented")
    if float(privacy["client_sampling_rate"]) != 1.0 or float(participation_fraction) != 1.0:
        raise ValueError("The accountant requires full participation: client_sampling_rate and "
                         "participation.fraction must both be 1.0")
    if int(privacy["rounds"]) != int(training_rounds):
        raise ValueError("privacy.rounds must equal training.rounds (every released round is accounted)")
    result = account(int(privacy["rounds"]), float(privacy["noise_multiplier"]), float(privacy["delta"]))
    target = privacy.get("target_epsilon")
    if target is not None and result["epsilon"] > float(target) + 1e-9:
        raise ValueError(f"Configured mechanism gives epsilon={result['epsilon']:.4f} > target {target}")
    return result

# --------------------------------------------------------------------------- mechanism


def clip_update(update, clipping_norm):
    """Scale a flat update so its L2 norm is at most `clipping_norm`; returns (clipped, original_norm)."""
    norm_value = float(torch.linalg.vector_norm(update))
    factor = min(1.0, clipping_norm / norm_value) if norm_value > 0 else 1.0
    return update * factor, norm_value


def _float_keys(state):
    return [key for key, value in state.items() if value.is_floating_point()]


def flatten(state, keys):
    return torch.cat([state[key].reshape(-1).to(torch.float64) for key in keys])


def dp_fedavg_aggregate(global_state, client_states, client_sample_counts, clipping_norm, noise_multiplier,
                        generator):
    """One round of client-level DP aggregation (see module docstring)."""
    if not client_states or len(client_states) != len(client_sample_counts):
        raise ValueError("Need one sample count per client state")
    counts = np.asarray(client_sample_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("Client sample counts must be positive")
    weights = counts / counts.sum()
    keys = _float_keys(global_state)
    reference = flatten(global_state, keys)
    aggregate = torch.zeros_like(reference)
    norms, clipped = [], []
    for weight, state in zip(weights, client_states):
        if list(state) != list(global_state):
            raise ValueError("Client state keys differ from the global model")
        update, norm_value = clip_update(flatten(state, keys) - reference, clipping_norm)
        aggregate += float(weight) * update
        norms.append(norm_value)
        clipped.append(norm_value > clipping_norm)
    sensitivity = clipping_norm * float(weights.max())
    noise_std = noise_multiplier * sensitivity
    noise = torch.normal(0.0, noise_std, size=reference.shape, generator=generator, dtype=torch.float64)
    new_flat = reference + aggregate + noise
    new_state, offset = OrderedDict(), 0
    for key, value in global_state.items():
        if value.is_floating_point():
            n = value.numel()
            tensor = new_flat[offset:offset + n].reshape(value.shape).to(value.dtype)
            if key.endswith("running_var"):
                tensor = tensor.clamp_min(0.0)  # post-processing: variances must be non-negative
            new_state[key] = tensor
            offset += n
        else:  # integer BatchNorm counters: public function of sample counts, aggregated as in FedAvg
            stacked = torch.stack([s[key].to(torch.float64) for s in client_states])
            mean = torch.tensordot(torch.as_tensor(weights, dtype=torch.float64), stacked, dims=1)
            new_state[key] = torch.round(mean).to(value.dtype)
    diagnostics = {"sensitivity": sensitivity, "noise_std": noise_std,
                   "noise_l2_norm": float(torch.linalg.vector_norm(noise)),
                   "aggregate_signal_l2_norm": float(torch.linalg.vector_norm(aggregate)),
                   "update_norms": norms, "clipped": clipped, "dimension": int(reference.numel())}
    return new_state, diagnostics


def noise_generator(seed, noise_seed_offset, round_index):
    """Dedicated, reproducible noise stream per round (independent of training RNG).

    A seeded PRNG makes experiments reproducible; a real deployment must draw noise
    from a cryptographically secure source, and the guarantee assumes ideal Gaussian
    sampling (floating-point effects are not modelled).
    """
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + int(noise_seed_offset) + int(round_index))
