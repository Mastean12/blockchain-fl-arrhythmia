"""Controlled malicious-client attacks on plain FedAvg (Day 17; no defense).

Threat model
------------
One persistent compromised client (chosen by a seeded draw) participates in every
round. It receives the broadcast global model w_t, trains honestly to obtain its
honest state w_k, and then transmits a manipulated state w_t + Delta' instead of
w_t + Delta, where Delta = w_k - w_t over all floating-point state entries
(weights, biases, BatchNorm affine parameters and running statistics). Integer
BatchNorm counters are sent unchanged. The attacker reports its true sample count,
does not collude, and does not see other clients' updates. The server runs
unmodified sample-weighted FedAvg without any validation of client updates.

Attacks
-------
* ``sign_flip``: Delta' = -scale * Delta
* ``scale``: Delta' = scale * Delta (boosting / model-replacement style magnitude)
* ``gaussian_norm_matched``: Delta' = multiplier * ||Delta|| * z / ||z||, z ~ N(0, I)
  from a seeded generator (random direction with the honest update's magnitude)
"""
from collections import OrderedDict

import numpy as np
import torch

ATTACK_TYPES = {"sign_flip", "scale", "gaussian_norm_matched"}


def validate_attack_config(attack):
    required = {"enabled", "name", "type", "selection_seed_offset"}
    missing = required - set(attack)
    if missing:
        raise ValueError(f"attack config missing {sorted(missing)}")
    if attack["enabled"] is not True:
        raise ValueError("attack.enabled must be true when an attack section is present")
    if attack["type"] not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack type {attack['type']!r}")
    if attack["type"] in {"sign_flip", "scale"} and not float(attack.get("scale", 0)) > 0:
        raise ValueError("sign_flip and scale attacks need a positive 'scale'")
    if attack["type"] == "gaussian_norm_matched":
        if not float(attack.get("norm_multiplier", 0)) > 0 or "noise_seed_offset" not in attack:
            raise ValueError("gaussian_norm_matched needs a positive 'norm_multiplier' and a 'noise_seed_offset'")
    return attack


def choose_attacker(seed, selection_seed_offset, num_clients):
    """Seeded, recorded choice of the single compromised client."""
    return int(np.random.default_rng([int(seed), int(selection_seed_offset)]).integers(num_clients))


def noise_generator(seed, noise_seed_offset, round_index):
    return torch.Generator().manual_seed(int(seed) * 1_000_033 + int(noise_seed_offset) + int(round_index))


def _float_keys(state):
    return [key for key, value in state.items() if value.is_floating_point()]


def apply_attack(global_state, honest_state, attack, generator=None):
    """Return (malicious_state, diagnostics). The honest state is not modified."""
    keys = _float_keys(global_state)
    reference = torch.cat([global_state[k].reshape(-1).to(torch.float64) for k in keys])
    honest = torch.cat([honest_state[k].reshape(-1).to(torch.float64) for k in keys]) - reference
    honest_norm = float(torch.linalg.vector_norm(honest))
    kind = attack["type"]
    if kind == "sign_flip":
        malicious = -float(attack["scale"]) * honest
    elif kind == "scale":
        malicious = float(attack["scale"]) * honest
    elif kind == "gaussian_norm_matched":
        if generator is None:
            raise ValueError("gaussian_norm_matched needs a seeded generator")
        z = torch.normal(0.0, 1.0, size=honest.shape, generator=generator, dtype=torch.float64)
        malicious = float(attack["norm_multiplier"]) * honest_norm * z / torch.linalg.vector_norm(z)
    else:
        raise ValueError(f"Unknown attack type {kind!r}")
    flat = reference + malicious
    out, offset = OrderedDict(), 0
    for key, value in honest_state.items():
        if value.is_floating_point():
            n = value.numel()
            out[key] = flat[offset:offset + n].reshape(value.shape).to(value.dtype)
            offset += n
        else:
            out[key] = value.clone()
    malicious_norm = float(torch.linalg.vector_norm(malicious))
    cosine = float(torch.dot(malicious, honest) / (malicious_norm * honest_norm)) if honest_norm and malicious_norm else 0.0
    return out, {"honest_update_norm": honest_norm, "malicious_update_norm": malicious_norm,
                 "cosine_malicious_vs_honest": cosine}


def flat_float_state(state):
    """float64 vector of all floating-point entries (for trajectory / deviation analysis)."""
    return torch.cat([state[k].reshape(-1).to(torch.float64) for k in _float_keys(state)]).numpy()
