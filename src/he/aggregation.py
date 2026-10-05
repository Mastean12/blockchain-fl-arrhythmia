"""HE-FedAvg: sample-weighted FedAvg where the server aggregates CKKS-encrypted client updates.

Per round:
  1. each client computes its update Delta_k = w_k - w_t (all floating-point state
     entries) in plaintext and encrypts it with the shared public key;
  2. the server, holding only the public context, computes Enc(sum_k p_k Delta_k)
     with public weights p_k = n_k / sum_j n_j;
  3. the key holders decrypt it and set w_{t+1} = w_t + sum_k p_k Delta_k.

Integer BatchNorm counters are aggregated in plaintext exactly as in FedAvg; they
are a public function of the client sample counts. For diagnostics only, the
plaintext FedAvg aggregate of the same client states is computed alongside, to
measure the CKKS numerical error. A real server could not compute this.
"""
from collections import OrderedDict
import time

import numpy as np
import torch

from src.federated.fedavg import fedavg_aggregate


def _float_keys(state):
    return [key for key, value in state.items() if value.is_floating_point()]


def _flatten(state, keys):
    return torch.cat([state[key].reshape(-1).to(torch.float64) for key in keys]).numpy()


def he_fedavg_aggregate(global_state, client_states, client_sample_counts, session):
    counts = np.asarray(client_sample_counts, dtype=np.float64)
    if not client_states or len(client_states) != len(counts) or np.any(counts <= 0):
        raise ValueError("Need one positive sample count per client state")
    weights = counts / counts.sum()
    keys = _float_keys(global_state)
    reference = _flatten(global_state, keys)
    updates = []
    for state in client_states:
        if list(state) != list(global_state):
            raise ValueError("Client state keys differ from the global model")
        updates.append(_flatten(state, keys) - reference)

    t0 = time.perf_counter()
    encrypted = [session.encrypt(update) for update in updates]          # clients
    t1 = time.perf_counter()
    aggregated = session.aggregate(encrypted, weights)                    # server (public context)
    t2 = time.perf_counter()
    decrypted = session.decrypt(aggregated, len(reference))              # key holders
    t3 = time.perf_counter()

    new_flat = reference + decrypted
    new_state, offset = OrderedDict(), 0
    plaintext = fedavg_aggregate(client_states, client_sample_counts)    # diagnostic only
    for key, value in global_state.items():
        if value.is_floating_point():
            n = value.numel()
            new_state[key] = torch.from_numpy(new_flat[offset:offset + n].copy()).reshape(value.shape).to(value.dtype)
            offset += n
        else:
            new_state[key] = plaintext[key].clone()
    exact_update = np.tensordot(weights, np.stack(updates), axes=1)  # exact float64 weighted sum
    agg_error = decrypted - exact_update  # CKKS error of the decrypted aggregate update
    state_error = _flatten(new_state, keys) - _flatten(plaintext, keys)  # vs plaintext FedAvg float32 state
    diagnostics = {
        "encrypt_seconds": t1 - t0, "aggregate_seconds": t2 - t1, "decrypt_seconds": t3 - t2,
        "ciphertext_chunks_per_client": len(encrypted[0]),
        "uplink_ciphertext_bytes": [sum(len(b) for b in chunks) for chunks in encrypted],
        "downlink_ciphertext_bytes": sum(len(b) for b in aggregated),
        "aggregate_max_abs_error": float(np.max(np.abs(agg_error))),
        "aggregate_rms_error": float(np.sqrt(np.mean(agg_error ** 2))),
        "aggregate_max_abs_value": float(np.max(np.abs(exact_update))),
        "state_max_abs_error_vs_plaintext_fedavg": float(np.max(np.abs(state_error))),
        "dimension": int(len(reference)),
    }
    return new_state, diagnostics
