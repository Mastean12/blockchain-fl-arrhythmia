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
from src.federated.privacy import clip_update


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


def he_dp_fedavg_aggregate(global_state, client_states, client_sample_counts, session,
                           clipping_norm, noise_multiplier, noise_generators):
    """Client-side clipping/noising followed by CKKS-encrypted weighted aggregation.

    Every client adds independent Gaussian noise before encryption. The server receives
    only serialized ciphertexts and performs the weighted sum; it has no secret key.
    Noise for client k is calibrated so its weighted contribution alone has standard
    deviation ``noise_multiplier * clipping_norm * max(weight)``. Thus even if all
    other clients reveal their noise, the per-client Gaussian mechanism has multiplier
    at least ``noise_multiplier`` under fixed public weights.
    """
    counts = np.asarray(client_sample_counts, dtype=np.float64)
    if (not client_states or len(client_states) != len(counts) or np.any(counts <= 0)
            or len(noise_generators) != len(client_states)):
        raise ValueError("Need one positive sample count and private noise stream per client")
    if clipping_norm <= 0 or noise_multiplier <= 0:
        raise ValueError("clipping_norm and noise_multiplier must be positive")
    weights = counts / counts.sum()
    keys = _float_keys(global_state)
    reference = torch.from_numpy(_flatten(global_state, keys))
    encrypted_updates, exact_weighted_noisy_updates = [], []
    encrypt_seconds = 0.0
    uplink_bytes = []
    target_weighted_noise_std = float(noise_multiplier * clipping_norm * weights.max())
    for weight, state, generator in zip(weights, client_states, noise_generators):
        if list(state) != list(global_state):
            raise ValueError("Client state keys differ from the global model")
        delta = torch.from_numpy(_flatten(state, keys)) - reference
        clipped, _ = clip_update(delta, float(clipping_norm))
        # Local noise is calibrated so p_k * sigma_k >= the aggregate mechanism's
        # target standard deviation, giving a conservative client-level bound.
        client_noise_std = target_weighted_noise_std / float(weight)
        noise = torch.normal(0.0, client_noise_std, size=clipped.shape, generator=generator,
                             dtype=torch.float64)
        noisy_update = (clipped + noise).numpy()
        exact_weighted_noisy_updates.append(float(weight) * noisy_update)
        started = time.perf_counter()
        chunks = session.encrypt(noisy_update)
        encrypt_seconds += time.perf_counter() - started
        uplink_bytes.append(sum(len(blob) for blob in chunks))
        encrypted_updates.append(chunks)

    started = time.perf_counter()
    server = session.server_view()
    encrypted_aggregate = server.aggregate(encrypted_updates, weights)
    aggregate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    noisy_aggregate = session.decrypt(encrypted_aggregate, len(reference))
    decrypt_seconds = time.perf_counter() - started

    exact_noisy_aggregate = np.sum(np.stack(exact_weighted_noisy_updates), axis=0)
    error = noisy_aggregate - exact_noisy_aggregate
    new_flat = reference.numpy() + noisy_aggregate
    plain_counts = OrderedDict()
    for key, value in global_state.items():
        if value.is_floating_point():
            continue
        stacked = torch.stack([state[key].to(torch.float64) for state in client_states])
        plain_counts[key] = torch.round(torch.tensordot(torch.as_tensor(weights, dtype=torch.float64),
                                                        stacked, dims=1)).to(value.dtype)
    new_state, offset = OrderedDict(), 0
    for key, value in global_state.items():
        if value.is_floating_point():
            n = value.numel()
            updated = torch.from_numpy(new_flat[offset:offset + n].copy()).reshape(value.shape).to(value.dtype)
            if key.endswith("running_var"):
                updated = updated.clamp_min(0.0)
            new_state[key] = updated
            offset += n
        else:
            new_state[key] = plain_counts[key].clone()
    diagnostics = {
        "protocol": "client_clip_gaussian_noise_then_ckks_weighted_sum",
        "participants": len(client_states),
        "noise_multiplier_minimum": float(noise_multiplier),
        "noise_std_minimum_client_update": float(min(target_weighted_noise_std / w for w in weights)),
        "weighted_noise_std_per_client": target_weighted_noise_std,
        "encrypt_seconds_all_clients": encrypt_seconds,
        "aggregate_seconds_server": aggregate_seconds,
        "decrypt_seconds_key_holders": decrypt_seconds,
        "uplink_ciphertext_bytes_total": int(sum(uplink_bytes)),
        "uplink_ciphertext_bytes_per_client_mean": float(np.mean(uplink_bytes)),
        "downlink_ciphertext_bytes": int(sum(len(blob) for blob in encrypted_aggregate)),
        "aggregate_max_abs_error": float(np.max(np.abs(error))),
        "aggregate_rms_error": float(np.sqrt(np.mean(error ** 2))),
        "dimension": int(len(reference)),
    }
    return new_state, diagnostics


def aggregate_he_dp_v2(global_state, client_states, fixed_weights, session, clipping_norm,
                       noise_multiplier, noise_generator):
    """Fixed-roster protocol: clip -> public-key encrypt -> weighted CKKS sum ->
    trusted decryptor -> one aggregate Gaussian perturbation.

    The returned diagnostics include clean-aggregate-dependent CKKS error and
    must be kept inside the trusted execution boundary. The function never
    writes those values or the noise realization to disk.
    """
    weights = np.asarray(fixed_weights, dtype=np.float64)
    if not client_states or len(client_states) != len(weights):
        raise ValueError("fixed roster requires exactly one update per configured weight")
    if not np.isfinite(weights).all() or np.any(weights <= 0) or abs(weights.sum() - 1.0) > 1e-12:
        raise ValueError("fixed weights must be finite, positive, and sum to one; no renormalization")
    if clipping_norm <= 0 or noise_multiplier <= 0:
        raise ValueError("clipping_norm and noise_multiplier must be positive")
    keys = _float_keys(global_state)
    reference = _flatten(global_state, keys)
    client_encryptor = session.client_encryptor()
    server = session.server_view()
    clipped_updates, encrypted_updates = [], []
    encrypt_seconds = 0.0
    uplink_bytes = []
    for state in client_states:
        if list(state) != list(global_state):
            raise ValueError("Client state keys differ from the global model")
        delta = torch.from_numpy(_flatten(state, keys) - reference)
        clipped, _ = clip_update(delta, float(clipping_norm))
        clipped_np = clipped.numpy()
        clipped_updates.append(clipped_np)
        started = time.perf_counter()
        chunks = client_encryptor.encrypt(clipped_np)
        encrypt_seconds += time.perf_counter() - started
        encrypted_updates.append(chunks)
        uplink_bytes.append(sum(len(blob) for blob in chunks))

    started = time.perf_counter()
    encrypted_sum = server.aggregate(encrypted_updates, weights)
    aggregate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    clean_aggregate = session.decrypt(encrypted_sum, len(reference))
    decrypt_seconds = time.perf_counter() - started

    exact_clean = np.tensordot(weights, np.stack(clipped_updates), axes=1)
    ckks_error = clean_aggregate - exact_clean
    sensitivity = float(clipping_norm) * float(weights.max())
    noise_std = float(noise_multiplier) * sensitivity
    noise_started = time.perf_counter()
    noise = torch.normal(0.0, noise_std, size=torch.Size([len(reference)]),
                         generator=noise_generator, dtype=torch.float64).numpy()
    noise_seconds = time.perf_counter() - noise_started
    released_update = clean_aggregate + noise
    new_flat = reference + released_update
    new_state, offset = OrderedDict(), 0
    for key, value in global_state.items():
        if value.is_floating_point():
            n = value.numel()
            updated = torch.from_numpy(new_flat[offset:offset + n].copy()).reshape(value.shape).to(value.dtype)
            if key.endswith("running_var"):
                updated = updated.clamp_min(0.0)
            new_state[key] = updated
            offset += n
        else:
            # Integer BatchNorm counters are not data-dependent releases.
            new_state[key] = value.clone()
    return new_state, {
        "protocol": "fixed_weight_clip_ckks_sum_trusted_decryptor_aggregate_gaussian_v2",
        "participants": len(client_states), "sensitivity_l2": sensitivity,
        "aggregate_noise_std": noise_std, "noise_multiplier": float(noise_multiplier),
        "encrypt_seconds_all_clients": encrypt_seconds,
        "aggregate_seconds_server": aggregate_seconds,
        "decrypt_seconds_trusted_decryptor": decrypt_seconds,
        "noise_seconds_trusted_decryptor": noise_seconds,
        "uplink_ciphertext_bytes_total": int(sum(uplink_bytes)),
        "uplink_ciphertext_bytes_per_client": [int(v) for v in uplink_bytes],
        "downlink_ciphertext_bytes": int(sum(len(blob) for blob in encrypted_sum)),
        "ckks_clean_aggregate_max_abs_error": float(np.max(np.abs(ckks_error))),
        "ckks_clean_aggregate_rms_error": float(np.sqrt(np.mean(ckks_error ** 2))),
        "dimension": int(len(reference)),
    }
