"""Orchestrate the simulated FedAvg baseline: partition, train, validate, test once.

global model -> broadcast -> local client training -> collect client states
-> sample-weighted FedAvg -> new global model -> validation -> next round.

Validation is the only split used during training and for selecting the final
round (minimum validation loss, the same rule as centralized_cnn_v1). The test
split is opened once, after selection, by `evaluate_test_once`.
"""
import copy
import csv
import json
from pathlib import Path
import random
import secrets
import time

import numpy as np
import torch

from src.models.cnn1d import ECG1DCNN, count_trainable_parameters, model_architecture
from src.models.train_cnn import ECGSegmentDataset

from .config import load_federated_config
from .data import ClientECGDataset, load_class_to_index
from .evaluation import (communication_cost, compare_with_centralized, evaluate_test_once,
                         validation_metrics)
from .fedavg import (batchnorm_state_keys, client_seed, client_start_state, client_update, fedavg_aggregate,
                     get_model_state, select_clients, set_model_state)
from .partition import build_client_partition, client_distribution_tables
from .privacy import (dp_fedavg_aggregate, he_dp_v2_account, noise_generator, private_noise_generator,
                      validate_privacy_config)
from . import plots


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def apply_overrides(config, overrides):
    """Merge `{section: {key: value}}` or `{key: value}` overrides into a copy of the config."""
    config = copy.deepcopy(config)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    return config


def prepare_partition(config, output_root=None, write=True):
    """Build the client partition and (optionally) write the distribution report."""
    part = config["partition"]
    class_to_index = load_class_to_index(config["data"]["label_config"])
    class_names = list(class_to_index)
    partition = build_client_partition(config["data"]["split_root"], int(part["num_clients"]), int(part["seed"]),
                                       part["strategy"], int(part.get("min_client_segments", 1)))
    summary, counts, proportions = client_distribution_tables(partition, class_names)
    if write:
        root = Path(output_root or config["output_root"])
        exp = config["experiment_id"]
        tables, metrics, figures = (root / d / "federated" for d in ("tables", "metrics", "figures"))
        for d in (tables, metrics, figures):
            d.mkdir(parents=True, exist_ok=True)
        _write_json(metrics / f"{exp}_client_partition.json", partition)
        _write_csv(tables / f"{exp}_client_summary.csv", summary)
        _write_csv(tables / f"{exp}_client_class_counts.csv", counts)
        _write_csv(tables / f"{exp}_client_class_proportions.csv", proportions)
        plots.plot_client_distribution(counts, class_names, figures / f"{exp}_client_distribution.png")
    return partition, summary, counts, proportions


def run_experiment(config_path="configs/federated/fedavg_v1.json", *, evaluate_test=True, output_root=None,
                   overrides=None, overwrite=False, log=print, comparison_prefix=""):
    config = apply_overrides(load_federated_config(config_path), overrides)
    privacy = config.get("privacy")
    he_dp_v2 = privacy is not None and privacy.get("protocol") == "he_dp_v2_aggregate_noise"
    accounting = None
    if privacy is not None:
        if config["aggregation"].get("batchnorm", "aggregate") != "aggregate":
            raise ValueError("Differential privacy is a separate one-factor experiment; do not combine with local BatchNorm")
        if he_dp_v2:
            if privacy.get("enabled") is not True:
                raise ValueError("HE+DP v2 requires privacy.enabled=true")
            if float(config["participation"]["fraction"]) != 1.0:
                raise ValueError("HE+DP v2 requires the complete fixed roster every round")
            if privacy.get("adjacency") != "add_remove_one_client_slot":
                raise ValueError("HE+DP v2 supports only add/remove-one-client-slot adjacency")
            roster = privacy.get("fixed_roster")
            if (not isinstance(roster, list)
                    or len(roster) != int(config["partition"]["num_clients"])
                    or any(not isinstance(client_id, str) or not client_id for client_id in roster)
                    or len(set(roster)) != len(roster)):
                raise ValueError("fixed_roster must contain one unique client ID per configured client")
            fixed_weights = privacy.get("fixed_weights")
            if (not isinstance(fixed_weights, list)
                    or len(fixed_weights) != int(config["partition"]["num_clients"])):
                raise ValueError("fixed_weights must have one configured weight per roster slot")
            configured_rounds = privacy.get("rounds")
            if (not isinstance(configured_rounds, int) or isinstance(configured_rounds, bool)
                    or configured_rounds != int(config["training"]["rounds"])):
                raise ValueError("privacy.rounds must match training.rounds for HE+DP v2 accounting")
            if privacy.get("accountant") != "full_participation_GDP_plus_analytic_Gaussian_conversion":
                raise ValueError("HE+DP v2 requires the full-participation GDP accountant")
            if privacy.get("noise_placement") != "one Gaussian vector after trusted decryption of the homomorphic weighted sum":
                raise ValueError("HE+DP v2 requires aggregate Gaussian noise after trusted decryption")
            accounting = he_dp_v2_account(privacy["fixed_weights"], float(privacy["clipping_norm"]),
                                          float(privacy["noise_multiplier"]), int(config["training"]["rounds"]),
                                          float(privacy["delta"]), int(config["partition"]["num_clients"]))
        else:
            accounting = validate_privacy_config(privacy, config["training"]["rounds"],
                                                 config["participation"]["fraction"])
    he_cfg = config.get("homomorphic_encryption")
    he_dp = privacy is not None and he_cfg is not None and not he_dp_v2
    if he_dp and privacy.get("noise_placement") != "client_before_encryption":
        raise ValueError("Integrated HE+DP requires client clipping/noising before encryption")
    if he_dp_v2 and he_cfg is None:
        raise ValueError("HE+DP v2 requires homomorphic encryption configuration")
    if he_dp_v2 and (
            he_cfg.get("encrypted_quantity") != "clipped_client_update"
            or he_cfg.get("aggregation") != "fixed_public_weighted_fedavg"):
        raise ValueError("HE+DP v2 requires clipped client updates and fixed public weighted aggregation")
    he_session = None
    if he_cfg is not None:
        if config["aggregation"].get("batchnorm", "aggregate") != "aggregate":
            raise ValueError("HE aggregation does not support local BatchNorm")
        from src.he.aggregation import he_fedavg_aggregate
        from src.he.ckks import CKKSSession
        from src.he.config import validate_he_config
        he_params = validate_he_config(he_cfg)
    bc_cfg = config.get("blockchain")
    combined_he_dp_v2_blockchain = False
    if bc_cfg is not None:
        if bc_cfg.get("enabled") is not True or not bc_cfg.get("ledger_id"):
            raise ValueError("blockchain section needs enabled=true and a ledger_id")
        configured_anchor_path = bc_cfg.get("anchor_export_path")
        if configured_anchor_path is not None and (
                not isinstance(configured_anchor_path, str) or not configured_anchor_path.strip()):
            raise ValueError("blockchain.anchor_export_path must be a non-empty path string or null")
        mode = bc_cfg.get("mode")
        if mode not in (None, "lifecycle_gate_v2"):
            raise ValueError(f"Unsupported blockchain mode {mode!r}")
        combined_he_dp_v2_blockchain = (
            mode == "lifecycle_gate_v2" and he_dp_v2 and he_cfg is not None
            and he_cfg.get("scheme") == "CKKS"
        )
        if ((privacy is not None or he_cfg is not None
             or config["aggregation"].get("batchnorm", "aggregate") != "aggregate")
                and not combined_he_dp_v2_blockchain):
            raise ValueError("Blockchain supports plain FedAvg or HE+DP v2 aggregate-noise with CKKS "
                             "and lifecycle_gate_v2; other DP/HE/local-BatchNorm combinations are unsupported")
        from src.blockchain.fl_recorder import FLLedgerRecorder, state_hash
    attack_cfg = config.get("attack")
    if attack_cfg is not None:
        if (privacy is not None or he_cfg is not None or bc_cfg is not None
                or config["aggregation"].get("batchnorm", "aggregate") != "aggregate"):
            raise ValueError("Attacks are evaluated against plain FedAvg only (no DP, HE, blockchain or local BatchNorm)")
        from .attacks import apply_attack, choose_attacker, flat_float_state, noise_generator as attack_noise
        from .attacks import validate_attack_config
        validate_attack_config(attack_cfg)
    defense_cfg = config.get("defense")
    if defense_cfg is not None:
        if (privacy is not None or he_cfg is not None or bc_cfg is not None
                or config["aggregation"].get("batchnorm", "aggregate") != "aggregate"):
            raise ValueError("The proposed defense is evaluated with plain FedAvg only (no DP, HE, blockchain or local BatchNorm)")
        from .proposed_defense import partition_keys, robust_aggregate, validate_defense_config
        from .attacks import flat_float_state
        validate_defense_config(defense_cfg)
    root = Path(output_root or config["output_root"])
    config["output_root"] = str(root).replace("\\", "/")
    exp = config["experiment_id"]
    anchor_path = None
    if bc_cfg is not None:
        configured_anchor_path = bc_cfg.get("anchor_export_path")
        anchor_path = (Path(configured_anchor_path) if configured_anchor_path is not None
                       else root / "anchors" / f"{exp}_anchor.json")
        if anchor_path.exists():
            raise FileExistsError(f"refusing to overwrite existing ledger anchor: {anchor_path}")
    model_dir = root / "models" / "federated"
    metrics_dir, tables_dir, figures_dir = (root / d / "federated" for d in ("metrics", "tables", "figures"))
    model_path = model_dir / f"{exp}_global.pt"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"{model_path} exists; pass overwrite=True to replace this experiment")
    for d in (model_dir, metrics_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    private_seed_path = root / "private" / f"{exp}_noise_seeds.json"
    if he_dp or he_dp_v2:
        private_seed_path.parent.mkdir(parents=True, exist_ok=True)
        if private_seed_path.exists():
            seed_data = json.loads(private_seed_path.read_text(encoding="utf-8"))
            if he_dp_v2:
                private_noise_seeds = seed_data["trusted_decryptor_noise_seed"]
            else:
                private_noise_seeds = seed_data["client_seeds"]
                if len(private_noise_seeds) != int(config["partition"]["num_clients"]):
                    raise ValueError("Private HE+DP noise-seed artifact has the wrong client count")
        else:
            if he_dp_v2:
                private_noise_seeds = secrets.randbits(63)
                _write_json(private_seed_path, {"trusted_decryptor_noise_seed": private_noise_seeds,
                                                "warning": "Private reproducibility material; do not publish or commit"})
            else:
                private_noise_seeds = [secrets.randbits(63) for _ in range(int(config["partition"]["num_clients"]))]
                _write_json(private_seed_path, {"client_seeds": private_noise_seeds,
                                                "warning": "Private reproducibility material; do not publish or commit"})

    seed = int(config["random_seed"])
    train_cfg, runtime = config["training"], config["runtime"]
    seed_everything(seed)
    torch.set_num_threads(int(runtime["cpu_threads"]))
    device = runtime["device"]
    eval_bs = int(runtime["evaluation_batch_size"])

    class_to_index = load_class_to_index(config["data"]["label_config"])
    class_names = list(class_to_index)
    partition, summary, _, _ = prepare_partition(config, root, write=not he_dp_v2)
    if he_dp_v2 and [c["client_id"] for c in partition["clients"]] != privacy.get("fixed_roster"):
        raise ValueError("generated participant roster differs from the configured public fixed roster")
    split_root = Path(config["data"]["split_root"])
    client_data = [ClientECGDataset(split_root, c["records"], class_to_index) for c in partition["clients"]]
    for client, data in zip(partition["clients"], client_data):
        if len(data) != client["segments"]:
            raise RuntimeError(f"{client['client_id']}: loaded {len(data)} segments, expected {client['segments']}")
    validation = ECGSegmentDataset(split_root / "validation", class_to_index)
    log((f"Clients: {len(client_data)} fixed roster slots; validation: {len(validation)}" if he_dp_v2
         else f"Clients: {[len(d) for d in client_data]} segments; validation: {len(validation)}"))

    # Same initialization procedure as centralized_cnn_v1: seed, then construct the model.
    seed_everything(seed)
    global_model = ECG1DCNN(num_classes=len(class_names), dropout=float(config["model"]["dropout"])).to(device)
    worker = copy.deepcopy(global_model)
    global_state = get_model_state(global_model)
    trainable = count_trainable_parameters(global_model)
    # aggregation.batchnorm: "aggregate" = standard FedAvg (default); "local" = FedBN-style, where each
    # client keeps its own BatchNorm entries across rounds. In both modes the evaluated global model is
    # the sample-weighted average of all client states; in "local" mode its BatchNorm part is used only
    # for evaluation and is never broadcast back to clients.
    bn_mode = config["aggregation"].get("batchnorm", "aggregate")
    bn_keys = batchnorm_state_keys(global_model) if bn_mode == "local" else []
    local_bn = {}
    client_validation = []
    dp_rounds, dp_clients = [], []
    he_rounds = []
    he_dp_v2_public_rounds = []
    attack_rounds, trajectory = [], []
    defense_clients, defense_rounds = [], []
    defense_keys = partition_keys(global_model) if defense_cfg is not None else None
    recorder = None
    if bc_cfg is not None:
        # The HE+DP v2 recorder stores commitments only; plain FedAvg retains its existing state archive.
        recorder_metadata = {"aggregation": "FedAvg", "seed": seed,
                             "partition_strategy": config["partition"]["strategy"]}
        if combined_he_dp_v2_blockchain:
            recorder_metadata.update({"privacy_protocol": privacy["protocol"], "he_scheme": "CKKS",
                                      "weighting": "fixed_public"})
        recorder = FLLedgerRecorder(bc_cfg["ledger_id"], exp, global_state, root / "offchain" / exp,
                                    limits=bc_cfg.get("limits"),
                                    metadata=recorder_metadata,
                                    expected_roster=(privacy["fixed_roster"] if combined_he_dp_v2_blockchain
                                                     else [c["client_id"] for c in partition["clients"]]
                                                     if bc_cfg.get("mode") == "lifecycle_gate_v2" else None),
                                    expected_weights=(privacy["fixed_weights"]
                                                      if combined_he_dp_v2_blockchain else None))
    if he_cfg is not None:
        key_start = time.perf_counter()
        he_session = CKKSSession(he_params)
        float_entries = sum(v.numel() for v in global_state.values() if v.is_floating_point())
        he_setup = {"key_generation_seconds": time.perf_counter() - key_start,
                    "public_context_bytes": len(he_session.public_context_bytes), "slots": he_params.slots,
                    "encrypted_entries": float_entries,
                    "ciphertext_chunks_per_client": he_session.chunks(float_entries)}

    attacker = None
    if attack_cfg is not None:
        attacker = int(attack_cfg["client_index"]) if "client_index" in attack_cfg else choose_attacker(
            seed, attack_cfg["selection_seed_offset"], len(client_data))
        log(f"Attack {attack_cfg['name']}: compromised client {partition['clients'][attacker]['client_id']}")
    track_trajectory = attacker is not None or defense_cfg is not None
    if track_trajectory:
        trajectory.append(flat_float_state(global_state))

    history = [{"round": 0, "participants": 0, "client_train_loss_weighted": None,
                "client_train_accuracy_weighted": None, "elapsed_seconds": 0.0,
                **validation_metrics(global_model, validation, len(class_names), eval_bs, device)}]
    client_history = []
    best = {"round": 0, "loss": history[0]["validation_loss"], "state": global_state}
    participants_per_round = []
    started = time.perf_counter()
    num_clients = len(client_data)
    for round_index in range(1, int(train_cfg["rounds"]) + 1):
        previous_global_state = global_state
        selected = select_clients(num_clients, float(config["participation"]["fraction"]), seed, round_index)
        if he_dp_v2 and selected != list(range(num_clients)):
            raise RuntimeError("fixed-roster HE+DP v2 requires every configured client in every round")
        participants_per_round.append(len(selected))
        updates = []
        for k in selected:
            start_state = client_start_state(global_state, local_bn.get(k)) if bn_mode == "local" else global_state
            update = client_update(worker, start_state, client_data[k], local_epochs=int(train_cfg["local_epochs"]),
                                   batch_size=int(train_cfg["batch_size"]),
                                   learning_rate=float(train_cfg["learning_rate"]),
                                   optimizer_name=train_cfg["optimizer"], seed=client_seed(seed, round_index, k),
                                   device=device, num_workers=int(runtime["num_workers"]),
                                   shuffle=bool(train_cfg["shuffle_training"]))
            updates.append(update)
            if bn_mode == "local":
                local_bn[k] = {key: update["state"][key] for key in bn_keys}
            if not (he_dp or he_dp_v2):
                client_history.append({"round": round_index, "client_id": partition["clients"][k]["client_id"],
                                       "num_samples": update["num_samples"], "train_loss": update["train_loss"],
                                       "train_accuracy": update["train_accuracy"]})
        counts = [u["num_samples"] for u in updates]
        if attacker is not None and attacker in selected:
            position = selected.index(attacker)
            generator = (attack_noise(seed, attack_cfg["noise_seed_offset"], round_index)
                         if attack_cfg["type"] == "gaussian_norm_matched" else None)
            updates[position]["state"], attack_diag = apply_attack(global_state, updates[position]["state"],
                                                                   attack_cfg, generator)
            attack_rounds.append({"round": round_index, "attacker": partition["clients"][attacker]["client_id"],
                                  "attacker_weight": counts[position] / sum(counts), **attack_diag})
        round_ids = [partition["clients"][k]["client_id"] for k in selected]
        if recorder is not None and bc_cfg.get("mode") == "lifecycle_gate_v2":
            if combined_he_dp_v2_blockchain:
                recorder.prepare_private_round(
                    round_index, round_ids, [u["state"] for u in updates], global_state, privacy["fixed_weights"],
                    metadata={"privacy_protocol": privacy["protocol"], "he_scheme": "CKKS",
                              "aggregation": "aggregate_he_dp_v2", "weighting": "fixed_public"})
            else:
                recorder.prepare_round(round_index, round_ids, counts, [u["state"] for u in updates], global_state,
                                       metadata={"aggregation": "FedAvg", "participants": len(selected)})
        if defense_cfg is not None:
            ids = [partition["clients"][k]["client_id"] for k in selected]
            global_state, d_rows, d_summary = robust_aggregate(global_state, [u["state"] for u in updates], counts,
                                                               defense_cfg, defense_keys, ids)
            for k, row in zip(selected, d_rows):
                defense_clients.append({"round": round_index, "is_attacker": k == attacker, **row})
            defense_rounds.append({"round": round_index, **d_summary})
        elif he_dp:
            from src.he.aggregation import he_dp_fedavg_aggregate
            client_noise = [private_noise_generator(private_noise_seeds[k], round_index, k) for k in selected]
            global_state, he_dp_diag = he_dp_fedavg_aggregate(
                global_state, [u["state"] for u in updates], counts, he_session,
                float(privacy["clipping_norm"]), float(privacy["noise_multiplier"]), client_noise)
            he_dp_diag["round"] = round_index
            he_rounds.append(he_dp_diag)
            dp_rounds.append({"round": round_index, "privacy_unit": "client",
                              "epsilon": accounting["epsilon"], "delta": accounting["delta"],
                              "noise_multiplier_lower_bound": accounting["noise_multiplier"],
                              "sampling_rate": accounting["client_sampling_rate"]})
        elif he_dp_v2:
            from src.he.aggregation import aggregate_he_dp_v2
            noise_generator_v2 = torch.Generator().manual_seed(
                (int(private_noise_seeds) + round_index * 1_000_003) % (2 ** 63 - 1))
            aggregate_v2 = lambda: aggregate_he_dp_v2(
                global_state, [u["state"] for u in updates], privacy["fixed_weights"], he_session,
                float(privacy["clipping_norm"]), float(privacy["noise_multiplier"]), noise_generator_v2)
            if combined_he_dp_v2_blockchain:
                global_state, he_dp_diag = recorder.aggregate_private_if_prepared(
                    round_index, round_ids, [u["state"] for u in updates], privacy["fixed_weights"],
                    global_state, aggregate_v2)
            else:
                global_state, he_dp_diag = aggregate_v2()
            he_dp_diag["round"] = round_index
            he_rounds.append(he_dp_diag)
            he_dp_v2_public_rounds.append({k: he_dp_diag[k] for k in (
                "round", "participants", "encrypt_seconds_all_clients", "aggregate_seconds_server",
                "decrypt_seconds_trusted_decryptor", "noise_seconds_trusted_decryptor",
                "uplink_ciphertext_bytes_total",
                "downlink_ciphertext_bytes", "dimension")})
            dp_rounds.append({"round": round_index, "privacy_unit": accounting["privacy_unit"],
                              "epsilon": accounting["epsilon"], "delta": accounting["delta"],
                              "noise_multiplier": accounting["noise_multiplier"],
                              "sensitivity_l2": accounting["sensitivity_l2"],
                              "aggregate_noise_std": accounting["aggregate_noise_std"]})
        elif he_session is not None:
            global_state, he = he_fedavg_aggregate(global_state, [u["state"] for u in updates], counts, he_session)
            he_rounds.append({"round": round_index, "participants": len(selected),
                              "encrypt_seconds_all_clients": he["encrypt_seconds"],
                              "aggregate_seconds_server": he["aggregate_seconds"],
                              "decrypt_seconds": he["decrypt_seconds"],
                              "uplink_ciphertext_bytes_total": int(sum(he["uplink_ciphertext_bytes"])),
                              "uplink_ciphertext_bytes_per_client_mean": float(np.mean(he["uplink_ciphertext_bytes"])),
                              "downlink_ciphertext_bytes_per_client": he["downlink_ciphertext_bytes"],
                              "aggregate_max_abs_error": he["aggregate_max_abs_error"],
                              "aggregate_rms_error": he["aggregate_rms_error"],
                              "aggregate_max_abs_value": he["aggregate_max_abs_value"],
                              "state_max_abs_error_vs_plaintext_fedavg": he["state_max_abs_error_vs_plaintext_fedavg"],
                              "dimension": he["dimension"]})
        elif privacy is None:
            if recorder is not None and bc_cfg.get("mode") == "lifecycle_gate_v2":
                global_state = recorder.aggregate_if_prepared(
                    round_index, round_ids, counts, [u["state"] for u in updates],
                    lambda: fedavg_aggregate([u["state"] for u in updates], counts))
            else:
                global_state = fedavg_aggregate([u["state"] for u in updates], counts)
        else:
            global_state, dp = dp_fedavg_aggregate(
                global_state, [u["state"] for u in updates], counts, float(privacy["clipping_norm"]),
                float(privacy["noise_multiplier"]), noise_generator(seed, privacy["noise_seed_offset"], round_index))
            # Diagnostics below use unclipped norms: they are NOT covered by the DP guarantee and would not
            # be released in a deployment. They are kept only to interpret this simulation.
            dp_rounds.append({"round": round_index, "sensitivity": dp["sensitivity"], "noise_std": dp["noise_std"],
                              "noise_l2_norm": dp["noise_l2_norm"],
                              "aggregate_signal_l2_norm": dp["aggregate_signal_l2_norm"],
                              "noise_to_signal_ratio": dp["noise_l2_norm"] / max(dp["aggregate_signal_l2_norm"], 1e-12),
                              "clients_clipped": int(sum(dp["clipped"])), "median_update_norm": float(np.median(dp["update_norms"])),
                              "max_update_norm": float(max(dp["update_norms"])), "dimension": dp["dimension"]})
            for k, norm_value, was_clipped in zip(selected, dp["update_norms"], dp["clipped"]):
                dp_clients.append({"round": round_index, "client_id": partition["clients"][k]["client_id"],
                                   "update_l2_norm_unclipped": norm_value, "clipped": was_clipped})
        if recorder is not None:
            if combined_he_dp_v2_blockchain:
                recorder.complete_private_round(
                    round_index, round_ids, [u["state"] for u in updates], privacy["fixed_weights"],
                    # These are the exact pre-aggregation weights committed in the prepared phase.
                    previous_global_state=previous_global_state, global_state=global_state,
                    metadata={"privacy_protocol": privacy["protocol"], "he_scheme": "CKKS",
                              "aggregation": "aggregate_he_dp_v2", "weighting": "fixed_public"})
            elif bc_cfg.get("mode") == "lifecycle_gate_v2":
                recorder.complete_round(round_index, round_ids, counts, [u["state"] for u in updates], global_state,
                                        metadata={"aggregation": "FedAvg", "participants": len(selected)})
            else:
                recorder.record_round(round_index, round_ids, counts, [u["state"] for u in updates], global_state,
                                      metadata={"aggregation": "FedAvg", "participants": len(selected)})
        if track_trajectory:
            vector = flat_float_state(global_state)
            trajectory.append(vector)
            if attacker is not None:
                attack_rounds[-1]["global_state_finite"] = bool(np.isfinite(vector).all())
        set_model_state(global_model, global_state)
        weights = np.asarray(counts, dtype=float) / sum(counts)
        row = {"round": round_index, "participants": len(selected),
        "client_train_loss_weighted": (None if (he_dp or he_dp_v2) else
                   float(np.dot(weights, [u["train_loss"] for u in updates]))),
               "client_train_accuracy_weighted": (None if (he_dp or he_dp_v2) else
                   float(np.dot(weights, [u["train_accuracy"] for u in updates]))),
               "elapsed_seconds": time.perf_counter() - started,
               **validation_metrics(global_model, validation, len(class_names), eval_bs, device)}
        history.append(row)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if bn_mode == "local":
            # Validation-only diagnostic: each client's personalised model (shared weights + own BatchNorm).
            for k in sorted(local_bn):
                set_model_state(worker, client_start_state(global_state, local_bn[k]))
                client_validation.append({"round": round_index, "client_id": partition["clients"][k]["client_id"],
                                          **validation_metrics(worker, validation, len(class_names), eval_bs, device)})
        if row["validation_loss"] < best["loss"]:
            best = {"round": round_index, "loss": row["validation_loss"], "state": global_state,
                    "local_bn": {k: dict(v) for k, v in local_bn.items()}}
    elapsed = time.perf_counter() - started

    selected_block = None
    if recorder is not None:
        valid, problems = recorder.ledger.validate()
        if not valid:
            raise RuntimeError(f"Ledger invalid before checkpoint selection: {problems[:3]}")
        selected_block = next(block for block in recorder.ledger.blocks
                              if block.round == best["round"] and block.metadata.get("phase") != "prepared")
        if state_hash(best["state"]) != selected_block.aggregation_hash:
            raise RuntimeError("selected checkpoint does not match its committed global-model hash")
    set_model_state(global_model, best["state"])
    artifact = {"model_state_dict": best["state"], "architecture": model_architecture(), "input_shape": [2, 216],
                "class_to_index": class_to_index, "selected_round_by_validation_loss": best["round"],
                "federated_config": config}
    if not he_dp_v2:
        artifact["partition"] = partition
    if bn_mode == "local":
        artifact["batchnorm_mode"] = bn_mode
        artifact["client_batchnorm_states"] = {partition["clients"][k]["client_id"]: v
                                               for k, v in best.get("local_bn", {}).items()}
    torch.save(artifact, model_path)
    if client_validation:
        _write_csv(metrics_dir / f"{exp}_client_validation_history.csv", client_validation)
    if attacker is not None:
        _write_csv(metrics_dir / f"{exp}_attack_round_diagnostics.csv", attack_rounds)
    if track_trajectory:
        np.savez_compressed(metrics_dir / f"{exp}_global_trajectory.npz", states=np.stack(trajectory))
    if defense_cfg is not None:
        _write_csv(metrics_dir / f"{exp}_defense_client_diagnostics.csv", defense_clients)
        _write_csv(metrics_dir / f"{exp}_defense_round_diagnostics.csv", defense_rounds)
    if recorder is not None:
        bc_final = recorder.finalize(root / "ledgers" / f"{exp}.jsonl", anchor_path)
        bc_final["selected_round"] = best["round"]
        bc_final["selected_checkpoint_hash_matches_ledger"] = (
            state_hash(best["state"]) == selected_block.aggregation_hash)
        if not bc_final["selected_checkpoint_hash_matches_ledger"]:
            raise RuntimeError("selected checkpoint hash differs from its finalized ledger record")
        _write_csv(metrics_dir / f"{exp}_blockchain_round_timings.csv", recorder.timings)
        _write_json(metrics_dir / f"{exp}_blockchain_summary.json", {"blockchain_config": bc_cfg, **bc_final})
    if he_session is not None:
        import tenseal
        if he_dp_v2:
            _write_json(root / "private" / f"{exp}_sensitive_diagnostics.json", he_rounds)
            _write_csv(metrics_dir / f"{exp}_he_dp_v2_costs.csv", he_dp_v2_public_rounds)
        else:
            _write_csv(metrics_dir / f"{exp}_he_round_diagnostics.csv", he_rounds)
        _write_json(metrics_dir / f"{exp}_he_setup.json",
                    {"he_config": he_cfg, **he_setup, "tenseal_version": tenseal.__version__})
    if privacy is not None:
        _write_csv(metrics_dir / f"{exp}_dp_round_diagnostics.csv", dp_rounds)
        if dp_clients:
            _write_csv(metrics_dir / f"{exp}_dp_client_update_norms.csv", dp_clients)
        _write_json(metrics_dir / f"{exp}_privacy_accounting.json",
                    {"privacy_config": privacy, "accounting": accounting, "training_seed": seed,
                     "partition_seed": config["partition"]["seed"],
                     "noise_seed_rule": ("private trusted-decryptor seed + round derivation; seed artifact excluded from public metrics"
                                         if he_dp_v2 else "private per-client run seed + round/client derivation; seed artifact excluded from public metrics"
                                         if he_dp else "seed * 1000003 + noise_seed_offset + round"),
                     "selection_note": "Round selection uses only the validation split, which is not client data; "
                                       "selecting among released models is post-processing."})
    _write_csv(metrics_dir / f"{exp}_round_history.csv", history)
    if client_history:
        _write_csv(metrics_dir / f"{exp}_client_round_history.csv", client_history)
    comm = communication_cost(best["state"], participants_per_round, trainable)
    _write_json(metrics_dir / f"{exp}_communication_cost.json", comm)
    if he_dp_v2:
        plain_broadcast_per_client = int(sum(t.numel() * t.element_size() for t in best["state"].values()))
        encrypted_uplink = sum(r["uplink_ciphertext_bytes_total"] for r in he_dp_v2_public_rounds)
        aggregate_ciphertext = sum(r["downlink_ciphertext_bytes"] for r in he_dp_v2_public_rounds)
        model_broadcast = plain_broadcast_per_client * num_clients * len(he_dp_v2_public_rounds)
        _write_json(metrics_dir / f"{exp}_he_dp_v2_communication_cost.json", {
            "measured_client_ciphertext_uplink_bytes": encrypted_uplink,
            "measured_encrypted_aggregate_to_decryptor_bytes": aggregate_ciphertext,
            "estimated_noisy_model_broadcast_bytes": model_broadcast,
            "measured_plus_estimated_total_bytes": encrypted_uplink + aggregate_ciphertext + model_broadcast,
            "public_context_setup_bytes": he_setup["public_context_bytes"],
            "assumptions": ["ciphertext sizes measured from serialized TenSEAL payloads",
                            "global model broadcast estimated as one native state_dict copy per client per round",
                            "transport framing and blockchain are excluded"]})

    central_history = json.loads(Path("results/metrics/centralized_cnn_v1_history.json").read_text(encoding="utf-8"))
    central_epoch = central_history["history"][central_history["best_epoch_by_validation_loss"] - 1]
    reference = {"validation_accuracy": central_epoch["validation_accuracy"],
                 "validation_macro_f1": central_epoch["validation_macro_f1_all_labels"]}
    plots.plot_convergence(history, client_history, best["round"], reference, figures_dir / f"{exp}_convergence.png")
    if client_history:
        plots.plot_client_losses(client_history, figures_dir / f"{exp}_client_losses.png")

    selected_row = history[best["round"]]
    run_summary = {
        "experiment_id": exp, "config": config, "trainable_parameters": trainable,
        "num_clients": num_clients,
        "validation_segments": len(validation), "rounds": int(train_cfg["rounds"]),
        "selected_round_by_validation_loss": best["round"], "selected_round_validation": selected_row,
        "final_round_validation": history[-1], "training_elapsed_seconds": elapsed,
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "model_path": str(model_path).replace("\\", "/"), "test_evaluated": False,
        "centralized_validation_reference_epoch1": reference,
    }
    if not he_dp_v2:
        run_summary["client_segments"] = [len(d) for d in client_data]
    if privacy is not None:
        run_summary["privacy"] = {"config": privacy, "accounting": accounting}
    if he_dp:
        run_summary["integrated_privacy_pipeline"] = {
            "protocol": "client clipping and independent Gaussian noise before CKKS encryption; server performs encrypted weighted sum",
            "privacy_unit": "one client, add/remove adjacency with fixed public sample weights",
            "noise_seed_artifact": str(private_seed_path).replace("\\", "/"),
            "seed_artifact_public": False,
            "caveat": "single-process simulation with a shared client secret key; no network isolation or malicious-party protection",
        }
    if he_dp_v2:
        run_summary["integrated_privacy_pipeline"] = {
            "protocol": "fixed public weights; clip each float client delta; CKKS aggregate; trusted decryptor adds one Gaussian to the clean aggregate",
            "privacy_unit": accounting["privacy_unit"], "adjacency": accounting["adjacency"],
            "trusted_decryptor_sees_clean_aggregate": True,
            "server_sees_individual_plaintext_updates": False,
            "noise_seed_artifact": str(private_seed_path).replace("\\", "/"),
            "seed_artifact_public": False,
            "process_isolation_enforced": False,
            "privacy_scope": "conditional on the trusted-decryptor exclusion and ideal Gaussian mechanism; covers only released noised global models",
        }
    if he_session is not None:
        run_summary["homomorphic_encryption"] = {"config": he_cfg, **he_setup}
    if recorder is not None:
        run_summary["blockchain"] = {"config": bc_cfg, **bc_final}
    if attacker is not None:
        run_summary["attack"] = {"config": attack_cfg, "attacker_index": attacker,
                                 "attacker_client_id": partition["clients"][attacker]["client_id"],
                                 "attacker_segments": partition["clients"][attacker]["segments"],
                                 "non_finite_global_rounds": int(sum(not r["global_state_finite"] for r in attack_rounds))}
    if defense_cfg is not None:
        run_summary["defense"] = {"config": defense_cfg,
                                  "defense_seconds_total": float(sum(r["defense_seconds"] for r in defense_rounds))}
    if bn_mode == "local":
        run_summary["batchnorm_mode"] = bn_mode
        run_summary["batchnorm_state_keys"] = bn_keys
    if evaluate_test:
        result, test_summary = evaluate_test_once(global_model, split_root / "test", class_names, root, exp,
                                                  device, eval_bs, overwrite)
        plots.plot_confusion_matrix(result["confusion_matrix"], class_names,
                                    ("FedBN-style evaluation model" if bn_mode == "local"
                                     else "HE+DP-FedAvg global model" if he_dp
                                     else "DP-FedAvg global model" if privacy is not None
                                     else "HE-FedAvg global model" if he_session is not None
                                     else ("Defended FedAvg" if defense_cfg is not None else "FedAvg")
                                     + (f" under {attack_cfg['name']} attack" if attacker is not None else "")
                                     if attacker is not None or defense_cfg is not None
                                     else "FedAvg global model")
                                    + f" (round {best['round']}) — held-out test confusion matrix",
                                    figures_dir / f"{exp}_confusion_matrix.png")
        aggregate_rows, per_class_rows = compare_with_centralized(result["aggregate"], result["per_class"], label=exp)
        _write_csv(tables_dir / f"{comparison_prefix}centralized_vs_fedavg_aggregate.csv", aggregate_rows)
        _write_csv(tables_dir / f"{comparison_prefix}centralized_vs_fedavg_per_class.csv", per_class_rows)
        run_summary.update({"test_evaluated": True, "test_metrics": test_summary["metrics"]})
    _write_json(metrics_dir / f"{exp}_run_summary.json", run_summary)
    return run_summary


if __name__ == "__main__":
    summary = run_experiment()
    print(json.dumps({k: summary[k] for k in ("selected_round_by_validation_loss", "test_evaluated")}, indent=2))
