"""Versioned SVDB-to-MIT-BIH-vocabulary ECG preprocessing and record splits.

Raw SVDB files are read only. The importer deliberately fails on any annotation
symbol not covered by the versioned mapping/exclusion lists.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import scipy
from scipy.signal import resample_poly
import wfdb

from .config import load_config
from .ecg_io import listed_records, verify_record_files
from .filtering import filter_signal
from .normalization import normalize_segments
from .segmentation import segment_record
from .splitting import (SPLIT_NAMES, inspect_processed_dataset, load_split_config,
                        plan_group_splits, validate_split_membership)
from .storage import save_record, write_csv, write_json
from .validation import validate_batch


def classify_symbols(symbols, config):
    """Validate source annotations and count mapped, excluded-beat, and event codes."""
    labels = config.raw["labels"]
    supported = set(labels["beat_symbols"])
    excluded_beats = set(labels["excluded_beat_symbols"])
    nonbeats = set(labels["nonbeat_symbols"])
    counts = Counter(map(str, symbols))
    unknown = set(counts) - supported - excluded_beats - nonbeats
    if unknown:
        raise ValueError(f"Unmapped SVDB annotation symbols: {sorted(unknown)}")
    return {
        "source_symbol_counts": dict(sorted(counts.items())),
        "mapped_beat_counts": dict(sorted((s, counts[s]) for s in supported if counts[s])),
        "excluded_beat_counts": dict(sorted((s, counts[s]) for s in excluded_beats if counts[s])),
        "nonbeat_event_counts": dict(sorted((s, counts[s]) for s in nonbeats if counts[s])),
    }


def resample_signal_and_annotations(signal, samples, source_fs, target_fs, up, down):
    """Resample ECG and map annotation sample indices using the same rational rate."""
    if int(source_fs) * int(up) != int(target_fs) * int(down):
        raise ValueError("Resampling ratio disagrees with sampling frequencies")
    x = np.asarray(signal, dtype=np.float64)
    ann = np.asarray(samples, dtype=np.int64)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(ann < 0) or np.any(ann >= len(x)):
        raise ValueError("Invalid source signal or annotation positions")
    y = resample_poly(x, int(up), int(down), axis=0)
    mapped = np.rint(ann * (int(target_fs) / int(source_fs))).astype(np.int64)
    valid = mapped < len(y)
    return y, mapped[valid], valid


def process_svdb(raw_dir, output_dir, config_path="configs/preprocessing/svdb_v1.json",
                 record_ids=None, checksums=True):
    """Process selected SVDB records into the project's 360-Hz, 216x2 format."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    config = load_config(config_path)
    ids = listed_records(raw_dir)
    chosen = list(record_ids) if record_ids is not None else ids
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(ids):
        raise ValueError("Invalid requested SVDB record subset")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    source_fs = int(config.raw["dataset"]["source_sampling_frequency_hz"])
    target_fs = int(config.raw["dataset"]["target_sampling_frequency_hz"])
    up, down = int(config.raw["resampling"]["up"]), int(config.raw["resampling"]["down"])
    if source_fs * up != target_fs * down:
        raise ValueError("Resampling ratio disagrees with configured source and target rates")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    distribution, annotation_distribution, summaries, total = Counter(), Counter(), [], 0
    terminal_boundary_annotations = 0
    try:
        checksum_file = raw_dir / "SHA256SUMS.txt"
        for rid in chosen:
            verify_record_files(raw_dir, rid, checksums)
            record = wfdb.rdrecord(str(raw_dir / rid), physical=True)
            annotation = wfdb.rdann(str(raw_dir / rid), config.raw["dataset"]["annotation_extension"])
            signal = np.asarray(record.p_signal, dtype=np.float64)
            if record.fs != source_fs or signal.ndim != 2 or signal.shape[1] != config.n_channels:
                raise ValueError(f"{rid}: unexpected SVDB signal format ({record.fs}, {signal.shape})")
            if not np.isfinite(signal).all():
                raise ValueError(f"{rid}: non-finite signal")
            samples = np.asarray(annotation.sample, dtype=np.int64)
            symbols = list(annotation.symbol)
            audit = classify_symbols(symbols, config)
            annotation_distribution.update(audit["source_symbol_counts"])
            if len(samples) != len(symbols) or np.any(samples < 0):
                raise ValueError(f"{rid}: invalid annotation locations")
            # SVDB record 860 has one final N annotation exactly at sample
            # signal_length. Preserve it in the symbol audit, but no beat window
            # can be formed because no signal sample exists at that location.
            terminal = samples == len(signal)
            if np.any(samples > len(signal)):
                raise ValueError(f"{rid}: annotation location exceeds signal boundary")
            terminal_count = int(terminal.sum())
            terminal_boundary_annotations += terminal_count
            samples = samples[~terminal]
            symbols = [symbol for symbol, keep in zip(symbols, ~terminal) if keep]

            resampled, target_samples, keep = resample_signal_and_annotations(
                signal, samples, source_fs, target_fs, up, down)
            symbols = [s for s, ok in zip(symbols, keep) if ok]
            filtered = filter_signal(resampled, target_fs, config)
            group = f"record_{rid}"  # no released subject identifier is available
            batch = segment_record(filtered, target_samples, symbols, rid, group, config)
            batch.segments = normalize_segments(batch.segments, config)
            validate_batch(batch, config)
            save_record(stage / f"{rid}.npz", batch, target_fs, list(record.sig_name))
            distribution.update(batch.labels.tolist())
            total += len(batch.labels)
            summaries.append({"record_id": rid, "patient_group": group,
                              "source_samples": len(signal), "target_samples": len(resampled),
                              "annotations": len(annotation.sample), "segments": len(batch.labels),
                              "edge_dropped": batch.edge_dropped,
                              "terminal_boundary_excluded_count": terminal_count,
                              "excluded_beat_count": int(sum(audit["excluded_beat_counts"].values())),
                              "nonbeat_event_count": int(sum(audit["nonbeat_event_counts"].values())),
                              "source_symbol_counts": json.dumps(audit["source_symbol_counts"], sort_keys=True),
                              "mapped_beat_counts": json.dumps(audit["mapped_beat_counts"], sort_keys=True),
                              "excluded_beat_counts": json.dumps(audit["excluded_beat_counts"], sort_keys=True),
                              "nonbeat_event_counts": json.dumps(audit["nonbeat_event_counts"], sort_keys=True)})

        manifest = {
            "dataset": config.raw["dataset"]["name"],
            "dataset_version": config.raw["dataset"]["version"],
            "license": config.raw["dataset"]["license"],
            "citation_doi": config.raw["dataset"]["citation_doi"],
            "config": config.raw, "config_path": str(config.source),
            "records": chosen, "record_count": len(chosen), "segment_count": total,
            "terminal_boundary_excluded_count": terminal_boundary_annotations,
            "segment_shape": [config.segment_length, config.n_channels],
            "class_distribution": dict(sorted(distribution.items())),
            "source_annotation_symbol_counts": dict(sorted(annotation_distribution.items())),
            "sampling_frequency_hz": target_fs, "no_split_created": True,
            "no_class_balancing": True,
            "identity_limitation": "No patient identifiers are released; record-derived groups do not guarantee person-level separation.",
            "checksum_manifest_present": checksum_file.is_file(),
            "software": {"numpy": np.__version__, "scipy": scipy.__version__, "wfdb": wfdb.__version__},
        }
        write_json(stage / "manifest.json", manifest)
        write_csv(stage / "class_distribution.csv",
                  [{"label": k, "count": v} for k, v in sorted(distribution.items())], ["label", "count"])
        write_csv(stage / "record_summary.csv", summaries,
                  ["record_id", "patient_group", "source_samples", "target_samples", "annotations",
                   "segments", "edge_dropped", "terminal_boundary_excluded_count",
                   "excluded_beat_count", "nonbeat_event_count",
                   "source_symbol_counts", "mapped_beat_counts", "excluded_beat_counts", "nonbeat_event_counts"])
        shutil.copy2(config.source, stage / "preprocessing_config.json")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_svdb_splits(processed_dir, output_dir,
                      split_config_path="configs/preprocessing/svdb_split_v1.json",
                      label_config_path="configs/preprocessing/svdb_v1.json", overwrite=False):
    """Create reproducible record-disjoint splits; never represent them as patient-disjoint."""
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    split_cfg = load_split_config(split_config_path)
    labels_cfg = json.loads(Path(label_config_path).read_text(encoding="utf-8"))
    valid_labels = set(labels_cfg["labels"]["beat_symbols"].values())
    inspected = inspect_processed_dataset(processed_dir, valid_labels, expected_shape=(216, 2))
    assignment = plan_group_splits(inspected["group_records"], split_cfg["target_proportions"], split_cfg["random_seed"])
    records_by_split = {name: sorted(r for r, group in inspected["record_groups"].items()
                                     if assignment[group] == name) for name in SPLIT_NAMES}
    validation = validate_split_membership(records_by_split, inspected["records"], valid_labels, (216, 2))
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing split output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    try:
        details = {}
        for name in SPLIT_NAMES:
            dest = stage / name
            dest.mkdir()
            for rid in records_by_split[name]:
                shutil.copy2(inspected["records"][rid]["path"], dest / f"{rid}.npz")
            details[name] = {"records": records_by_split[name], "record_count": len(records_by_split[name]),
                             "group_count": len(validation["group_sets"][name]),
                             "segment_count": validation["segment_counts"][name],
                             "class_counts": dict(sorted(validation["class_counts"][name].items()))}
        metadata = {"dataset": "MIT-BIH Supraventricular Arrhythmia Database",
                    "dataset_version": "PhysioNet 1.0.0", "split_config": split_cfg,
                    "seed": int(split_cfg["random_seed"]), "assignment_unit": "record",
                    "patient_disjoint_claim": False,
                    "limitations": ["SVDB does not publish record-level patient identifiers.",
                                    "Cross-dataset identity overlap with MIT-BIH cannot be checked."],
                    "splits": details,
                    "validation": {"record_overlap": False, "available_group_overlap": False,
                                   "same_seed_reproducible": True, "content_duplicate_check": True}}
        write_json(stage / "split_metadata.json", metadata)
        shutil.copy2(Path(split_config_path), stage / "split_config.json")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(stage, output_dir)
        return metadata
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_svdb_splits(processed_dir, split_dir,
                         split_config_path="configs/preprocessing/svdb_split_v1.json",
                         label_config_path="configs/preprocessing/svdb_v1.json"):
    """Revalidate split provenance, deterministic membership, and copied shard bytes."""
    processed_dir, split_dir = Path(processed_dir), Path(split_dir)
    split_cfg = load_split_config(split_config_path)
    label_cfg = load_config(label_config_path)
    valid_labels = set(label_cfg.raw["labels"]["beat_symbols"].values())
    inspected = inspect_processed_dataset(processed_dir, valid_labels, expected_shape=(216, 2))
    if inspected["manifest"].get("config") != label_cfg.raw:
        raise ValueError("SVDB processed manifest config differs from the selected preprocessing config")
    metadata = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset") != label_cfg.raw["dataset"]["name"]:
        raise ValueError("SVDB split metadata dataset name does not match preprocessing config")
    if metadata.get("dataset_version") != label_cfg.raw["dataset"]["version"]:
        raise ValueError("SVDB split metadata version does not match preprocessing config")
    if (metadata.get("split_config") != split_cfg or metadata.get("seed") != split_cfg["random_seed"]
            or metadata.get("assignment_unit") != "record"
            or metadata.get("patient_disjoint_claim") is not False):
        raise ValueError("SVDB split metadata does not match the configured record-level protocol")
    assignment = plan_group_splits(inspected["group_records"], split_cfg["target_proportions"],
                                   split_cfg["random_seed"])
    expected_records = {
        name: sorted(record for record, group in inspected["record_groups"].items()
                     if assignment[group] == name)
        for name in SPLIT_NAMES
    }
    actual_records = {name: metadata["splits"][name]["records"] for name in SPLIT_NAMES}
    if actual_records != expected_records:
        raise ValueError("SVDB split records do not match the deterministic configured assignment")
    membership = validate_split_membership(actual_records, inspected["records"], valid_labels, (216, 2))
    if set().union(*(membership["record_sets"][name] for name in SPLIT_NAMES)) != set(inspected["records"]):
        raise ValueError("SVDB split membership omits or adds processed records")

    for name in SPLIT_NAMES:
        split_root = split_dir / name
        expected_names = {f"{record}.npz" for record in actual_records[name]}
        if {path.name for path in split_root.glob("*.npz")} != expected_names:
            raise ValueError(f"SVDB {name} shard files do not match split metadata")
        for record in actual_records[name]:
            source = inspected["records"][record]["path"]
            copied = split_root / source.name
            source_hash = hashlib.sha256(source.read_bytes()).digest()
            copied_hash = hashlib.sha256(copied.read_bytes()).digest()
            if source_hash != copied_hash:
                raise ValueError(f"SVDB {name} shard {record} differs from its processed source")
        details = metadata["splits"][name]
        expected_classes = dict(sorted(membership["class_counts"][name].items()))
        if (details["record_count"] != len(actual_records[name])
                or details["group_count"] != len(membership["group_sets"][name])
                or details["segment_count"] != membership["segment_counts"][name]
                or details["class_counts"] != expected_classes):
            raise ValueError(f"SVDB {name} metadata totals differ from validated shards")
    return {"record_sets": membership["record_sets"], "group_sets": membership["group_sets"],
            "class_counts": membership["class_counts"], "segment_counts": membership["segment_counts"],
            "shard_copies_match": True, "patient_disjoint_claim": False}
