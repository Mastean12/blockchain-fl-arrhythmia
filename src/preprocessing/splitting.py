"""Reproducible record-group splits for the processed MIT-BIH segments.

This module separates complete record groups. It does not stratify or balance
labels, and does not split individual ECG windows across partitions.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import shutil
import tempfile

import numpy as np

from .storage import load_record, write_csv, write_json

SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_FIELDS = {
    "segments", "labels", "source_symbols", "record_ids", "patient_groups",
    "annotation_samples", "starts", "ends", "sampling_frequency_hz", "channel_names",
}


def _largest_remainder_counts(n_groups, proportions):
    raw = {name: n_groups * float(proportions[name]) for name in SPLIT_NAMES}
    counts = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    remaining = n_groups - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (raw[name] - counts[name], -SPLIT_NAMES.index(name)), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def load_split_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    proportions = config["target_proportions"]
    if set(proportions) != set(SPLIT_NAMES):
        raise ValueError(f"target_proportions must define {SPLIT_NAMES}")
    if any(float(value) <= 0 for value in proportions.values()) or not np.isclose(sum(proportions.values()), 1.0):
        raise ValueError("Split proportions must be positive and sum to 1")
    if int(config["random_seed"]) < 0:
        raise ValueError("random_seed must be nonnegative")
    return config


def inspect_processed_dataset(processed_dir, valid_labels, expected_shape=(216, 2), expected_groups=None):
    """Read shards and verify schemas, shapes, labels, finite data, and group identity."""
    root = Path(processed_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing processed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_records = [str(x) for x in manifest["records"]]
    shards = sorted(root.glob("*.npz"))
    by_record = {p.stem: p for p in shards}
    missing = sorted(set(expected_records) - set(by_record))
    extras = sorted(set(by_record) - set(expected_records))
    if missing or extras or len(shards) != len(expected_records):
        raise ValueError(f"Processed record/shard mismatch; missing={missing}, extras={extras}")

    records = {}
    record_groups = {}
    group_records = {}
    total = 0
    for record_id in expected_records:
        data = load_record(by_record[record_id])
        absent_fields = EXPECTED_FIELDS - set(data)
        if absent_fields:
            raise ValueError(f"{record_id}: missing metadata fields {sorted(absent_fields)}")
        n = len(data["labels"])
        if data["segments"].shape != (n, *expected_shape):
            raise ValueError(f"{record_id}: invalid segment dimensions {data['segments'].shape}")
        if not np.isfinite(data["segments"]).all():
            raise ValueError(f"{record_id}: non-finite segment values")
        if n == 0:
            raise ValueError(f"{record_id}: empty processed record; no record is silently excluded")
        for key in ("source_symbols", "record_ids", "patient_groups", "annotation_samples", "starts", "ends"):
            if len(data[key]) != n:
                raise ValueError(f"{record_id}: metadata length mismatch for {key}")
        if set(map(str, data["labels"])) - set(valid_labels):
            raise ValueError(f"{record_id}: invalid labels {sorted(set(map(str, data['labels'])) - set(valid_labels))}")
        if set(map(str, data["record_ids"])) != {record_id}:
            raise ValueError(f"{record_id}: record_ids field disagrees with shard name")
        groups = set(map(str, data["patient_groups"]))
        if len(groups) != 1:
            raise ValueError(f"{record_id}: expected one stable group key, found {groups}")
        group = groups.pop()
        if expected_groups is not None:
            expected_group = expected_groups.get(record_id, f"record_{record_id}")
            if group != expected_group:
                raise ValueError(f"{record_id}: group metadata {group!r} != configured value {expected_group!r}")
        records[record_id] = {"path": by_record[record_id], "data": data, "segments": n, "group": group}
        record_groups[record_id] = group
        group_records.setdefault(group, []).append(record_id)
        total += n
    if total != int(manifest["segment_count"]):
        raise ValueError(f"Segment total mismatch: shards={total}, manifest={manifest['segment_count']}")
    return {"manifest": manifest, "records": records, "record_groups": record_groups,
            "group_records": group_records, "segment_count": total}


def plan_group_splits(group_records, proportions, seed):
    """Assign whole, sorted group keys using a seeded permutation and largest remainder."""
    groups = sorted(group_records)
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("At least three distinct record groups are needed for three splits")
    counts = _largest_remainder_counts(len(groups), proportions)
    if any(counts[name] == 0 for name in SPLIT_NAMES):
        raise ValueError("Target proportions yield an empty split at this group count")
    permutation = np.random.default_rng(int(seed)).permutation(len(groups))
    shuffled = [groups[i] for i in permutation]
    assignment = {}
    start = 0
    for name in SPLIT_NAMES:
        for group in shuffled[start:start + counts[name]]:
            assignment[group] = name
        start += counts[name]
    return assignment


def _sample_digest(segment, label):
    digest = hashlib.sha256()
    digest.update(str(label).encode("utf-8"))
    digest.update(np.ascontiguousarray(segment).view(np.uint8))
    return digest.digest()


def validate_split_membership(split_records, records, valid_labels, expected_shape=(216, 2),
                              check_content_duplicates=True):
    """Validate split disjointness, shard data, schema, and exact cross-split duplicates."""
    record_sets = {name: set(split_records[name]) for name in SPLIT_NAMES}
    group_sets = {name: {records[r]["group"] for r in split_records[name]} for name in SPLIT_NAMES}
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            if record_sets[left] & record_sets[right]:
                raise ValueError(f"Record overlap: {left} and {right}")
            if group_sets[left] & group_sets[right]:
                raise ValueError(f"Group overlap: {left} and {right}")

    provenance_seen = {}
    content_seen = {} if check_content_duplicates else None
    class_counts = {name: Counter() for name in SPLIT_NAMES}
    segment_counts = {name: 0 for name in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for record_id in split_records[split]:
            if record_id not in records:
                raise ValueError(f"Unknown record {record_id} in {split}")
            data = records[record_id]["data"]
            n = len(data["labels"])
            required = EXPECTED_FIELDS - set(data)
            if required:
                raise ValueError(f"{record_id}: missing fields {sorted(required)}")
            if data["segments"].shape != (n, *expected_shape):
                raise ValueError(f"{record_id}: invalid signal dimensions")
            if not np.isfinite(data["segments"]).all():
                raise ValueError(f"{record_id}: NaN/Inf in ECG segments")
            if set(map(str, data["labels"])) - set(valid_labels):
                raise ValueError(f"{record_id}: invalid class label")
            for key in ("source_symbols", "record_ids", "patient_groups", "annotation_samples", "starts", "ends"):
                if len(data[key]) != n:
                    raise ValueError(f"{record_id}: metadata length mismatch for {key}")
            if set(map(str, data["record_ids"])) != {record_id}:
                raise ValueError(f"{record_id}: record provenance mismatch")
            if set(map(str, data["patient_groups"])) != {records[record_id]["group"]}:
                raise ValueError(f"{record_id}: group provenance mismatch")
            class_counts[split].update(map(str, data["labels"]))
            segment_counts[split] += n
            for i in range(n):
                provenance = (record_id, int(data["annotation_samples"][i]), str(data["labels"][i]))
                previous = provenance_seen.get(provenance)
                if previous is not None and previous != split:
                    raise ValueError(f"Duplicate annotation sample appears in {previous} and {split}: {provenance}")
                provenance_seen[provenance] = split
                if check_content_duplicates:
                    key = _sample_digest(data["segments"][i], data["labels"][i])
                    previous = content_seen.get(key)
                    if previous is not None and previous != split:
                        raise ValueError(f"Duplicate segment content appears in {previous} and {split}")
                    content_seen[key] = split
    return {"record_sets": record_sets, "group_sets": group_sets,
            "class_counts": class_counts, "segment_counts": segment_counts}


def build_splits(processed_dir, output_dir, split_config_path, label_config_path,
                 expected_shape=(216, 2), overwrite=False):
    """Build group-disjoint split shards and metadata; refuse overwrite by default."""
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    split_cfg = load_split_config(split_config_path)
    labels_cfg = json.loads(Path(label_config_path).read_text(encoding="utf-8"))
    valid_labels = set(labels_cfg["labels"]["beat_symbols"].values())
    expected_groups = labels_cfg["labels"].get("patient_group_overrides", {})
    inspected = inspect_processed_dataset(processed_dir, valid_labels, expected_shape, expected_groups)
    assignment = plan_group_splits(inspected["group_records"], split_cfg["target_proportions"], split_cfg["random_seed"])
    if assignment != plan_group_splits(inspected["group_records"], split_cfg["target_proportions"], split_cfg["random_seed"]):
        raise RuntimeError("Seeded split assignment is not reproducible")
    split_records = {name: sorted(r for r, group in inspected["record_groups"].items() if assignment[group] == name) for name in SPLIT_NAMES}
    validation = validate_split_membership(split_records, inspected["records"], valid_labels, expected_shape)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing split output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    try:
        total_segments = inspected["segment_count"]
        split_details = {}
        for split in SPLIT_NAMES:
            destination = stage / split
            destination.mkdir()
            for record_id in split_records[split]:
                shutil.copy2(inspected["records"][record_id]["path"], destination / f"{record_id}.npz")
            n_segments = validation["segment_counts"][split]
            dist = dict(sorted(validation["class_counts"][split].items()))
            split_details[split] = {
                "records": split_records[split],
                "record_count": len(split_records[split]),
                "groups": sorted(validation["group_sets"][split]),
                "group_count": len(validation["group_sets"][split]),
                "segment_count": n_segments,
                "segment_proportion": n_segments / total_segments,
                "record_proportion": len(split_records[split]) / len(inspected["records"]),
                "class_distribution": dist,
                "absent_classes": sorted(valid_labels - set(dist)),
            }
            write_csv(stage / f"{split}_class_distribution.csv",
                      [{"label": label, "count": count} for label, count in sorted(validation["class_counts"][split].items())],
                      ["label", "count"])
        overlap = {f"{a}_vs_{b}": {"record_ids": sorted(validation["record_sets"][a] & validation["record_sets"][b]),
                                    "group_ids": sorted(validation["group_sets"][a] & validation["group_sets"][b])}
                   for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]}
        metadata = {
            "dataset": inspected["manifest"]["dataset"],
            "dataset_version": inspected["manifest"]["dataset_version"],
            "source_manifest": "data/processed/mitbih_v1/manifest.json",
            "strategy": "seeded random assignment of complete record groups; labels are not used to rebalance or stratify",
            "separation_unit": "patient_group key from Day 3 processed shards; keys are record-derived pseudonyms except the documented 201/202 shared-tape group",
            "random_seed": int(split_cfg["random_seed"]),
            "target_proportions": split_cfg["target_proportions"],
            "total_records": len(inspected["records"]),
            "total_groups": len(inspected["group_records"]),
            "total_segments": total_segments,
            "segment_shape": list(expected_shape),
            "valid_labels": sorted(valid_labels),
            "splits": split_details,
            "overlap_checks": overlap,
            "duplicate_sample_check": "exact ECG segment bytes plus label, checked across split partitions",
            "exclusions": [],
            "limitations": [
                "MIT-BIH provides 48 records from 47 subjects, but a complete record-to-subject mapping is not present in the processed files.",
                "Records 201 and 202 are grouped because the PhysioNet record notes identify the same source analog tape; other group keys are record-derived and do not assert distinct identities for unknown patients.",
                "Accordingly, this guarantees no split overlap by available record/group key, but does not establish complete person-level separation beyond the documented pair.",
                "Random group allocation is not label-stratified; rare classes may be absent or sparse in validation/test. No records or labels are removed or rebalanced."
            ],
            "validation": {"record_overlap": False, "group_overlap": False, "duplicate_provenance_across_splits": False,
                           "duplicate_content_across_splits": False, "valid_labels": True, "finite_values": True,
                           "expected_dimensions": list(expected_shape), "metadata_fields": sorted(EXPECTED_FIELDS),
                           "same_seed_reproducible": True}
        }
        write_json(stage / "split_metadata.json", metadata)
        shutil.copy2(Path(split_config_path), stage / "split_config.json")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return metadata


def validate_saved_splits(split_dir, label_config_path, expected_shape=(216, 2)):
    """Reopen saved split shards and rerun all disjointness/schema checks."""
    root = Path(split_dir)
    metadata = json.loads((root / "split_metadata.json").read_text(encoding="utf-8"))
    cfg = json.loads(Path(label_config_path).read_text(encoding="utf-8"))
    valid_labels = set(cfg["labels"]["beat_symbols"].values())
    expected_groups = cfg["labels"].get("patient_group_overrides", {})
    records = {}
    split_records = {}
    for split in SPLIT_NAMES:
        split_records[split] = sorted(p.stem for p in (root / split).glob("*.npz"))
        for record_id in split_records[split]:
            data = load_record(root / split / f"{record_id}.npz")
            group_vals = set(map(str, data["patient_groups"]))
            if len(group_vals) != 1:
                raise ValueError(f"{record_id}: multiple group IDs in saved shard")
            group = group_vals.pop()
            expected_group = expected_groups.get(record_id, f"record_{record_id}")
            if group != expected_group:
                raise ValueError(f"{record_id}: group metadata {group!r} != configured value {expected_group!r}")
            records[record_id] = {"data": data, "group": group}
    actual = validate_split_membership(split_records, records, valid_labels, expected_shape)
    if actual["segment_counts"] != {s: metadata["splits"][s]["segment_count"] for s in SPLIT_NAMES}:
        raise ValueError("Saved split segment totals disagree with metadata")
    return actual
