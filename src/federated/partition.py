"""Group-level assignment of the existing training split to simulated FL clients.

Only `train/` shards listed as training records in `split_metadata.json` may be
assigned. Complete inferred subject-equivalence groups are the assignment unit;
individual ECG windows are never distributed across clients.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np


def _load_split_metadata(split_root):
    return json.loads((Path(split_root) / "split_metadata.json").read_text(encoding="utf-8"))


def discover_training_groups(split_root):
    """Return `{group: {"records", "segments", "class_counts"}}` for the training split only.

    Raises if the `train/` shards disagree with the saved split metadata or if any
    training record/group is also listed for validation or test.
    """
    root = Path(split_root)
    metadata = _load_split_metadata(root)
    splits = metadata["splits"]
    train_records = set(splits["train"]["records"])
    train_groups = set(splits["train"]["groups"])
    held_out_records = set(splits["validation"]["records"]) | set(splits["test"]["records"])
    held_out_groups = set(splits["validation"]["groups"]) | set(splits["test"]["groups"])
    if train_records & held_out_records or train_groups & held_out_groups:
        raise ValueError("Split metadata lists a training record/group in validation or test")

    paths = sorted((root / "train").glob("*.npz"))
    if {p.stem for p in paths} != train_records:
        raise ValueError("train/ shards do not match the training records in split metadata")

    groups = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            record_ids = set(map(str, stored["record_ids"]))
            group_ids = set(map(str, stored["patient_groups"]))
            labels = stored["labels"].astype(str)
        if record_ids != {path.stem} or len(group_ids) != 1:
            raise ValueError(f"{path.name}: expected one record and one group per shard")
        group = group_ids.pop()
        if group not in train_groups:
            raise ValueError(f"{path.name}: group {group} is not a training group")
        entry = groups.setdefault(group, {"records": [], "segments": 0, "class_counts": Counter()})
        entry["records"].append(path.stem)
        entry["segments"] += len(labels)
        entry["class_counts"].update(labels.tolist())
    if set(groups) != train_groups:
        raise ValueError("Training groups found in shards differ from split metadata")
    return groups


def assign_groups_to_clients(group_ids, num_clients, seed, strategy="random_group_equal_count"):
    """Assign whole groups to clients reproducibly.

    `random_group_equal_count`: sort the group IDs, permute them with
    `numpy.random.default_rng(seed)`, and cut the permutation into `num_clients`
    contiguous chunks whose group counts differ by at most one. Labels and segment
    counts are not used, so the natural (non-IID) class mix of each record is kept.
    """
    if strategy != "random_group_equal_count":
        raise ValueError(f"Unsupported partition strategy {strategy!r}")
    ordered = sorted(group_ids)
    if not 1 <= num_clients <= len(ordered):
        raise ValueError("num_clients must be between 1 and the number of training groups")
    permuted = np.random.default_rng(seed).permutation(len(ordered))
    chunks = np.array_split(permuted, num_clients)
    return [sorted(ordered[i] for i in chunk) for chunk in chunks]


def build_client_partition(split_root, num_clients, seed, strategy="random_group_equal_count",
                           min_client_segments=1):
    """Build and validate the client partition from the training split."""
    groups = discover_training_groups(split_root)
    assignment = assign_groups_to_clients(list(groups), num_clients, seed, strategy)
    clients = []
    for index, client_groups in enumerate(assignment):
        counts = Counter()
        for group in client_groups:
            counts.update(groups[group]["class_counts"])
        records = sorted(r for group in client_groups for r in groups[group]["records"])
        clients.append({
            "client_id": f"client_{index:02d}",
            "groups": client_groups,
            "records": records,
            "segments": int(sum(counts.values())),
            "class_counts": dict(sorted(counts.items())),
        })
    partition = {"strategy": strategy, "seed": int(seed), "num_clients": int(num_clients),
                 "min_client_segments": int(min_client_segments), "clients": clients}
    validate_client_partition(partition, split_root)
    return partition


def validate_client_partition(partition, split_root):
    """Check client disjointness, full training coverage, and absence of held-out data."""
    metadata = _load_split_metadata(split_root)
    splits = metadata["splits"]
    held_out_records = set(splits["validation"]["records"]) | set(splits["test"]["records"])
    held_out_groups = set(splits["validation"]["groups"]) | set(splits["test"]["groups"])
    seen_groups, seen_records = Counter(), Counter()
    for client in partition["clients"]:
        seen_groups.update(client["groups"])
        seen_records.update(client["records"])
        if client["segments"] < partition["min_client_segments"]:
            raise ValueError(f"{client['client_id']} has {client['segments']} segments, "
                             f"below min_client_segments={partition['min_client_segments']}")
    if any(n > 1 for n in seen_groups.values()) or any(n > 1 for n in seen_records.values()):
        raise ValueError("A group or record is assigned to more than one client")
    if set(seen_groups) & held_out_groups or set(seen_records) & held_out_records:
        raise ValueError("A validation/test group or record was assigned to a client")
    if set(seen_groups) != set(splits["train"]["groups"]) or set(seen_records) != set(splits["train"]["records"]):
        raise ValueError("Client partition does not cover every training group exactly once")
    total = sum(client["segments"] for client in partition["clients"])
    if total != int(splits["train"]["segment_count"]):
        raise ValueError(f"Client segments ({total}) differ from training segments")
    return True


def client_distribution_tables(partition, class_names):
    """Return (summary_rows, class_count_rows, class_proportion_rows) for reporting."""
    total = sum(c["segments"] for c in partition["clients"])
    summary, counts, proportions = [], [], []
    for client in partition["clients"]:
        n = client["segments"]
        cc = client["class_counts"]
        summary.append({
            "client_id": client["client_id"], "groups": len(client["groups"]),
            "records": len(client["records"]), "segments": n,
            "share_of_training_segments": n / total,
            "classes_present": sum(1 for c in class_names if cc.get(c, 0) > 0),
            "classes_absent": " ".join(c for c in class_names if cc.get(c, 0) == 0),
            "record_ids": " ".join(client["records"]),
        })
        counts.append({"client_id": client["client_id"], **{c: int(cc.get(c, 0)) for c in class_names}, "total": n})
        proportions.append({"client_id": client["client_id"], **{c: cc.get(c, 0) / n for c in class_names}})
    return summary, counts, proportions
