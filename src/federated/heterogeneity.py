"""Descriptive label-heterogeneity measures for a client partition (no balancing)."""
from itertools import combinations

import numpy as np


def jensen_shannon(p, q):
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint support)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def client_heterogeneity(partition, class_names):
    """Per-client rows plus partition-level summary statistics."""
    clients = partition["clients"]
    matrix = np.array([[c["class_counts"].get(name, 0) for name in class_names] for c in clients], dtype=float)
    pooled = matrix.sum(axis=0)
    rows = []
    for client, counts in zip(clients, matrix):
        n = counts.sum()
        dominant = int(np.argmax(counts))
        nonmajor = [(class_names[i], int(counts[i])) for i in np.argsort(-counts) if counts[i] > 0 and i != dominant]
        rows.append({
            "client_id": client["client_id"], "groups": len(client["groups"]), "segments": int(n),
            "classes_present": int((counts > 0).sum()),
            "dominant_class": class_names[dominant], "dominant_share": float(counts[dominant] / n),
            "second_class": nonmajor[0][0] if nonmajor else "",
            "second_share": float(nonmajor[0][1] / n) if nonmajor else 0.0,
            "missing_classes": " ".join(name for name, c in zip(class_names, counts) if c == 0),
            "js_divergence_vs_pooled_bits": jensen_shannon(counts, pooled),
        })
    clients_per_class = {name: int((matrix[:, i] > 0).sum()) for i, name in enumerate(class_names)}
    pairwise = [jensen_shannon(matrix[a], matrix[b]) for a, b in combinations(range(len(clients)), 2)]
    sizes = matrix.sum(axis=1)
    summary = {
        "client_class_absences": int(sum((matrix[:, i] == 0).sum() for i in range(len(class_names)) if pooled[i] > 0)),
        "classes_on_all_clients": int(sum(v == len(clients) for v in clients_per_class.values())),
        "classes_on_one_client": int(sum(v == 1 for v in clients_per_class.values())),
        "clients_per_class": clients_per_class,
        "mean_js_divergence_vs_pooled_bits": float(np.mean([r["js_divergence_vs_pooled_bits"] for r in rows])),
        "mean_pairwise_js_divergence_bits": float(np.mean(pairwise)) if pairwise else 0.0,
        "max_pairwise_js_divergence_bits": float(np.max(pairwise)) if pairwise else 0.0,
        "client_segments_min": int(sizes.min()), "client_segments_max": int(sizes.max()),
        "client_segments_cv": float(sizes.std(ddof=0) / sizes.mean()),
    }
    return rows, summary
