"""Figures for the FedAvg baseline (static PNG, matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

FEDAVG_COLOR = "#2a78d6"     # categorical slot 1
REFERENCE_COLOR = "#eb6834"  # categorical slot 2
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5e0"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_convergence(history, client_history, selected_round, centralized_reference, path):
    """2x2 panels: validation accuracy, macro-F1, weighted-F1 and losses by round.

    `centralized_reference` holds the frozen centralized checkpoint's validation
    accuracy/macro-F1 (epoch 1), drawn as a dashed horizontal reference.
    """
    rounds = [row["round"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    panels = [("validation_accuracy", "Validation accuracy", centralized_reference.get("validation_accuracy")),
              ("validation_macro_f1", "Validation macro-F1 (15 classes)", centralized_reference.get("validation_macro_f1")),
              ("validation_weighted_f1", "Validation weighted-F1", None)]
    for ax, (key, title, reference) in zip(axes.flat, panels):
        _style(ax)
        ax.plot(rounds, [row[key] for row in history], color=FEDAVG_COLOR, linewidth=2,
                marker="o", markersize=4, label="FedAvg global model")
        if reference is not None:
            ax.axhline(reference, color=REFERENCE_COLOR, linewidth=2, linestyle="--",
                       label="Centralized checkpoint (epoch 1), validation")
        ax.axvline(selected_round, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Communication round", color=MUTED)
        ax.legend(fontsize=8, frameon=False, loc="best")
    ax = axes.flat[3]
    _style(ax)
    ax.plot(rounds, [row["validation_loss"] for row in history], color=FEDAVG_COLOR, linewidth=2,
            marker="o", markersize=4, label="Global model, validation loss")
    train_rounds = [row["round"] for row in history if row.get("client_train_loss_weighted") is not None]
    ax.plot(train_rounds, [row["client_train_loss_weighted"] for row in history
                           if row.get("client_train_loss_weighted") is not None],
            color=REFERENCE_COLOR, linewidth=2, marker="s", markersize=4,
            label="Clients, local training loss (sample-weighted)")
    ax.axvline(selected_round, color=MUTED, linewidth=1, linestyle=":")
    ax.set_title("Cross-entropy loss", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Communication round", color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"FedAvg convergence (dotted line: selected round {selected_round}, minimum validation loss)",
                 fontsize=12, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_client_losses(client_history, path):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    _style(ax)
    clients = sorted({row["client_id"] for row in client_history})
    for i, client in enumerate(clients):
        rows = [r for r in client_history if r["client_id"] == client]
        ax.plot([r["round"] for r in rows], [r["train_loss"] for r in rows], color=CLIENT_COLORS[i % 8],
                linewidth=2, marker="o", markersize=4, label=client)
    ax.set_title("Local training loss per client", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Communication round", color=MUTED)
    ax.set_ylabel("Mean cross-entropy over local epoch", color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_client_distribution(count_rows, class_names, path):
    """Heatmap of per-client class counts on a log color scale, with counts annotated."""
    clients = [row["client_id"] for row in count_rows]
    counts = np.array([[row[c] for c in class_names] for row in count_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.2), layout="constrained")
    shown = np.where(counts > 0, np.log10(np.maximum(counts, 1)), np.nan)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f4f3ef")
    image = ax.imshow(shown, cmap=cmap, aspect="auto", vmin=-0.6, vmax=np.nanmax(shown))
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(clients)), [f"{row['client_id']} (n={row['total']:,})" for row in count_rows])
    vmax = np.nanmax(shown)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = int(counts[i, j])
            dark = value > 0 and np.log10(value) > 0.6 * vmax
            ax.text(j, i, f"{value:,}" if value else "0", ha="center", va="center", fontsize=7,
                    color="white" if dark else (MUTED if value == 0 else INK))
    ticks = [t for t in range(0, int(np.floor(vmax)) + 1)]
    bar = fig.colorbar(image, ax=ax, label="Segments (log scale)", fraction=0.03, ticks=ticks)
    bar.ax.set_yticklabels([f"{10 ** t:,}" for t in ticks])
    ax.set_title("Training segments per client and class (grey cells: class absent)", loc="left",
                 fontsize=11, color=INK)
    ax.set_xlabel("Beat label", color=MUTED)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of segments")
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, xlabel="Predicted label",
           ylabel="True label", title=title)
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                    color="white" if cm[r, c] > threshold else "black", fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


SEED_STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]  # colour + marker per seed


def _seed_style(seeds):
    return {seed: SEED_STYLES[i % len(SEED_STYLES)] for i, seed in enumerate(sorted(seeds))}


def plot_robustness_convergence(history, reference, path, title="FedAvg robustness"):
    """Rows: partitions; columns: validation loss, macro-F1, accuracy; one line per seed."""
    partitions = list(dict.fromkeys(history["partition"]))
    styles = _seed_style(history["seed"].unique())
    columns = [("validation_loss", "Validation loss"), ("validation_macro_f1", "Validation macro-F1 (15 classes)"),
               ("validation_accuracy", "Validation accuracy")]
    fig, axes = plt.subplots(len(partitions), 3, figsize=(15, 4.2 * len(partitions)), layout="constrained",
                             squeeze=False, sharey="col")
    for r, partition in enumerate(partitions):
        subset = history[history["partition"] == partition]
        for c, (key, panel_title) in enumerate(columns):
            ax = axes[r][c]
            _style(ax)
            for seed, run in subset.groupby("seed"):
                color, marker = styles[seed]
                ax.plot(run["round"], run[key], color=color, marker=marker, markersize=4, linewidth=2,
                        label=f"seed {seed}")
                best = run.loc[run["validation_loss"].idxmin()]
                ax.plot(best["round"], best[key], marker=marker, markersize=10, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.5, linestyle="none")
            ax.axhline(reference[key], color=MUTED, linewidth=1.5, linestyle="--",
                       label="Centralized checkpoint (epoch 1)")
            if key == "validation_loss" and history[key].max() > 50 * history[key].min():
                ax.set_yscale("log")  # DP runs span many orders of magnitude
                panel_title += " (log scale)"
            ax.set_title(f"{partition} partition — {panel_title}", loc="left", fontsize=10, color=INK)
            ax.set_xlabel("Communication round", color=MUTED)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}: validation metrics by round (open marker = round selected by minimum "
                 "validation loss)", fontsize=12, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_robustness_test_metrics(runs, reference, path, title="FedAvg robustness"):
    """One dot per run for test accuracy, macro-F1 and weighted-F1, with the centralized value dashed."""
    partitions = list(dict.fromkeys(runs["partition"]))
    styles = _seed_style(runs["seed"].unique())
    panels = [("test_accuracy", "Test accuracy"), ("test_macro_f1", "Test macro-F1 (15 classes)"),
              ("test_weighted_f1", "Test weighted-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(len(partitions))))
        for x, partition in enumerate(partitions):
            subset = runs[runs["partition"] == partition].sort_values("seed")
            offsets = np.linspace(-0.12, 0.12, len(subset)) if len(subset) > 1 else [0.0]
            for dx, (_, row) in zip(offsets, subset.iterrows()):
                color, marker = styles[row["seed"]]
                ax.plot(x + dx, row[key], marker=marker, markersize=9, color=color, linestyle="none",
                        label=f"seed {row['seed']}" if x == 0 else None)
        ax.axhline(reference[key], color=MUTED, linewidth=1.5, linestyle="--", label="centralized_cnn_v1")
        ax.set_xticks(range(len(partitions)), [f"{p}\npartition" for p in partitions])
        ax.set_xlim(-0.6, len(partitions) - 0.4)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}: held-out test metrics per run (each evaluated once)", fontsize=12, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_paired_slopes(paired, reference, path, arm="fedbn", arm_title="FedBN-style\n(local BatchNorm)",
                       suptitle="Seed-paired comparison: same partition, initialization and training order; only "
                                "BatchNorm handling differs"):
    """Seed-paired slope chart: standard FedAvg vs the diagnostic arm, with the centralized value dashed."""
    panels = [("test_accuracy", "Test accuracy"), ("test_macro_f1", "Test macro-F1 (15 classes)"),
              ("test_weighted_f1", "Test weighted-F1"), ("test_macro_recall", "Test macro recall (15 classes)")]
    styles = _seed_style(paired["seed"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.6), layout="constrained")
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        rows = paired[paired["metric"] == key]
        for _, row in rows.iterrows():
            color, marker = styles[row["seed"]]
            ax.plot([0, 1], [row["fedavg"], row[arm]], color=color, marker=marker, markersize=8, linewidth=2,
                    label=f"seed {row['seed']}")
        ax.axhline(reference[key], color=MUTED, linewidth=1.5, linestyle="--", label="centralized_cnn_v1")
        ax.set_xticks([0, 1], ["FedAvg\n(Day 11)", arm_title])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=12, color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dp_noise_vs_signal(diagnostics, path):
    """Per-round L2 norm of the added noise vs the clipped weighted client signal (log scale), one line pair per seed."""
    styles = _seed_style(diagnostics["seed"].unique())
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    _style(ax)
    for seed, run in diagnostics.groupby("seed"):
        color, marker = styles[seed]
        ax.plot(run["round"], run["noise_l2_norm"], color=color, marker=marker, markersize=5, linewidth=2,
                label=f"seed {seed}: noise")
        ax.plot(run["round"], run["aggregate_signal_l2_norm"], color=color, marker=marker, markersize=5, linewidth=2,
                linestyle=":", markerfacecolor="none", label=f"seed {seed}: clipped client signal")
    ax.set_yscale("log")
    ax.set_xlabel("Communication round", color=MUTED)
    ax.set_ylabel("L2 norm of aggregate component", color=MUTED)
    ax.set_title("DP-FedAvg: Gaussian noise vs clipped, weighted client signal per round (non-private diagnostic)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_he_error(rounds, path):
    """Per-round CKKS error of the decrypted aggregate (max and RMS, log scale), one colour per seed."""
    styles = _seed_style(rounds["seed"].unique())
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    _style(ax)
    for seed, run in rounds.groupby("seed"):
        color, marker = styles[seed]
        ax.plot(run["round"], run["aggregate_max_abs_error"], color=color, marker=marker, markersize=5, linewidth=2,
                label=f"seed {seed}: max |error|")
        ax.plot(run["round"], run["aggregate_rms_error"], color=color, marker=marker, markersize=5, linewidth=2,
                linestyle=":", markerfacecolor="none", label=f"seed {seed}: RMS error")
    ax.set_yscale("log")
    ax.set_xlabel("Communication round", color=MUTED)
    ax.set_ylabel("Decrypted aggregate minus exact float64 aggregate", color=MUTED)
    ax.set_title("HE-FedAvg: CKKS numerical error of the aggregated update per round", loc="left",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ledger_scaling(scaling, path):
    """Block creation and validation time vs chain length (log-log), plus storage per block."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    ax = axes[0]
    _style(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator())
    ax.plot(scaling["chain_length_blocks"], scaling["create_seconds_total"], color=FEDAVG_COLOR, marker="o",
            markersize=6, linewidth=2, label="create chain (all blocks)")
    ax.plot(scaling["chain_length_blocks"], scaling["validate_seconds_total"], color=REFERENCE_COLOR, marker="s",
            markersize=6, linewidth=2, label="validate chain (all blocks)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chain length (blocks; 5 client records per round block)", color=MUTED)
    ax.set_ylabel("Seconds (median)", color=MUTED)
    ax.set_title("Ledger creation and validation time", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    _style(ax)
    ax.plot(scaling["chain_length_blocks"], scaling["file_bytes"] / 1e6, color=FEDAVG_COLOR, marker="o",
            markersize=6, linewidth=2, label="ledger file size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chain length (blocks)", color=MUTED)
    ax.set_ylabel("MB on disk (JSON Lines)", color=MUTED)
    per_block = scaling["file_bytes_per_block"].iloc[-1]
    ax.set_title(f"Ledger storage (about {per_block:,.0f} bytes per block)", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_blockchain_overhead(per_seed, path):
    """Mean per-round time of each recording step vs the mean FedAvg round time (log scale), per seed."""
    styles = _seed_style(per_seed["seed"].unique())
    components = [("hash_ms_per_round", "Hash 5 client states + global"),
                  ("block_creation_ms_per_round", "Create and append block"),
                  ("off_chain_save_ms_per_round", "Save off-chain archive"),
                  ("validate_ms_full_chain", "Validate full chain (once per run)"),
                  ("fl_round_ms", "FedAvg round (training, aggregation, validation)")]
    data = per_seed.assign(fl_round_ms=1e3 * per_seed["training_seconds_with"] / 20)
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    _style(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator())
    for i, (key, label) in enumerate(components):
        for j, (_, row) in enumerate(data.sort_values("seed").iterrows()):
            color, marker = styles[row["seed"]]
            ax.plot(row[key], i + (j - 1) * 0.18, marker=marker, color=color, markersize=9, linestyle="none",
                    label=f"seed {row['seed']}" if i == 0 else None)
    ax.set_yticks(range(len(components)), [label for _, label in components])
    ax.invert_yaxis()
    ax.set_xscale("log")
    values = data[[key for key, _ in components]].to_numpy()
    ax.set_xlim(values.min() / 3, values.max() * 3)
    ax.set_xlabel("Milliseconds (log scale)", color=MUTED)
    ax.set_title("Blockchain recording overhead per FedAvg round (real runs, 5 clients)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)
