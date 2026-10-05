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
