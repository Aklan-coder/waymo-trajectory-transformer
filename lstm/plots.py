"""Figures for the report. Colors follow the course-wide convention so Pallavi's and Abdullah's
figures look like one set (colors belong to the model, not to its rank):

    ground truth / history : near-black ink     LSTM : blue      constant velocity : orange
    (reserved for the Transformer: aqua)

Every figure has a legend and uses a second cue besides color (line style / marker), so it still
reads in grayscale print.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_STYLE = {   # color + line style + marker per model
    "Constant velocity": dict(color=ORANGE, ls=(0, (5, 2.5)), marker="s"),
    "LSTM": dict(color=BLUE, ls="-", marker="o"),
    "Transformer": dict(color=AQUA, ls=(0, (1, 1.5)), marker="^"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
})


def _style(name):
    return MODEL_STYLE.get(name, dict(color=INK2, ls="-", marker="o"))


# --------------------------------------------------------------------------- training curves
def plot_loss_curves(history_csv, out_png):
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(h.epoch, h.train_ade, color=INK2, lw=2, ls=(0, (5, 2.5)), solid_capstyle="round", label="Training ADE")
    ax.plot(h.epoch, h.val_ade, color=BLUE, lw=2, solid_capstyle="round", label="Validation ADE")
    best = h.val_ade.idxmin()
    ax.scatter([h.epoch[best]], [h.val_ade[best]], s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"best {h.val_ade[best]:.2f} m\n(epoch {int(h.epoch[best])})", (h.epoch[best], h.val_ade[best]),
                textcoords="offset points", xytext=(10, 12), color=INK2, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average displacement error (m)")
    ax.set_title("LSTM training and validation error")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# --------------------------------------------------------------------------- bar charts
def _rounded_bar_ends(ax, bars):
    """Square at the baseline, soft data end: approximated by a thin surface-colored cap. Kept simple."""
    for b in bars:
        b.set_linewidth(0)


def plot_metric_bars(summary, out_png, title="Prediction error on the validation set"):
    """summary: dict model -> dict(ADE=(mean, lo, hi), FDE=(mean, lo, hi)). Bars with 95% CI whiskers."""
    models = list(summary)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width, gap = 0.26, 0.02
    xs = np.arange(2)
    for i, m in enumerate(models):
        st = _style(m)
        offs = (i - (len(models) - 1) / 2) * (width + gap)
        vals = np.array([summary[m][k][0] for k in ("ADE", "FDE")])
        err = np.array([[summary[m][k][0] - summary[m][k][1], summary[m][k][2] - summary[m][k][0]] for k in ("ADE", "FDE")]).T
        bars = ax.bar(xs + offs, vals, width, color=st["color"], label=m, hatch=None if m == "LSTM" else "////",
                      edgecolor=SURFACE, linewidth=0, zorder=2)
        _rounded_bar_ends(ax, bars)
        ax.errorbar(xs + offs, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        top = max(s["FDE"][2] for s in summary.values())
        for x, v, hi in zip(xs + offs, vals, vals + err[1]):
            ax.text(x, hi + 0.015 * top, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(xs, ["ADE\n(average over 3 s)", "FDE\n(at 3 s)"])
    ax.set_ylabel("Error (m), lower is better")
    ax.set_title(title)
    ax.set_ylim(0, 1.12 * max(s["FDE"][2] for s in summary.values()))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, "Whiskers: 95% bootstrap confidence interval over vehicles", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def plot_by_group(table, models, out_png, metric="ADE", title=None, min_n=20):
    """table: DataFrame with columns group, n, and '<model> <metric>' columns. One cluster per group."""
    t = table[table.n >= min_n].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width, gap = 0.26, 0.02
    xs = np.arange(len(t))
    for i, m in enumerate(models):
        st = _style(m)
        offs = (i - (len(models) - 1) / 2) * (width + gap)
        vals = t[f"{m} {metric}"].to_numpy()
        ax.bar(xs + offs, vals, width, color=st["color"], label=m, linewidth=0, zorder=2,
               hatch=None if m == "LSTM" else "////", edgecolor=SURFACE)
    ax.set_xticks(xs, [f"{g}\n(n={n})" for g, n in zip(t.group, t.n)])
    ax.set_ylabel(f"Mean {metric} (m), lower is better")
    ax.set_title(title or f"{metric} by kind of motion")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# --------------------------------------------------------------------------- trajectory overlays
def plot_overlays(x, gt, preds, idx, out_png, title, ncols=3):
    """Small multiples of individual vehicles in the heading frame.

    x: [N, 11, F] inputs (positions in columns 0:2), gt: [N, 30, 2], preds: dict model -> [N, 30, 2].
    """
    n = len(idx)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False)
    for ax, i in zip(axes.ravel(), idx):
        hist = x[i, :, :2]
        ax.plot(hist[:, 0], hist[:, 1], color=INK, lw=2, solid_capstyle="round", label="Past 1 s")
        ax.plot(np.r_[hist[-1:, 0], gt[i, :, 0]], np.r_[hist[-1:, 1], gt[i, :, 1]], color=INK, lw=2, ls=(0, (1, 1.8)),
                solid_capstyle="round", label="True future 3 s")
        for m, p in preds.items():
            st = _style(m)
            e = np.linalg.norm(p[i, -1] - gt[i, -1])
            ax.plot(np.r_[hist[-1:, 0], p[i, :, 0]], np.r_[hist[-1:, 1], p[i, :, 1]], color=st["color"], ls=st["ls"], lw=2,
                    label=f"{m} (FDE {e:.1f} m)")
            ax.scatter([p[i, -1, 0]], [p[i, -1, 1]], s=56, color=st["color"], marker=st["marker"], edgecolor=SURFACE,
                       linewidth=1.5, zorder=4)
        ax.scatter([gt[i, -1, 0]], [gt[i, -1, 1]], s=56, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.scatter([0], [0], s=30, color=SURFACE, edgecolor=INK, linewidth=1.5, zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Example {i}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="best", handlelength=2.2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("Forward distance (m)", color=INK2, fontsize=9)
    fig.supylabel("Sideways distance (m)", color=INK2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
    fig.savefig(out_png)
    plt.close(fig)
