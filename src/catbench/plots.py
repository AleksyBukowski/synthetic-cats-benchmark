"""Small matplotlib helpers used by the notebooks."""
import matplotlib.pyplot as plt
import numpy as np

from .config import BREEDS

BAR_COLOR = "#3b6fb6"


def metric_bars(runs, backbone: str, split: str = "test",
                metrics=(("mau_recall", "Egyptian Mau recall (primary)"),
                         ("macro_f1", "Macro-F1, 8 classes (guardrail)"))):
    """One panel per metric: mean over seeds per condition, error bar = ±1 std."""
    sub = runs[(runs.backbone == backbone) & (runs.split == split)]
    conditions = list(dict.fromkeys(sub.condition))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.6))
    for ax, (col, title) in zip(np.atleast_1d(axes), metrics):
        g = sub.groupby("condition")[col]
        mean, std = g.mean()[conditions], g.std(ddof=1).fillna(0)[conditions]
        ax.bar(conditions, mean, yerr=std, color=BAR_COLOR, width=0.55, capsize=4)
        for x, m in enumerate(mean):
            ax.text(x, 0.03, f"{m:.3f}", ha="center", va="bottom", color="white", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(f"{backbone} - {split} set, mean ± std over seeds", fontsize=12)
    fig.tight_layout()
    return fig


def confusion_grid(confusion: dict, backbone: str, conditions):
    """Row-normalised test confusion matrices (summed over seeds), one per condition."""
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.2 * len(conditions), 5))
    short = [b.replace("Egyptian Mau", "Eg. Mau").replace("Russian Blue", "Rus. Blue") for b in BREEDS]
    for ax, cond in zip(axes, conditions):
        cm = confusion[(backbone, cond)]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(BREEDS)):
            for j in range(len(BREEDS)):
                if norm[i, j] >= 0.05:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(BREEDS)), short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(BREEDS)), short, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_title(cond)
    axes[0].set_ylabel("true")
    fig.suptitle(f"{backbone} - test confusion (rows sum to 1)")
    fig.tight_layout()
    return fig


def image_grid(images, titles=None, ncols: int = 8, size: float = 1.6):
    """Show a list of PIL images in a grid."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    for k, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if k < len(images):
            ax.imshow(images[k])
            if titles:
                ax.set_title(titles[k], fontsize=7)
    fig.tight_layout()
    return fig
