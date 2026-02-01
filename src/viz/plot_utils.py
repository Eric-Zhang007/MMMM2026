import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(fig, out_path: str, dpi: int = 200) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def heatmap(ax, data: np.ndarray, x_labels, y_labels, title: str, vmin=None, vmax=None):
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    return im
