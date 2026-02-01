import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.viz.plot_utils import _ensure_dir, save_fig


def herfindahl(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(np.sum(p ** 2))


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p)))


def main(run_dir: str):
    pred_path = os.path.join(run_dir, "pred_fan_shares_enriched.csv")
    df = pd.read_csv(pred_path)

    fig_dir = os.path.join(run_dir, "figures")
    _ensure_dir(fig_dir)

    # Compute concentration metrics per season-week on active teams
    rows = []
    for (s, w), g in df.groupby(["season", "week"]):
        p = g["P_fan"].to_numpy(dtype=float)
        rows.append({"season": int(s), "week": int(w), "HHI": herfindahl(p), "Entropy": entropy(p)})
    m = pd.DataFrame(rows)

    for metric in ["HHI", "Entropy"]:
        pivot = m.pivot(index="season", columns="week", values=metric).sort_index()
        data = pivot.to_numpy()
        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect="auto")
        ax.set_xlabel("Week")
        ax.set_ylabel("Season")
        ax.set_title(f"{metric} of P_fan distribution (season x week)")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(x)) for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(x)) for x in pivot.index])
        fig.colorbar(im, ax=ax)
        save_fig(fig, os.path.join(fig_dir, f"heatmap_pfan_{metric.lower()}_season_week.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
