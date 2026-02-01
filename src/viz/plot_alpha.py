import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.viz.plot_utils import _ensure_dir, save_fig


def main(run_dir: str):
    pred_path = os.path.join(run_dir, "pred_fan_shares_enriched.csv")
    df = pd.read_csv(pred_path)

    fig_dir = os.path.join(run_dir, "figures")
    _ensure_dir(fig_dir)

    # alpha is constant within a week, repeated for each team row
    wk = df.groupby(["season", "week"], as_index=False)["alpha"].mean()
    seas = wk.groupby("season", as_index=False)["alpha"].mean().sort_values("season")

    fig, ax = plt.subplots()
    ax.plot(seas["season"].to_numpy(), seas["alpha"].to_numpy())
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean alpha")
    ax.set_title("Alpha by season (mean across weeks)")
    save_fig(fig, os.path.join(fig_dir, "alpha_by_season.png"))

    # heatmap season x week
    pivot = wk.pivot(index="season", columns="week", values="alpha").sort_index()
    data = pivot.to_numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto")
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    ax.set_title("Alpha heatmap (season x week)")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    fig.colorbar(im, ax=ax)
    save_fig(fig, os.path.join(fig_dir, "alpha_heatmap_season_week.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
