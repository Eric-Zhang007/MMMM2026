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

    # For each season-week, take mean P_fan of true eliminated teams (exclude withdrew)
    df["is_elim"] = (df["true_eliminated"].fillna("") == "Y")
    df["is_wd"] = (df["true_withdrew"].fillna("") == "Y")
    d = df[df["is_elim"] & (~df["is_wd"])].copy()
    if len(d) == 0:
        raise RuntimeError("No eliminated rows found in pred_fan_shares_enriched.csv")

    agg = d.groupby(["season", "week"], as_index=False)["P_fan"].mean()
    pivot = agg.pivot(index="season", columns="week", values="P_fan").sort_index()

    data = pivot.to_numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto")
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    ax.set_title("Mean P_fan of eliminated teams (season x week)")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(x)) for x in pivot.index])
    fig.colorbar(im, ax=ax)
    save_fig(fig, os.path.join(fig_dir, "heatmap_elim_pfan_season_week.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
