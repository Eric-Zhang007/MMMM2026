import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.viz.plot_utils import _ensure_dir, save_fig


def main(run_dir: str, top_n: int = 20):
    sum_path = os.path.join(run_dir, "team_season_summary.csv")
    if not os.path.exists(sum_path):
        raise RuntimeError("team_season_summary.csv not found. Run training/export first.")
    df = pd.read_csv(sum_path)

    fig_dir = os.path.join(run_dir, "figures")
    _ensure_dir(fig_dir)

    # Top by frac_outlier_95
    d1 = df.sort_values(["frac_outlier_95", "Z_max_abs"], ascending=[False, False]).head(top_n)
    labels = [f"S{int(s)}:{tid}" for s, tid in zip(d1["season"], d1["team_id"])]
    x = np.arange(len(d1))

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.4), 4))
    ax.bar(x, d1["frac_outlier_95"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("frac_outlier_95")
    ax.set_title(f"Most controversial (by frac_outlier_95), top {top_n}")
    save_fig(fig, os.path.join(fig_dir, "controversy_top_frac_outlier_95.png"))

    # Top by Z_max_abs
    d2 = df.sort_values(["Z_max_abs", "frac_outlier_95"], ascending=[False, False]).head(top_n)
    labels = [f"S{int(s)}:{tid}" for s, tid in zip(d2["season"], d2["team_id"])]
    x = np.arange(len(d2))

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.4), 4))
    ax.bar(x, d2["Z_max_abs"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Z_max_abs")
    ax.set_title(f"Most extreme Z (by max |Z|), top {top_n}")
    save_fig(fig, os.path.join(fig_dir, "controversy_top_zmaxabs.png"))

    # Scatter: mean vs max abs with size by outlier fraction
    fig, ax = plt.subplots()
    ax.scatter(df["Z_mean"].to_numpy(), df["Z_max_abs"].to_numpy(), s=20)
    ax.set_xlabel("Z_mean")
    ax.set_ylabel("Z_max_abs")
    ax.set_title("Controversy map (mean Z vs max |Z|)")
    save_fig(fig, os.path.join(fig_dir, "controversy_scatter_mean_vs_maxabs.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    main(args.run_dir, top_n=args.top_n)
