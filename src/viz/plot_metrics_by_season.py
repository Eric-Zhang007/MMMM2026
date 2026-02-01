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

    # Build week-level metrics directly from exported S_total / true labels.
    # Exclude withdrew teams when comparing eliminated vs survivors.
    wk_rows = []
    for (s, w), g in df.groupby(["season", "week"]):
        g = g.copy()
        g["is_wd"] = (g["true_withdrew"].fillna("") == "Y")
        g_eff = g[~g["is_wd"]]
        L = g_eff[g_eff["true_eliminated"].fillna("") == "Y"]
        K = len(L)
        if K == 0:
            continue
        risk = (-g_eff["S_total"].to_numpy(dtype=float))
        order = np.argsort(-risk)  # descending risk
        bottomk = set(order[: min(K, len(order))])
        elim_idx = set(L.index.map(lambda idx: g_eff.index.get_loc(idx)).tolist())
        bottomk_acc = sum(1 for x in elim_idx if x in bottomk) / float(K)

        # pairwise order
        W = [i for i in range(len(g_eff)) if i not in elim_idx]
        pairwise = np.mean([1.0 if risk[l] > risk[w2] else 0.0 for l in elim_idx for w2 in W]) if W else np.nan

        top1 = np.nan
        if K == 1:
            top1 = 1.0 if order[0] in elim_idx else 0.0

        wk_rows.append({"season": int(s), "week": int(w), "BottomKAcc": bottomk_acc, "PairwiseOrderAcc": pairwise, "Top1Acc": top1})

    m = pd.DataFrame(wk_rows)
    if len(m) == 0:
        raise RuntimeError("No eligible weeks for metric plotting.")

    seas = m.groupby("season", as_index=False).agg(
        BottomKAcc=("BottomKAcc", "mean"),
        PairwiseOrderAcc=("PairwiseOrderAcc", "mean"),
        Top1Acc=("Top1Acc", "mean"),
    ).sort_values("season")

    for col in ["BottomKAcc", "PairwiseOrderAcc", "Top1Acc"]:
        fig, ax = plt.subplots()
        ax.plot(seas["season"].to_numpy(), seas[col].to_numpy())
        ax.set_xlabel("Season")
        ax.set_ylabel(col)
        ax.set_title(f"{col} by season")
        save_fig(fig, os.path.join(fig_dir, f"metric_{col.lower()}_by_season.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.run_dir)
