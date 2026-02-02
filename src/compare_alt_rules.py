import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def _flag_series_to_bool(s: pd.Series) -> pd.Series:
    # exported flags are usually "Y" or NaN
    return s.fillna("").astype(str).eq("Y")


def compute_alt_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Recover judge percentage: D = P_fan - J_pct  =>  J_pct = P_fan - D
    df["J_pct"] = df["P_fan"] - df["D_pf_minus_j"]

    df["is_withdrew"] = _flag_series_to_bool(df.get("true_withdrew", pd.Series(index=df.index)))
    df["is_elim"] = _flag_series_to_bool(df.get("true_eliminated", pd.Series(index=df.index)))
    df["orig_pred_elim"] = _flag_series_to_bool(df.get("pred_eliminated", pd.Series(index=df.index)))

    # Your requested alternative rule:
    # seasons 1-2: percent, seasons 3-27: rank, seasons 28-34: percent
    df["alt_rule"] = np.where(
        df["season"].isin([1, 2]),
        "percent",
        np.where(df["season"].between(3, 27), "rank",
                 np.where(df["season"].between(28, 34), "percent", "other")),
    )

    df["alt_score"] = np.nan
    df["alt_risk"] = np.nan
    df["alt_pred_elim"] = False

    def per_week(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        eff = g[~g["is_withdrew"]].copy()

        if len(eff) == 0:
            g["K_eff"] = 0
            g["week_any_diff"] = False
            g["week_diff_count"] = 0
            g["week_diff_rate_people"] = np.nan
            g["week_diff_rate_elim"] = np.nan
            return g

        K = int(eff["is_elim"].sum())
        g["K_eff"] = K

        if K == 0:
            g["week_any_diff"] = False
            g["week_diff_count"] = 0
            g["week_diff_rate_people"] = np.nan
            g["week_diff_rate_elim"] = np.nan
            return g

        rule = eff["alt_rule"].iloc[0]

        if rule == "percent":
            # score = J_pct + P_fan, higher is safer => risk = -score
            eff["alt_score"] = eff["J_pct"] + eff["P_fan"]
            eff["alt_risk"] = -eff["alt_score"]
            eff = eff.sort_values(
                ["alt_risk", "p_elim", "S_total"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            bottom = set(eff.head(K)["team_id"])

        elif rule == "rank":
            # Rank higher-is-better for J_pct and P_fan
            eff["judge_rank"] = eff["J_pct"].rank(method="average", ascending=False)
            eff["fan_rank"] = eff["P_fan"].rank(method="average", ascending=False)

            # Larger sum rank means worse => higher risk is more dangerous
            eff["alt_risk"] = eff["judge_rank"] + eff["fan_rank"]
            eff["alt_score"] = -eff["alt_risk"]

            # Tie-break: smaller (J_pct + P_fan) is worse, then larger p_elim
            eff["sumpercent"] = eff["J_pct"] + eff["P_fan"]
            eff = eff.sort_values(
                ["alt_risk", "sumpercent", "p_elim"],
                ascending=[False, True, False],
                kind="mergesort",
            )
            bottom = set(eff.head(K)["team_id"])

        else:
            # Fallback: reuse model's risk ordering
            eff["alt_risk"] = -eff["S_total"]
            eff["alt_score"] = eff["S_total"]
            eff = eff.sort_values(["alt_risk"], ascending=[False], kind="mergesort")
            bottom = set(eff.head(K)["team_id"])

        g["alt_pred_elim"] = g["team_id"].isin(bottom) & (~g["is_withdrew"])
        g.loc[eff.index, "alt_score"] = eff["alt_score"]
        g.loc[eff.index, "alt_risk"] = eff["alt_risk"]

        eff_mask = ~g["is_withdrew"]
        orig_set = set(g.loc[eff_mask & g["orig_pred_elim"], "team_id"])
        alt_set = set(g.loc[eff_mask & g["alt_pred_elim"], "team_id"])
        symm = len(orig_set.symmetric_difference(alt_set))
        n_people = int(eff_mask.sum())

        g["week_any_diff"] = symm > 0
        g["week_diff_count"] = symm
        g["week_diff_rate_people"] = symm / n_people if n_people > 0 else np.nan
        g["week_diff_rate_elim"] = symm / (2 * K) if K > 0 else np.nan
        return g

    df = df.groupby(["season", "week"], group_keys=False).apply(per_week)

    df["diff_pred_elim"] = (df["alt_pred_elim"].astype(bool) != df["orig_pred_elim"].astype(bool)) & (~df["is_withdrew"])

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="pred_fan_shares_enriched.csv from a trial")
    ap.add_argument("--out_csv", type=str, required=True, help="output row-level diff csv")
    ap.add_argument("--out_week_csv", type=str, default=None, help="optional week-level summary csv")
    ap.add_argument("--out_json", type=str, default=None, help="optional overall summary json")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    out = compute_alt_predictions(df)

    # Row-level output
    out2 = out.copy()
    out2["orig_pred_eliminated_flag"] = np.where(out2["orig_pred_elim"], "Y", "")
    out2["alt_pred_eliminated_flag"] = np.where(out2["alt_pred_elim"], "Y", "")
    out2["diff_flag"] = np.where(out2["diff_pred_elim"], "Y", "")

    cols = [
        "season", "week", "team_id",
        "alt_rule",
        "P_fan", "J_pct", "D_pf_minus_j",
        "S_total", "alpha", "p_elim",
        "orig_pred_eliminated_flag", "alt_pred_eliminated_flag", "diff_flag",
        "true_eliminated", "true_withdrew",
        "K_eff", "week_any_diff", "week_diff_count", "week_diff_rate_people", "week_diff_rate_elim",
        "alt_score", "alt_risk",
        "Z_pf_minus_j", "Z_outlier_95",
    ]
    cols = [c for c in cols if c in out2.columns]
    out2.to_csv(args.out_csv, index=False)

    # Overall + week summary
    eligible = out[(~out["is_withdrew"]) & (out["K_eff"] > 0)]
    overall_person_diff_rate = float(eligible["diff_pred_elim"].mean()) if len(eligible) else float("nan")

    week_summary = eligible.groupby(["season", "week"]).agg(
        K=("K_eff", "first"),
        n_people=("team_id", "count"),
        any_diff=("week_any_diff", "first"),
        symm_diff=("week_diff_count", "first"),
        diff_rate_people=("week_diff_rate_people", "first"),
        diff_rate_elim=("week_diff_rate_elim", "first"),
    ).reset_index()

    overall_week_any_diff_rate = float(week_summary["any_diff"].mean()) if len(week_summary) else float("nan")
    overall_week_diff_rate_elim = float(week_summary["diff_rate_elim"].mean()) if len(week_summary) else float("nan")

    summary: Dict[str, Any] = {
        "N_weeks": int(len(week_summary)),
        "N_rows": int(len(eligible)),
        "overall_person_diff_rate": overall_person_diff_rate,
        "overall_week_any_diff_rate": overall_week_any_diff_rate,
        "overall_week_diff_rate_elim": overall_week_diff_rate_elim,
    }

    if args.out_week_csv:
        week_summary.to_csv(args.out_week_csv, index=False)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
