#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare LMM-ready long panel from:
1) 2026_MCM_Problem_C_Data.csv (wide by week judge scores)
2) lmm_long.csv (season-week-celebrity aggregated Google Trends)

Outputs a long table with one row per active (season, week, celebrity, partner):
- judge targets: j_total, j_pct, y_judge (configurable)
- trends: g_raw (log ratio to anchor if provided), g_z_within_sw, m_s (twist enable)
- ids: team_id, celeb_id, partner_id
- helpful flags: is_withdrew, withdrew_week, is_active

Example:
python prepare_lmm_dataset_from_trend_table_v2.py \
  --raw_csv 2026_MCM_Problem_C_Data.csv \
  --trends_csv lmm_long.csv \
  --out_csv lmm_dataset.csv \
  --trend_col trend_mean \
  --judge_mode season_z_of_pct \
  --enable_min 21 --enable_max 34 \
  --eps 1.0 \
  --condition_md condition.md
"""

import argparse
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def normalize_name(s: pd.Series) -> pd.Series:
    # Lowercase, strip, collapse spaces, remove most punctuation.
    def _norm_one(x: str) -> str:
        x = str(x).strip().lower()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^a-z0-9 ]+", "", x)
        return x
    return s.fillna("").map(_norm_one)


def parse_condition_md(path: str) -> Dict[int, str]:
    season_to_anchor: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*[:：]\s*(.+?)\s*$", line)
            if m:
                season_to_anchor[int(m.group(1))] = m.group(2).strip()
                continue
            m = re.match(r"^season\s*(\d+)\s*[-–>]+\s*(.+?)\s*$", line, flags=re.IGNORECASE)
            if m:
                season_to_anchor[int(m.group(1))] = m.group(2).strip()
    if not season_to_anchor:
        raise ValueError("Failed to parse condition.md into season->anchor mapping.")
    return season_to_anchor


def wide_to_long_judges(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wide week{w}_judge{k}_score columns into long rows.
    A row is considered 'active' for that week if any judge score > 0.
    """
    df = raw.copy()

    # identify all week judge score columns
    pat = re.compile(r"^week(\d+)_judge(\d+)_score$", flags=re.IGNORECASE)
    week_cols: Dict[int, List[str]] = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            w = int(m.group(1))
            week_cols.setdefault(w, []).append(c)

    if not week_cols:
        raise ValueError("No week*_judge*_score columns found in raw_csv.")

    weeks = sorted(week_cols.keys())

    long_rows = []
    for w in weeks:
        cols = week_cols[w]
        tmp = df[["season", "celebrity_name", "ballroom_partner", "results", "placement"] + cols].copy()

        # numeric judge scores, keep NaN as NaN
        for c in cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        # active if any judge score strictly > 0
        scores = tmp[cols]
        active = (scores.fillna(0.0) > 0.0).any(axis=1)
        tmp = tmp.loc[active].copy()
        tmp["week"] = w

        # judge total: sum of available judges this week (ignore NaN)
        tmp["j_total"] = scores.loc[active].sum(axis=1, skipna=True)

        # keep individual judge totals if needed later
        long_rows.append(tmp[["season", "week", "celebrity_name", "ballroom_partner", "results", "placement", "j_total"]])

    out = pd.concat(long_rows, ignore_index=True)
    out["team_id"] = out["celebrity_name"].astype(str) + "__" + out["ballroom_partner"].astype(str)
    out["celeb_id"] = out["celebrity_name"].astype(str)
    out["partner_id"] = out["ballroom_partner"].astype(str)

    # withdrew flag (season-level); withdrew_week inferred as last active week
    out["is_withdrew"] = out.groupby(["season", "team_id"])["results"].transform(
        lambda s: s.astype(str).str.contains("Withdrew", na=False).any()
    )
    last_week = out.groupby(["season", "team_id"])["week"].transform("max")
    out["withdrew_week"] = np.where(out["is_withdrew"], last_week, np.nan)

    out["is_active"] = 1
    return out


def add_judge_targets(long_df: pd.DataFrame, judge_mode: str) -> pd.DataFrame:
    df = long_df.copy()

    # judge percentage within (season, week)
    df["j_pct"] = df.groupby(["season", "week"])["j_total"].transform(lambda x: x / (x.sum() + 1e-12))

    if judge_mode == "season_z_of_pct":
        df["y_judge"] = df.groupby("season")["j_pct"].transform(zscore)
    elif judge_mode == "week_z_of_pct":
        df["y_judge"] = df.groupby(["season", "week"])["j_pct"].transform(zscore)
    elif judge_mode == "season_z_of_total":
        df["y_judge"] = df.groupby("season")["j_total"].transform(zscore)
    else:
        raise ValueError("judge_mode must be one of: season_z_of_pct, week_z_of_pct, season_z_of_total")

    # centered week within season (for random slope usage later)
    df["week_c"] = df["week"] - df.groupby("season")["week"].transform("mean")
    return df


def add_trends(
    df: pd.DataFrame,
    trends: pd.DataFrame,
    trend_col: str,
    enable_min: int,
    enable_max: int,
    eps: float,
    condition_md: Optional[str],
) -> pd.DataFrame:
    tr = trends.copy()
    if trend_col not in tr.columns:
        raise ValueError(f"trend_col={trend_col} not found in trends_csv. Available: {list(tr.columns)}")

    tr = tr[["season", "week", "celebrity_name", trend_col]].copy()
    tr[trend_col] = pd.to_numeric(tr[trend_col], errors="coerce")
    tr = tr.dropna(subset=[trend_col])

    # robust join key
    tr["celeb_key"] = normalize_name(tr["celebrity_name"])
    df2 = df.copy()
    df2["celeb_key"] = normalize_name(df2["celeb_id"])

    merged = df2.merge(tr[["season", "week", "celeb_key", trend_col]],
                       on=["season", "week", "celeb_key"], how="left")

    merged["trend_val"] = merged[trend_col].fillna(0.0)

    # twist enable by season range
    merged["m_s"] = ((merged["season"] >= enable_min) & (merged["season"] <= enable_max)).astype(int)

    if condition_md is None:
        # without anchor: log(trend + eps)
        merged["g_raw"] = np.log(merged["trend_val"] + eps)
    else:
        season_to_anchor = parse_condition_md(condition_md)
        # build anchor trend per season-week using trends table
        tr_anchor = tr.copy()
        tr_anchor["anchor_name"] = tr_anchor["season"].map(season_to_anchor).fillna("")
        tr_anchor["anchor_key"] = normalize_name(tr_anchor["anchor_name"])
        # join anchor trend
        anchor = tr_anchor[tr_anchor["celeb_key"] == tr_anchor["anchor_key"]][["season","week",trend_col]].rename(
            columns={trend_col: "trend_anchor"}
        )
        merged = merged.merge(anchor, on=["season","week"], how="left")
        merged["trend_anchor"] = merged["trend_anchor"].fillna(0.0)
        merged["g_raw"] = np.log((merged["trend_val"] + eps) / (merged["trend_anchor"] + eps))

    # within season-week zscore for stability
    merged["g_z_within_sw"] = merged.groupby(["season","week"])["g_raw"].transform(zscore).fillna(0.0)

    # if season not enabled, zero-out trend features
    merged.loc[merged["m_s"] == 0, ["trend_val", "g_raw", "g_z_within_sw"]] = 0.0

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, required=True)
    ap.add_argument("--trends_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--trend_col", type=str, default="trend_mean", choices=["trend_mean","trend_sum","trend_max"])
    ap.add_argument("--judge_mode", type=str, default="season_z_of_pct",
                    choices=["season_z_of_pct","week_z_of_pct","season_z_of_total"])
    ap.add_argument("--enable_min", type=int, default=21)
    ap.add_argument("--enable_max", type=int, default=34)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--condition_md", type=str, default=None)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw_csv)
    trends = pd.read_csv(args.trends_csv)

    long_df = wide_to_long_judges(raw)
    long_df = add_judge_targets(long_df, judge_mode=args.judge_mode)
    out = add_trends(
        long_df,
        trends=trends,
        trend_col=args.trend_col,
        enable_min=args.enable_min,
        enable_max=args.enable_max,
        eps=args.eps,
        condition_md=args.condition_md,
    )

    # final column set
    keep = [
        "season","week","week_c",
        "team_id","celeb_id","partner_id",
        "j_total","j_pct","y_judge",
        "trend_val","g_raw","g_z_within_sw","m_s",
        "is_withdrew","withdrew_week","results","placement",
    ]
    keep = [c for c in keep if c in out.columns]
    out2 = out[keep].copy()

    out2.to_csv(args.out_csv, index=False)
    print(f"Saved {len(out2)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
