#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare an LMM-ready long table by merging:
1) competition raw panel (season-week-celebrity/partner) and
2) a pre-aggregated Google Trends table (like lmm_long.csv you provided).

This script supports condition-based normalization (anchor celebrity per season)
and produces per-row trend features:
- g_raw      : log(trend + eps) (or log ratio if condition provided)
- g_z_within_sw : z-score within each (season, week)
- m_s        : twist switch (1 for seasons in [enable_min, enable_max], else 0)

It also builds judge targets:
- j_total (sum of judge scores if available)
- j_pct   (within-week share among active)
- y_judge (configurable standardization)

Example:
python prepare_lmm_dataset_from_trend_table.py \
  --raw_csv 2026_MCM_Problem_C_Data.csv \
  --trends_csv lmm_long.csv \
  --condition_md condition.md \
  --out_csv lmm_dataset.csv \
  --trend_col trend_mean \
  --judge_mode season_z_of_pct \
  --enable_min 21 --enable_max 34
"""

import argparse
import os
import re
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def load_condition_md(path: str) -> Dict[int, str]:
    season_to_anchor: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = [
        r"^\s*(\d+)\s*[:：]\s*(.+?)\s*$",
        r"^\s*Season\s*(\d+)\s*[-–>]+\s*(.+?)\s*$",
    ]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for pat in patterns:
            m = re.match(pat, line, flags=re.IGNORECASE)
            if m:
                season_to_anchor[int(m.group(1))] = m.group(2).strip()
                break

    if not season_to_anchor:
        raise ValueError("Failed to parse condition_md (no season -> anchor extracted).")
    return season_to_anchor


def infer_judge_total(df: pd.DataFrame) -> pd.Series:
    # If j_total exists, use it.
    for cand in ["j_total", "judge_total", "total_judge", "judges_total"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")

    # Sum judge score columns if present.
    # Try common patterns: judge1_score.. judge4_score, or columns containing 'judge' and 'score'
    judge_cols: List[str] = []
    for c in df.columns:
        lc = c.lower()
        if ("judge" in lc and "score" in lc) or re.match(r"^j\d+(_score)?$", lc):
            judge_cols.append(c)

    if judge_cols:
        return pd.to_numeric(df[judge_cols], errors="coerce").sum(axis=1)

    raise ValueError("Cannot infer judge total: no j_total/judge_total and no judge score columns detected.")


def build_targets(df_raw: pd.DataFrame, judge_mode: str) -> pd.DataFrame:
    df = df_raw.copy()

    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("raw_csv must contain columns: season, week")

    if "celebrity_name" not in df.columns:
        # try a few alternatives
        for cand in ["celebrity", "star", "celeb", "name"]:
            if cand in df.columns:
                df["celebrity_name"] = df[cand].astype(str)
                break
    if "celebrity_name" not in df.columns:
        raise ValueError("raw_csv must contain celebrity_name (or a compatible column).")

    # Active indicator
    if "is_active" in df.columns:
        df = df[df["is_active"].astype(int) == 1].copy()

    df["j_total"] = infer_judge_total(df)

    # Within-week judge percentage among active
    df["j_pct"] = df.groupby(["season", "week"])["j_total"].transform(lambda x: x / (x.sum() + 1e-12))

    if judge_mode == "season_z_of_pct":
        df["y_judge"] = df.groupby(["season"])["j_pct"].transform(zscore)
    elif judge_mode == "week_z_of_pct":
        df["y_judge"] = df.groupby(["season", "week"])["j_pct"].transform(zscore)
    elif judge_mode == "season_z_of_total":
        df["y_judge"] = df.groupby(["season"])["j_total"].transform(zscore)
    else:
        raise ValueError("judge_mode must be one of: season_z_of_pct, week_z_of_pct, season_z_of_total")

    # IDs
    if "ballroom_partner" in df.columns:
        df["partner_id"] = df["ballroom_partner"].astype(str)
    elif "partner_name" in df.columns:
        df["partner_id"] = df["partner_name"].astype(str)
    else:
        df["partner_id"] = ""

    df["celeb_id"] = df["celebrity_name"].astype(str)
    df["team_id"] = np.where(df["partner_id"].astype(str) != "",
                             df["celeb_id"] + "__" + df["partner_id"].astype(str),
                             df["celeb_id"])

    # week centered within season
    df["week_c"] = df["week"] - df.groupby("season")["week"].transform("mean")

    return df


def prepare_trend_features(
    trends: pd.DataFrame,
    trend_col: str,
    eps: float,
    enable_min: int,
    enable_max: int,
    season_to_anchor: Optional[Dict[int, str]],
) -> pd.DataFrame:
    tr = trends.copy()

    needed = {"season", "week", "celebrity_name", trend_col}
    missing = [c for c in needed if c not in tr.columns]
    if missing:
        raise ValueError(f"trends_csv missing columns: {missing}")

    tr["season"] = tr["season"].astype(int)
    tr["week"] = tr["week"].astype(int)
    tr["celebrity_name"] = tr["celebrity_name"].astype(str)

    tr["_trend"] = pd.to_numeric(tr[trend_col], errors="coerce").fillna(0.0)

    # twist switch
    tr["m_s"] = ((tr["season"] >= enable_min) & (tr["season"] <= enable_max)).astype(int)

    if season_to_anchor is not None:
        anchor_df = pd.DataFrame({
            "season": list(season_to_anchor.keys()),
            "anchor_name": list(season_to_anchor.values()),
        }).astype({"season": int, "anchor_name": str})
        tr = tr.merge(anchor_df, on="season", how="left")

        # pull anchor trend per (season, week)
        anchor_tr = tr[tr["celebrity_name"] == tr["anchor_name"]][["season", "week", "_trend"]] \
            .rename(columns={"_trend": "_anchor_trend"})
        tr = tr.merge(anchor_tr, on=["season", "week"], how="left")

        # if anchor missing for a (season, week), set anchor trend to nan -> will become 0 contribution after m_s gate
        tr["_anchor_trend"] = tr["_anchor_trend"].fillna(np.nan)

        # log ratio relative to anchor
        tr["g_raw"] = np.log((tr["_trend"] + eps) / (tr["_anchor_trend"] + eps))
    else:
        tr["g_raw"] = np.log(tr["_trend"] + eps)

    # z-score within (season, week)
    tr["g_z_within_sw"] = tr.groupby(["season", "week"])["g_raw"].transform(zscore).fillna(0.0)

    # apply twist gate: seasons outside range -> zero trend effect
    tr.loc[tr["m_s"] == 0, ["g_raw", "g_z_within_sw"]] = 0.0

    keep = ["season", "week", "celebrity_name", "g_raw", "g_z_within_sw", "m_s"]
    return tr[keep].drop_duplicates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, required=True)
    ap.add_argument("--trends_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)

    ap.add_argument("--trend_col", type=str, default="trend_mean", help="trend column to use from trends_csv")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--judge_mode", type=str, default="season_z_of_pct",
                    choices=["season_z_of_pct", "week_z_of_pct", "season_z_of_total"])

    ap.add_argument("--condition_md", type=str, default=None,
                    help="optional condition.md (season -> anchor celebrity) for ratio normalization")
    ap.add_argument("--enable_min", type=int, default=21)
    ap.add_argument("--enable_max", type=int, default=34)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw_csv)
    tr = pd.read_csv(args.trends_csv)

    season_to_anchor = None
    if args.condition_md:
        season_to_anchor = load_condition_md(args.condition_md)

    df = build_targets(raw, args.judge_mode)
    gfeat = prepare_trend_features(
        trends=tr,
        trend_col=args.trend_col,
        eps=args.eps,
        enable_min=args.enable_min,
        enable_max=args.enable_max,
        season_to_anchor=season_to_anchor,
    )

    out = df.merge(
        gfeat,
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    out[["g_raw", "g_z_within_sw", "m_s"]] = out[["g_raw", "g_z_within_sw", "m_s"]].fillna(0.0)

    # keep a compact set of columns; also pass through any x_ identity vector columns if present in raw
    x_cols = [c for c in out.columns if c.startswith("x_")]
    keep_cols = [
        "season", "week", "week_c",
        "team_id", "celeb_id", "partner_id",
        "celebrity_name",
        "j_total", "j_pct", "y_judge",
        "g_raw", "g_z_within_sw", "m_s",
    ] + x_cols

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()
    out.to_csv(args.out_csv, index=False)
    print(f"Saved {len(out)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
