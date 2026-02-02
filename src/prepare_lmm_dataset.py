import argparse
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def safe_logit(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    p2 = p.clip(eps, 1 - eps)
    return np.log(p2 / (1 - p2))


@dataclass
class ConditionMap:
    season_to_anchor: Dict[int, str]


def load_condition_md(path: str) -> ConditionMap:
    season_to_anchor: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 允许格式如: "21: Taylor Swift" 或 "Season 21 -> Taylor Swift" 等
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
                s = int(m.group(1))
                anchor = m.group(2).strip()
                season_to_anchor[s] = anchor
                break

    if not season_to_anchor:
        raise ValueError("condition.md 解析失败，未提取到 season -> anchor 映射，请检查格式。")

    return ConditionMap(season_to_anchor=season_to_anchor)


def read_trends_zip(zip_path: str) -> pd.DataFrame:
    # 期望 zip 内包含若干 csv，至少有 columns: season, week, term, value
    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                df = pd.read_csv(f)
            cols = {c.lower(): c for c in df.columns}

            needed = ["season", "week", "term"]
            for k in needed:
                if k not in cols:
                    raise ValueError(f"趋势文件 {name} 缺少列 {k}")

            # value 列允许叫 value, index, score
            value_col = None
            for cand in ["value", "index", "score", "trend"]:
                if cand in cols:
                    value_col = cols[cand]
                    break
            if value_col is None:
                raise ValueError(f"趋势文件 {name} 缺少数值列 value/index/score/trend")

            out = pd.DataFrame({
                "season": df[cols["season"]].astype(int),
                "week": df[cols["week"]].astype(int),
                "term": df[cols["term"]].astype(str),
                "trend": pd.to_numeric(df[value_col], errors="coerce"),
            })
            rows.append(out)

    if not rows:
        raise ValueError("zip 内未找到 csv")

    all_tr = pd.concat(rows, ignore_index=True)
    all_tr = all_tr.dropna(subset=["trend"])
    return all_tr


def build_google_feature(
    trends: pd.DataFrame,
    cond: ConditionMap,
    eps: float = 1.0,
    enable_seasons: Tuple[int, int] = (21, 34),
) -> pd.DataFrame:
    # trends: season week term trend
    # 输出: season week term g_log_ratio g_z_within_sw m_s
    season_min, season_max = enable_seasons

    # anchor trend
    anchors = []
    for s, anchor in cond.season_to_anchor.items():
        anchors.append({"season": int(s), "anchor": str(anchor)})
    df_anchor = pd.DataFrame(anchors)

    tr = trends.merge(df_anchor, on="season", how="left")
    if tr["anchor"].isna().any():
        missing = sorted(tr.loc[tr["anchor"].isna(), "season"].unique().tolist())
        raise ValueError(f"condition 缺少 seasons: {missing}")

    # 获取 anchor 序列
    anchor_tr = tr[tr["term"] == tr["anchor"]][["season", "week", "trend"]].rename(columns={"trend": "trend_anchor"})
    tr2 = tr.merge(anchor_tr, on=["season", "week"], how="left")
    # 若某周 anchor 缺失，直接丢弃该周全部 term，避免污染
    tr2 = tr2.dropna(subset=["trend_anchor"])

    tr2["g_log_ratio"] = np.log((tr2["trend"] + eps) / (tr2["trend_anchor"] + eps))

    # week 内 zscore
    tr2["g_z_within_sw"] = tr2.groupby(["season", "week"])["g_log_ratio"].transform(zscore)

    # twist 开关
    tr2["m_s"] = ((tr2["season"] >= season_min) & (tr2["season"] <= season_max)).astype(int)
    # 关闭季直接置零
    tr2.loc[tr2["m_s"] == 0, ["g_log_ratio", "g_z_within_sw"]] = 0.0

    return tr2[["season", "week", "term", "g_log_ratio", "g_z_within_sw", "m_s"]]


def build_judge_targets(df_raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    # 需要列: season week team_id j_total 或者四位评委分数列
    df = df_raw.copy()

    if "team_id" not in df.columns:
        # 兼容 celebrity + partner 字段
        if "celebrity_name" in df.columns and "ballroom_partner" in df.columns:
            df["team_id"] = df["celebrity_name"].astype(str) + "__" + df["ballroom_partner"].astype(str)
        else:
            raise ValueError("原始数据缺少 team_id 或 celebrity_name/ballroom_partner")

    if "j_total" not in df.columns:
        judge_cols = [c for c in df.columns if c.lower().startswith("judge") and c.lower().endswith("_score")]
        if len(judge_cols) >= 1:
            df["j_total"] = df[judge_cols].sum(axis=1)
        else:
            # 尝试四位评委通用列名
            cand = [c for c in df.columns if "score" in c.lower() and "judge" in c.lower()]
            if cand:
                df["j_total"] = df[cand].sum(axis=1)
            else:
                raise ValueError("无法构造 j_total，请检查评委分数字段")

    # active 判定
    if "is_active" in df.columns:
        df = df[df["is_active"] == 1].copy()

    # J_pct 以当周 active 总分归一
    df["j_pct"] = df.groupby(["season", "week"])["j_total"].transform(lambda x: x / (x.sum() + 1e-9))

    if mode == "season_z_of_pct":
        df["y_judge"] = df.groupby(["season"])["j_pct"].transform(zscore)
    elif mode == "week_z_of_pct":
        df["y_judge"] = df.groupby(["season", "week"])["j_pct"].transform(zscore)
    elif mode == "season_z_of_total":
        df["y_judge"] = df.groupby(["season"])["j_total"].transform(zscore)
    else:
        raise ValueError("mode 必须是 season_z_of_pct, week_z_of_pct, season_z_of_total")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, required=True)
    ap.add_argument("--condition_md", type=str, required=True)
    ap.add_argument("--trends_zip", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--judge_mode", type=str, default="season_z_of_pct",
                    choices=["season_z_of_pct", "week_z_of_pct", "season_z_of_total"])
    ap.add_argument("--trend_enable_seasons_min", type=int, default=21)
    ap.add_argument("--trend_enable_seasons_max", type=int, default=34)
    ap.add_argument("--eps", type=float, default=1.0)
    args = ap.parse_args()

    df_raw = pd.read_csv(args.raw_csv)
    cond = load_condition_md(args.condition_md)
    trends = read_trends_zip(args.trends_zip)

    gfeat = build_google_feature(
        trends=trends,
        cond=cond,
        eps=args.eps,
        enable_seasons=(args.trend_enable_seasons_min, args.trend_enable_seasons_max),
    )

    df_j = build_judge_targets(df_raw, mode=args.judge_mode)

    # 这里假设 raw_csv 里有 celebrity_name，作为 term 匹配趋势
    # term 对齐策略可根据你趋势数据实际 term 字段再调整
    if "celebrity_name" not in df_j.columns:
        raise ValueError("原始数据缺少 celebrity_name，无法对齐 Google Trends term")

    df_j["term"] = df_j["celebrity_name"].astype(str)

    df_long = df_j.merge(gfeat, on=["season", "week", "term"], how="left")
    df_long[["g_log_ratio", "g_z_within_sw", "m_s"]] = df_long[["g_log_ratio", "g_z_within_sw", "m_s"]].fillna(0.0)

    # week centered
    df_long["week_c"] = df_long["week"] - df_long.groupby("season")["week"].transform("mean")

    # 舞者 id
    if "ballroom_partner" in df_long.columns:
        df_long["partner_id"] = df_long["ballroom_partner"].astype(str)
    else:
        df_long["partner_id"] = df_long["team_id"].astype(str).str.split("__").str[-1]

    # 明星 id
    df_long["celeb_id"] = df_long["celebrity_name"].astype(str)

    # 你们的身份向量 x_i 通常来自你们特征工程，先在长表里留占位
    # 后续你可以把主模型那套 feature builder 输出并 join 进来
    # 这里用简单示例：年龄等数值列若存在就带上
    keep_cols = ["season", "week", "week_c", "team_id", "celeb_id", "partner_id",
                 "j_total", "j_pct", "y_judge", "g_log_ratio", "g_z_within_sw", "m_s"]
    extra = []
    for c in ["age", "gender", "occupation", "state", "country"]:
        if c in df_long.columns:
            extra.append(c)
    df_out = df_long[keep_cols + extra].copy()

    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df_out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
