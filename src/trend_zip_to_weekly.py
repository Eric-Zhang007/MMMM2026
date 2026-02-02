import argparse
import os
import re
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _list_csv_files(trend_path: str) -> List[str]:
    if os.path.isdir(trend_path):
        return [os.path.join(trend_path, f) for f in os.listdir(trend_path) if f.lower().endswith(".csv")]
    if trend_path.lower().endswith(".zip"):
        with zipfile.ZipFile(trend_path) as z:
            return [f for f in z.namelist() if f.lower().endswith(".csv")]
    raise ValueError(f"trend_path must be a directory or .zip: {trend_path}")


def _read_csv_any(trend_path: str, member: str) -> pd.DataFrame:
    # member is full path for dir mode; is internal name for zip mode
    if os.path.isdir(trend_path):
        return pd.read_csv(member)
    with zipfile.ZipFile(trend_path) as z:
        with z.open(member) as f:
            return pd.read_csv(f)


def _infer_season_from_filename(name: str) -> int:
    # supports patterns like 21_all.csv, season21_all.csv, S21_all.csv
    base = os.path.basename(name)
    m = re.search(r"(?:season|s)?\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot infer season from filename: {name}")
    return int(m.group(1))


def _extract_celeb(col: str) -> str:
    # Example: "Bindi Irwin DWTS: (美国)" -> "Bindi Irwin"
    # Also handles "DWTS: (United States)" etc
    m = re.match(r"^(.*?)\s+DWTS\s*:\s*\(.*\)\s*$", str(col))
    if m:
        return m.group(1).strip()
    return str(col).strip()


def infer_weeks_per_season(dwts_data_csv: str) -> Dict[int, int]:
    df = pd.read_csv(dwts_data_csv)
    # detect week numbers from column names like week3_judge2_score
    week_nums = sorted({int(m.group(1)) for c in df.columns for m in [re.match(r"^week(\d+)_judge\d+_score$", c)] if m})
    if not week_nums:
        raise ValueError("No week*_judge*_score columns found in dwts data csv.")

    out: Dict[int, int] = {}
    for season, g in df.groupby("season"):
        max_w = 0
        for w in week_nums:
            cols = [c for c in df.columns if re.match(rf"^week{w}_judge\d+_score$", c)]
            if not cols:
                continue
            vals = g[cols].to_numpy().astype(float)
            # week exists if there is at least one non-NaN value
            if np.isfinite(vals).any():
                max_w = max(max_w, w)
        out[int(season)] = int(max_w)
    return out


def season_week_binning(dates: pd.Series, n_weeks: int) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Series]:
    # The user stated each trend table starts ~3 days before season start and ends ~3 days after season end.
    d0 = pd.to_datetime(dates.min())
    d1 = pd.to_datetime(dates.max())
    season_start = d0 + pd.Timedelta(days=3)
    season_end = d1 - pd.Timedelta(days=3)

    # Use 7-day bins from season_start.
    # Week k is [season_start + 7*(k-1), season_start + 7*k)
    dt = pd.to_datetime(dates)
    week_idx = ((dt - season_start).dt.days // 7) + 1
    week_idx = week_idx.clip(lower=1, upper=n_weeks)
    return season_start, season_end, week_idx.astype(int)


def process_one(df_wide: pd.DataFrame, season: int, n_weeks: int, file_name: str) -> pd.DataFrame:
    if df_wide.shape[1] < 2:
        raise ValueError(f"{file_name}: need Time + at least one celeb column")

    time_col = df_wide.columns[0]
    df_wide = df_wide.rename(columns={time_col: "Time"})
    df_wide["Time"] = pd.to_datetime(df_wide["Time"])

    season_start, season_end, week = season_week_binning(df_wide["Time"], n_weeks)
    df_wide["week"] = week

    # Melt to long
    value_cols = [c for c in df_wide.columns if c not in ["Time", "week"]]
    df_long = df_wide.melt(id_vars=["Time", "week"], value_vars=value_cols, var_name="raw_col", value_name="trend")

    df_long["celebrity_name"] = df_long["raw_col"].map(_extract_celeb)
    df_long["season"] = int(season)

    # Clean numeric
    df_long["trend"] = pd.to_numeric(df_long["trend"], errors="coerce")

    agg = df_long.groupby(["season", "week", "celebrity_name"], as_index=False).agg(
        trend_mean=("trend", "mean"),
        trend_sum=("trend", "sum"),
        trend_max=("trend", "max"),
        n_days=("trend", "count"),
    )
    agg["season_start_est"] = season_start.date().isoformat()
    agg["season_end_est"] = season_end.date().isoformat()
    agg["source_file"] = os.path.basename(file_name)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trend_path", type=str, required=True, help="directory or zip containing season *_all.csv files")
    ap.add_argument("--dwts_data_csv", type=str, default="2026_MCM_Problem_C_Data.csv", help="used to infer number of weeks per season")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="all.csv", help="only process csv files whose name contains this substring")
    args = ap.parse_args()

    weeks_map = infer_weeks_per_season(args.dwts_data_csv)

    files = _list_csv_files(args.trend_path)
    files = [f for f in files if args.pattern.lower() in os.path.basename(f).lower()]

    if not files:
        raise RuntimeError(f"No csv files matched pattern '{args.pattern}' in {args.trend_path}")

    rows = []
    for f in files:
        season = _infer_season_from_filename(f)
        n_weeks = weeks_map.get(season, None)
        if not n_weeks or n_weeks <= 0:
            # skip seasons we cannot infer
            continue
        df_wide = _read_csv_any(args.trend_path, f)
        rows.append(process_one(df_wide, season, n_weeks, f))

    if not rows:
        raise RuntimeError("No season files processed. Check filenames and dwts_data_csv.")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
