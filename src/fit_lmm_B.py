import argparse
import json
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


def zscore(s):
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def safe_logit(p, eps=1e-6):
    p2 = np.clip(p, eps, 1 - eps)
    return np.log(p2 / (1 - p2))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def corr(y, yhat):
    if np.std(y) == 0 or np.std(yhat) == 0:
        return np.nan
    return float(np.corrcoef(y, yhat)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True, help="由 prepare_lmm_dataset.py 输出的长表，或你自己拼的表")
    ap.add_argument("--pfan_csv", type=str, required=True, help="pred_fan_shares_enriched.csv 之类，包含 P_fan 与 D_pf_minus_j")
    ap.add_argument("--out_prefix", type=str, required=True)
    ap.add_argument("--y_mode", type=str, default="D",
                    choices=["D", "logit_pfan"], help="D = P_fan - J_pct 或 logit(P_fan)")
    ap.add_argument("--use_gtrend", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)

    p = pd.read_csv(args.pfan_csv)
    # 需要列名适配，你们导出里通常有 season, week, team_id, P_fan, D_pf_minus_j
    cols = {c.lower(): c for c in p.columns}
    need = ["season", "week", "team_id"]
    for k in need:
        if k not in cols:
            raise ValueError(f"pfan_csv 缺少 {k}")

    # P_fan 列名兼容
    pfan_col = None
    for cand in ["p_fan", "pfan", "p_fan_mean", "p_fan_hat"]:
        if cand in cols:
            pfan_col = cols[cand]
            break
    if pfan_col is None:
        # 常见是 P_fan
        if "p_fan" in cols:
            pfan_col = cols["p_fan"]
        elif "p_fan" in p.columns:
            pfan_col = "p_fan"
        elif "P_fan" in p.columns:
            pfan_col = "P_fan"
        else:
            raise ValueError("pfan_csv 找不到 P_fan 列")

    d_col = None
    for cand in ["d_pf_minus_j", "d", "pfan_minus_j", "d_pf_j"]:
        if cand in cols:
            d_col = cols[cand]
            break
    if d_col is None and "D_pf_minus_j" in p.columns:
        d_col = "D_pf_minus_j"

    p2 = pd.DataFrame({
        "season": p[cols["season"]].astype(int),
        "week": p[cols["week"]].astype(int),
        "team_id": p[cols["team_id"]].astype(str),
        "P_fan": pd.to_numeric(p[pfan_col], errors="coerce"),
    })
    if d_col is not None:
        p2["D"] = pd.to_numeric(p[d_col], errors="coerce")

    df["team_id"] = df["team_id"].astype(str)
    df = df.merge(p2, on=["season", "week", "team_id"], how="left")
    df = df.dropna(subset=["P_fan", "partner_id", "celeb_id"])

    if args.y_mode == "D":
        if "D" not in df.columns or df["D"].isna().all():
            # 反推 D 也行: D = P_fan - J_pct
            if "j_pct" in df.columns:
                df["D"] = df["P_fan"] - df["j_pct"]
            else:
                raise ValueError("无法构造 D，缺少 D 列且缺少 j_pct")
        y = df["D"].astype(float)
        # 标准化一下，便于系数解释
        y = df.groupby("season")["D"].transform(zscore)
    else:
        y = safe_logit(df["P_fan"].astype(float).values)
        y = pd.Series(y, index=df.index)
        y = y.groupby(df["season"]).transform(zscore)

    # 固定效应
    X_parts = [pd.Series(1.0, index=df.index, name="intercept")]
    x_cols = [c for c in df.columns if c.startswith("x_")]
    if x_cols:
        X_parts.append(df[x_cols].astype(float))

    if args.use_gtrend:
        X_parts.append((df["g_z_within_sw"] * df["m_s"]).rename("gtrend"))

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    # 舞者随机效应仍可放，但解释要谨慎
    groups = df["partner_id"].astype(str)
    Z = pd.DataFrame({
        "re_intercept": 1.0,
        "re_week_c": df["week_c"].astype(float),
    })

    season_dum = pd.get_dummies(df["season"].astype(int), prefix="season", drop_first=True)
    celeb_dum = pd.get_dummies(df["celeb_id"].astype(str), prefix="celeb", drop_first=True)
    X_big = pd.concat([X, season_dum, celeb_dum], axis=1)

    model = MixedLM(endog=y.astype(float), exog=X_big, groups=groups, exog_re=Z)
    result = model.fit(reml=True, method="lbfgs", maxiter=200)

    yhat = result.fittedvalues.values
    metrics = {
        "RMSE": rmse(y.values, yhat),
        "MAE": mae(y.values, yhat),
        "R2": r2(y.values, yhat),
        "corr": corr(y.values, yhat),
        "note": "B 版是诊断模型，因变量为模型分解产物，舞者效应不可直接解释为现实世界舞者影响力",
    }

    with open(f"{args.out_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(f"{args.out_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())

    print("Saved:", f"{args.out_prefix}_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
