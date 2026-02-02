import argparse
import json
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


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


def nakagawa_r2(result, exog, groups, exog_re):
    # 近似计算 marginal/conditional R2
    # var_fixed = Var(Xb)
    xb = exog @ result.fe_params.values
    var_fixed = float(np.var(xb, ddof=0))

    # var_random 近似为随机效应方差之和
    var_random = 0.0
    if result.cov_re is not None:
        var_random = float(np.trace(result.cov_re))

    var_resid = float(result.scale)
    denom = var_fixed + var_random + var_resid
    if denom <= 0:
        return {"marginal_R2": np.nan, "conditional_R2": np.nan}

    return {
        "marginal_R2": var_fixed / denom,
        "conditional_R2": (var_fixed + var_random) / denom,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, required=True)
    ap.add_argument("--use_week_fixed_trend", action="store_true")
    ap.add_argument("--use_gtrend", action="store_true", help="是否启用趋势项，默认关")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    df = df.dropna(subset=["y_judge", "partner_id", "celeb_id", "season"])

    y = df["y_judge"].astype(float)

    # 固定效应 X
    X_parts = []

    # 截距
    X_parts.append(pd.Series(1.0, index=df.index, name="intercept"))

    # 明星身份向量 x_i
    # 这里假设你已在 df 中拼入 x0,x1,... 之类列名
    x_cols = [c for c in df.columns if c.startswith("x_")]
    if x_cols:
        X_parts.append(df[x_cols].astype(float))
    else:
        # 没有身份向量时，至少加 celeb 作为固定效应会很重，不建议
        pass

    # 可选的趋势项，只在 m_s=1 的季有效
    if args.use_gtrend:
        X_parts.append((df["g_z_within_sw"] * df["m_s"]).rename("gtrend"))

    # 周趋势固定效应，你说倾向不要，我默认关
    if args.use_week_fixed_trend:
        X_parts.append(df["week_c"].astype(float).rename("week_c"))

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    # 随机效应
    # 分组: 舞者
    groups = df["partner_id"].astype(str)

    # 随机效应设计矩阵
    # 你希望只保留带教能力，随机截距 + 随机斜率是最典型
    Z = pd.DataFrame({
        "re_intercept": 1.0,
        "re_week_c": df["week_c"].astype(float),
    })

    # 额外吸收 season 与 celeb 的偏差
    # MixedLM 只支持一个 groups，额外的随机截距用 fixed effects dummy 或者先残差化
    # 为了可运行与可解释，这里把 season 与 celeb 做成固定效应 dummy，规模可能较大但还能跑
    season_dum = pd.get_dummies(df["season"].astype(int), prefix="season", drop_first=True)
    celeb_dum = pd.get_dummies(df["celeb_id"].astype(str), prefix="celeb", drop_first=True)
    X_big = pd.concat([X, season_dum, celeb_dum], axis=1)

    model = MixedLM(endog=y, exog=X_big, groups=groups, exog_re=Z)
    result = model.fit(reml=True, method="lbfgs", maxiter=200)

    # 预测
    yhat = result.fittedvalues.values

    metrics = {
        "RMSE": rmse(y.values, yhat),
        "MAE": mae(y.values, yhat),
        "R2": r2(y.values, yhat),
        "corr": corr(y.values, yhat),
    }
    metrics.update(nakagawa_r2(result, X_big.values, groups.values, Z.values))

    with open(f"{args.out_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(f"{args.out_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())

    # 输出舞者 BLUP
    re = result.random_effects  # dict partner -> array
    blup_rows = []
    for pid, v in re.items():
        blup_rows.append({
            "partner_id": pid,
            "b_p_intercept": float(v[0]),
            "k_p_week_slope": float(v[1]) if len(v) > 1 else np.nan,
        })
    pd.DataFrame(blup_rows).sort_values("b_p_intercept", ascending=False).to_csv(
        f"{args.out_prefix}_partner_blup.csv", index=False
    )

    print("Saved:", f"{args.out_prefix}_metrics.json", f"{args.out_prefix}_partner_blup.csv")


if __name__ == "__main__":
    main()
