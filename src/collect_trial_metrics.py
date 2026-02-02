
import argparse
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, get_device, set_seed
from src.data import DWTSDataset
from src.features import FeatureBuilder, build_all_features
from src.model import DWTSModel
from src.evaluate import calculate_metrics


def _safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    ys = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if np.isnan(xf):
            continue
        ys.append(xf)
    if not ys:
        return None
    return float(np.mean(ys))


def _aggregate_like_train(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_weeks_used"] = int(len(metrics_list))

    if metrics_list:
        out["BottomKAcc"] = float(np.mean([x["BottomKAcc"] for x in metrics_list if x.get("BottomKAcc") is not None]))
        out["PairwiseOrderAcc"] = _safe_mean([x.get("PairwiseOrderAcc") for x in metrics_list])
        out["Top1Acc"] = _safe_mean([x.get("Top1Acc") for x in metrics_list])
        out["MeanMargin"] = float(np.mean([x["MeanMargin"] for x in metrics_list if x.get("MeanMargin") is not None]))
        out["AvgLogLik"] = _safe_mean([x.get("AvgLogLik") for x in metrics_list])
        out["RuleReproRateWeek"] = float(np.mean([x["RuleReproRateWeek"] for x in metrics_list if x.get("RuleReproRateWeek") is not None]))
        out["avg_K"] = float(np.mean([x.get("K", 0) for x in metrics_list]))
    else:
        out["BottomKAcc"] = 0.0
        out["PairwiseOrderAcc"] = None
        out["Top1Acc"] = None
        out["MeanMargin"] = 0.0
        out["AvgLogLik"] = None
        out["RuleReproRateWeek"] = 0.0
        out["avg_K"] = 0.0

    return out


def _discover_trial_dirs(tune_root: str) -> List[Tuple[Optional[int], str]]:
    dirs: List[Tuple[Optional[int], str]] = []
    for name in os.listdir(tune_root):
        full = os.path.join(tune_root, name)
        if not os.path.isdir(full):
            continue
        if "trial" not in name.lower():
            continue
        m = re.search(r"trial[_-]?(\d+)", name.lower())
        trial_num = int(m.group(1)) if m else None
        dirs.append((trial_num, full))
    dirs.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 10**18, x[1]))
    return dirs


def _load_cfg_for_run(run_dir: str) -> Dict[str, Any]:
    resolved = os.path.join(run_dir, "config_resolved.yaml")
    if os.path.exists(resolved):
        return load_config("configs/default.yaml", resolved, overrides=None)
    cfg2 = os.path.join(run_dir, "config.yaml")
    return load_config("configs/default.yaml", cfg2 if os.path.exists(cfg2) else None, overrides=None)


def _load_model(run_dir: str, cfg: Dict[str, Any], dataset: DWTSDataset, fb: FeatureBuilder, all_feats: torch.Tensor,
                checkpoint: str, device: torch.device):
    ckpt_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
    if not os.path.exists(ckpt_path):
        return None, None

    model = DWTSModel(
        num_celebs=len(dataset.celebrities),
        num_partners=len(dataset.partners),
        feat_dim=fb.dim,
        num_obs=dataset.num_obs,
        config=cfg,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate_run_dir(run_dir: str, checkpoint: str, device_name: str) -> Optional[Dict[str, Any]]:
    cfg = _load_cfg_for_run(run_dir)

    seed = int(cfg.get("training", {}).get("seed", cfg.get("seed", 42)))
    set_seed(seed)

    device = get_device(device_name if device_name != "auto" else str(cfg.get("device", "auto")))

    dataset = DWTSDataset(str(cfg["data_path"]), cfg)
    fb = FeatureBuilder(dataset.df, cfg)
    all_feats = build_all_features(dataset, fb).to(device)

    model, ckpt = _load_model(run_dir, cfg, dataset, fb, all_feats, checkpoint, device)
    if model is None or ckpt is None:
        return None

    metrics_list: List[Dict[str, Any]] = []
    with torch.no_grad():
        for week_data in dataset.panel:
            p_fan, s_total, alpha, _ = model(week_data, all_feats, mc_dropout=False, dropout_p=0.0)
            m = calculate_metrics(week_data, s_total.detach(), cfg)
            if m is not None:
                metrics_list.append(m)

    agg = _aggregate_like_train(metrics_list)

    monitor_name = str(ckpt.get("monitor_name", cfg.get("training", {}).get("monitor", "train_BottomKAcc")))
    ckpt_best_monitor = ckpt.get("best_monitor", None)

    row: Dict[str, Any] = {}
    row.update(agg)
    row["run_dir"] = run_dir
    row["checkpoint"] = checkpoint
    row["monitor_name"] = monitor_name
    row["ckpt_best_monitor"] = (None if ckpt_best_monitor is None else float(ckpt_best_monitor))

    monitor_key = monitor_name.replace("train_", "")
    row["monitor_value_recomputed"] = row.get(monitor_key, None)

    row["seed"] = seed
    row["patience"] = int(cfg.get("training", {}).get("patience", -1))
    row["lr"] = float(cfg.get("training", {}).get("lr", np.nan))
    row["weight_decay"] = float(cfg.get("training", {}).get("weight_decay", np.nan))

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune_root", type=str, required=True, help="包含各 trial 子目录的目录")
    ap.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--sort_by", type=str, default="ckpt_best_monitor",
                    help="排序列。推荐 ckpt_best_monitor 或 BottomKAcc 或 PairwiseOrderAcc")
    ap.add_argument("--descending", action="store_true", help="默认从小到大。加此参数改为从大到小")
    args = ap.parse_args()

    trial_dirs = _discover_trial_dirs(args.tune_root)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for trial_num, run_dir in trial_dirs:
        res = evaluate_run_dir(run_dir=run_dir, checkpoint=args.checkpoint, device_name=args.device)
        if res is None:
            skipped += 1
            continue
        res["trial"] = trial_num
        rows.append(res)

    if not rows:
        print("没有找到可用的 trial 结果，检查 tune_root 里是否有 model_best.pt 或 model_last.pt。")
        return

    df = pd.DataFrame(rows)

    sort_col = args.sort_by
    if sort_col not in df.columns:
        print(f"sort_by={sort_col} 不在列里，可选列包括：{list(df.columns)}")
        return

    df = df.sort_values(by=sort_col, ascending=not args.descending).reset_index(drop=True)

    out_csv = args.out_csv or os.path.join(args.tune_root, f"trial_metrics_sorted_by_{sort_col}.csv")
    df.to_csv(out_csv, index=False)

    print(f"已评估 {len(df)} 个 trial，跳过 {skipped} 个无 checkpoint 的目录。")
    print(f"汇总已保存到: {out_csv}")
    print("Top 10:")
    cols_show = [
        "trial",
        "ckpt_best_monitor",
        "monitor_name",
        "monitor_value_recomputed",
        "BottomKAcc",
        "PairwiseOrderAcc",
        "Top1Acc",
        "MeanMargin",
        "RuleReproRateWeek",
        "AvgLogLik",
        "avg_K",
        "lr",
        "weight_decay",
        "patience",
        "run_dir",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    print(df[cols_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
