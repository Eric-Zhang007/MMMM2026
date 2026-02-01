import argparse
import os
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, get_device, set_seed
from src.data import DWTSDataset
from src.features import FeatureBuilder, build_all_features
from src.model import DWTSModel
from src.evaluate import calculate_metrics


def _mean_skip_none(xs: List[Optional[float]]) -> Optional[float]:
    ys = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not ys:
        return None
    return float(np.mean(ys))


def bootstrap_metrics(
    run_dir: str,
    config_path: Optional[str] = None,
    checkpoint: str = "best",
    n_bootstrap: int = 200,
    seed: int = 42,
    device_name: str = "auto",
) -> str:
    """Bootstrap aggregated ranking metrics across weeks."""
    resolved_cfg_path = os.path.join(run_dir, "config_resolved.yaml")
    cfg = load_config("configs/default.yaml", config_path or resolved_cfg_path, overrides=None)

    set_seed(seed)
    device = get_device(device_name if device_name != "auto" else str(cfg.get("device", "auto")))

    dataset = DWTSDataset(str(cfg["data_path"]), cfg)
    fb = FeatureBuilder(dataset.df, cfg)
    all_feats = build_all_features(dataset, fb).to(device)

    model = DWTSModel(
        num_celebs=len(dataset.celebrities),
        num_partners=len(dataset.partners),
        feat_dim=fb.dim,
        num_obs=dataset.num_obs,
        config=cfg,
    ).to(device)

    ckpt_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    per_week = []
    with torch.no_grad():
        for week_data in dataset.panel:
            pf, st, alpha, _ = model(week_data, all_feats, mc_dropout=False, dropout_p=0.0)
            m = calculate_metrics(week_data, st, cfg)
            if m is None:
                continue
            per_week.append(
                {
                    "season": int(week_data["season"]),
                    "week": int(week_data["week"]),
                    "K": int(m.get("K", 0)),
                    "BottomKAcc": float(m["BottomKAcc"]),
                    "PairwiseOrderAcc": m.get("PairwiseOrderAcc", None),
                    "Top1Acc": m.get("Top1Acc", None),
                    "RuleReproRateWeek": float(m.get("RuleReproRateWeek", 0.0)),
                }
            )

    if not per_week:
        raise RuntimeError("No eligible weeks found for bootstrap (need at least one week with eliminated teams).")

    dfw = pd.DataFrame(per_week)
    N = len(dfw)

    rng = np.random.default_rng(seed)
    boot_rows = []
    for b in range(n_bootstrap):
        idx = rng.integers(low=0, high=N, size=N)
        samp = dfw.iloc[idx]

        bottomk = float(samp["BottomKAcc"].mean())
        pairwise = _mean_skip_none(samp["PairwiseOrderAcc"].tolist())
        top1 = _mean_skip_none(samp["Top1Acc"].tolist())
        rule_repro = float(samp["RuleReproRateWeek"].mean())

        boot_rows.append(
            {
                "bootstrap_id": b,
                "BottomKAcc": bottomk,
                "PairwiseOrderAcc": pairwise,
                "Top1Acc": top1,
                "RuleReproRateWeek": rule_repro,
            }
        )

    dfb = pd.DataFrame(boot_rows)

    def _ci(col: str):
        vals = dfb[col].dropna().to_numpy()
        if vals.size == 0:
            return (None, None, None)
        return (float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

    summary = {
        "N_weeks": N,
        "n_bootstrap": n_bootstrap,
        "BottomKAcc_mean_q025_q975": _ci("BottomKAcc"),
        "PairwiseOrderAcc_mean_q025_q975": _ci("PairwiseOrderAcc"),
        "Top1Acc_mean_q025_q975": _ci("Top1Acc"),
        "RuleReproRateWeek_mean_q025_q975": _ci("RuleReproRateWeek"),
    }

    out_dist = os.path.join(run_dir, "bootstrap_metrics_dist.csv")
    out_sum = os.path.join(run_dir, "bootstrap_metrics_summary.json")

    dfb.to_csv(out_dist, index=False)
    import json
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    bootstrap_metrics(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint=args.checkpoint,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        device_name=args.device,
    )
