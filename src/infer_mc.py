import argparse
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, get_device, set_seed
from src.data import DWTSDataset
from src.features import FeatureBuilder, build_all_features
from src.model import DWTSModel


def infer_mc(
    run_dir: str,
    config_path: Optional[str] = None,
    checkpoint: str = "best",
    mc_dropout: bool = True,
    dropout_p: float = 0.1,
    mc_passes: int = 1000,
    device_name: str = "auto",
) -> str:
    """MC-dropout inference producing per team-week CI for P_fan (and S_total optionally)."""
    resolved_cfg_path = os.path.join(run_dir, "config_resolved.yaml")
    cfg = load_config("configs/default.yaml", config_path or resolved_cfg_path, overrides=None)

    set_seed(int(cfg.get("seed", 42)))
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

    rows = []
    with torch.no_grad():
        for week_data in dataset.panel:
            # deterministic baseline
            pf0, st0, alpha, _ = model(week_data, all_feats, mc_dropout=False, dropout_p=0.0)

            if mc_dropout and mc_passes > 1 and dropout_p > 0.0:
                samples = []
                for _ in range(mc_passes):
                    pf_s, st_s, _, _ = model(week_data, all_feats, mc_dropout=True, dropout_p=dropout_p)
                    samples.append(pf_s.unsqueeze(0))
                pf_samples = torch.cat(samples, dim=0)  # [S, n_team]
                pf_mean = pf_samples.mean(dim=0)
                pf_std = pf_samples.std(dim=0, unbiased=False)
                lo = torch.quantile(pf_samples, 0.025, dim=0)
                hi = torch.quantile(pf_samples, 0.975, dim=0)
            else:
                pf_mean = pf0
                pf_std = torch.zeros_like(pf0)
                lo = pf0
                hi = pf0

            teams_idx = week_data["teams"]
            team_ids = [dataset.teams[i] for i in teams_idx]

            for i, tid in enumerate(team_ids):
                rows.append(
                    {
                        "season": int(week_data["season"]),
                        "week": int(week_data["week"]),
                        "team_id": tid,
                        "alpha": float(alpha.item()),
                        "P_fan": float(pf0[i].item()),
                        "P_fan_mean": float(pf_mean[i].item()),
                        "P_fan_std": float(pf_std[i].item()),
                        "P_fan_q025": float(lo[i].item()),
                        "P_fan_q975": float(hi[i].item()),
                        "S_total": float(st0[i].item()),
                    }
                )

    out_path = os.path.join(run_dir, "infer_mc_pfan.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--mc_dropout", action="store_true")
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--mc_passes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    infer_mc(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint=args.checkpoint,
        mc_dropout=args.mc_dropout,
        dropout_p=args.dropout_p,
        mc_passes=args.mc_passes,
        device_name=args.device,
    )
