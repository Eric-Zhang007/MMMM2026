"""Inference-only alpha extremes.

Usage:
python -m src.infer_force_alpha --run_dir outputs/.../abl_full__seed13
"""

import argparse
import os
import shutil
from typing import Any, Dict, Tuple

import torch

from src.utils import load_config
from src.data import DWTSDataset
from src.features import FeatureBuilder, build_all_features
from src.model import DWTSModel
from src.train import export_results


def _load_model_and_data(run_dir: str, cfg: Dict[str, Any]) -> Tuple[DWTSModel, DWTSDataset, torch.Tensor, FeatureBuilder]:
    dataset = DWTSDataset(str(cfg["data_path"]), cfg)
    fb = FeatureBuilder(dataset.df, cfg)

    device = torch.device(str(cfg["device"]))
    all_feats = build_all_features(dataset, fb).to(device)

    model = DWTSModel(
        num_celebs=len(dataset.celebrities),
        num_partners=len(dataset.partners),
        feat_dim=fb.dim,
        num_obs=dataset.num_obs,
        config=cfg,
    ).to(device)

    ckpt = torch.load(os.path.join(run_dir, "model_best.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dataset, all_feats, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config_resolved.yaml")
    cfg = load_config("configs/default.yaml", cfg_path, overrides=None)

    for fa in [0.0, 1.0]:
        out_dir = os.path.join(args.run_dir, f"infer_alpha{int(fa)}")
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(cfg_path, os.path.join(out_dir, "config_resolved.yaml"))

        cfg2 = dict(cfg)
        cfg2.setdefault("inference", {})
        cfg2["inference"]["force_alpha"] = fa

        model, dataset, all_feats, fb = _load_model_and_data(args.run_dir, cfg2)
        export_results(model, dataset, all_feats, fb, cfg2, out_dir)
        print(f"Exported forced-alpha={fa} to {out_dir}")


if __name__ == "__main__":
    main()
