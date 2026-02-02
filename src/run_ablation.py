"""Batch ablation runner.

Usage:
python -m src.run_ablations \
  --base_config configs/config_resolved_trial128.yaml \
  --output_root outputs/ablations_trial128 \
  --seeds 13,17,23
"""

import argparse
import csv
import json
import os
from typing import Dict, Any, List

from src.train import train


def _is_complete(run_dir: str) -> bool:
    need = ["model_best.pt", "pred_fan_shares_enriched.csv", "metrics_history.csv"]
    for f in need:
        if not os.path.exists(os.path.join(run_dir, f)):
            return False
    return True


def _parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _exp_list() -> List[Dict[str, Any]]:
    return [
        {"name": "full", "overrides": {}},

        # model ablations (training-level)
        {"name": "no_dzj", "overrides": {"model.perf_use_dzj": False}},
        {"name": "no_r", "overrides": {"model.use_r": False}},
        {"name": "no_phi", "overrides": {"model.use_phi": False, "model.lambda_phi_l1": 0.0, "model.lambda_phi_l2": 0.0}},
        {"name": "no_r_no_phi", "overrides": {"model.use_r": False, "model.use_phi": False, "model.lambda_phi_l1": 0.0, "model.lambda_phi_l2": 0.0}},

        # loss ablations (training-level)
        {"name": "no_Lr", "overrides": {"model.lambda_r": 0.0}},
        {"name": "no_Lrw", "overrides": {"model.lambda_rw": 0.0}},
        {"name": "no_Lr_no_Lrw", "overrides": {"model.lambda_r": 0.0, "model.lambda_rw": 0.0}},

        # phi regularization variants (training-level)
        {"name": "phi_L2_only", "overrides": {"model.phi_reg": "l2", "model.lambda_phi_l1": 0.0}},
        {"name": "phi_L1_only", "overrides": {"model.phi_reg": "l1", "model.lambda_phi_l2": 0.0, "training.weight_decay": 0.0}},
        {"name": "phi_none", "overrides": {"model.phi_reg": "none", "model.lambda_phi_l1": 0.0, "model.lambda_phi_l2": 0.0, "training.weight_decay": 0.0}},

        # alpha modes (training-level)
        {"name": "alpha_variance", "overrides": {"model.alpha_mode": "variance"}},
        {"name": "alpha_entropy", "overrides": {"model.alpha_mode": "entropy", "model.alpha_eps": 1e-8}},
        {"name": "alpha_hhi", "overrides": {"model.alpha_mode": "hhi", "model.alpha_eps": 1e-8}},

        # twist ablation: I_twist ≡ 0
        {"name": "twist_off", "overrides": {"loss.twist_mode": "none"}},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="13,17,23")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    seeds = _parse_seeds(args.seeds)

    summary_path = os.path.join(args.output_root, "ablation_summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "exp_name", "seed", "best_monitor",
                    "BottomKAcc", "PairwiseOrderAcc", "Top1Acc", "MeanMargin", "AvgLogLik", "RuleReproRateWeek",
                    "run_dir",
                ],
            )
            w.writeheader()

    for exp in _exp_list():
        for seed in seeds:
            run_name = f"abl_{exp['name']}__seed{seed}"
            overrides: Dict[str, Any] = {}
            overrides.update(exp.get("overrides", {}))
            overrides["output_root"] = args.output_root
            overrides["run_name"] = run_name
            overrides["training.seed"] = seed
            run_dir = os.path.join(args.output_root, run_name)
            if _is_complete(run_dir):
                print(f"\n=== SKIP exp={exp['name']} seed={seed} (complete) run={run_name} ===")
                continue
            overrides["training.resume_from"] = "last"

            print(f"\n=== RUN exp={exp['name']} seed={seed} run={run_name} ===")
            if args.dry_run:
                continue

            best_val = float(train(config_path=args.base_config, overrides=overrides))

            # run_dir already set above
            best_metrics_path = os.path.join(run_dir, "best_metrics.json")
            last_metrics_path = os.path.join(run_dir, "last_metrics.json")
            if os.path.exists(best_metrics_path):
                with open(best_metrics_path, "r", encoding="utf-8") as f:
                    bm = json.load(f)
            elif os.path.exists(last_metrics_path):
                with open(last_metrics_path, "r", encoding="utf-8") as f:
                    bm = json.load(f)
            else:
                bm = {}

            row = {
                "exp_name": exp["name"],
                "seed": seed,
                "best_monitor": best_val,
                "BottomKAcc": bm.get("BottomKAcc", ""),
                "PairwiseOrderAcc": bm.get("PairwiseOrderAcc", ""),
                "Top1Acc": bm.get("Top1Acc", ""),
                "MeanMargin": bm.get("MeanMargin", ""),
                "AvgLogLik": bm.get("AvgLogLik", ""),
                "RuleReproRateWeek": bm.get("RuleReproRateWeek", ""),
                "run_dir": run_dir,
            }

            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
