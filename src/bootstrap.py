"""Deprecated wrapper.

Kept for backward compatibility with older runs.
Prefer:
  python -m src.infer_mc --run_dir <RUN_DIR> --mc_dropout --mc_passes 1000 --dropout_p 0.2
  python -m src.bootstrap_metrics --run_dir <RUN_DIR> --n_bootstrap 200
"""

import argparse
from src.infer_mc import infer_mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--mc_passes", type=int, default=200)
    parser.add_argument("--mc_dropout_p", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    infer_mc(
        run_dir=args.run_dir,
        config_path=args.config,
        checkpoint=args.checkpoint,
        mc_dropout=True,
        dropout_p=float(args.mc_dropout_p),
        mc_passes=int(args.mc_passes),
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
