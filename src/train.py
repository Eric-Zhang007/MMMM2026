import argparse
import json
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import DWTSDataset
from src.evaluate import calculate_metrics
from src.features import FeatureBuilder, build_all_features
from src.losses import compute_loss, get_prior_loss
from src.model import DWTSModel
from src.utils import (
    get_device,
    load_config,
    save_resolved_config,
    setup_logger,
    write_formulas,
)

DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")


def _set_cpu_threads_leave_one_core() -> None:
    """Set PyTorch intra-op/interop CPU thread counts.

    Default: use (cpu_count - 1) to leave 1 core for other tasks.
    Override: set env var DWTS_TORCH_THREADS (int) to force a specific value.
    This is useful for multi-process Optuna tuning where each worker should be single-threaded.
    """
    forced = os.environ.get("DWTS_TORCH_THREADS", "").strip()
    if forced:
        try:
            threads = int(forced)
            threads = max(1, threads)
        except ValueError:
            threads = None
    else:
        threads = None

    if threads is None:
        cpu = os.cpu_count() or 1
        threads = max(1, cpu - 1)  # leave 1 core for other tasks

    torch.set_num_threads(threads)
    # inter-op threads (PyTorch may require set before any parallel work)
    try:
        torch.set_num_interop_threads(min(threads, 4))
    except Exception:
        pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_resume_path(run_dir: str, resume_from: str) -> Optional[str]:
    if resume_from is None:
        return None
    resume_from = str(resume_from)
    if resume_from.lower() in ("none", ""):
        return None
    if resume_from.lower() in ("last", "best"):
        return os.path.join(run_dir, f"model_{resume_from.lower()}.pt")
    # treat as explicit path
    return resume_from


def _maybe_resume(
    run_dir: str,
    resume_from: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float, int]:
    ckpt_path = _resolve_resume_path(run_dir, resume_from)
    if not ckpt_path or not os.path.exists(ckpt_path):
        logging.info(f"No resume checkpoint loaded. resume_from={resume_from}")
        return 0, -float("inf"), 0

    logging.info(f"Resuming from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_acc = float(ckpt.get("best_acc", -float("inf")))
    patience_counter = int(ckpt.get("patience_counter", 0))
    return start_epoch, best_acc, patience_counter


def _build_optimizer(model: DWTSModel, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with param_groups: weight_decay only on phi, not on theta/u/beta."""
    lr = float(config["training"]["lr"])
    weight_decay_phi = float(config["training"].get("weight_decay", 0.0))

    phi_params = list(model.phi.parameters())
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("phi."):
            continue
        other_params.append(p)

    param_groups = [
        {"params": other_params, "weight_decay": 0.0},
        {"params": phi_params, "weight_decay": weight_decay_phi},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    sched_cfg = config["training"].get("scheduler", {}) or {}
    stype = str(sched_cfg.get("type", "none")).lower()

    if stype == "none":
        return None

    if stype == "plateau":
        factor = float(sched_cfg.get("plateau_factor", 0.5))
        patience = int(sched_cfg.get("plateau_patience", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr
        )

    if stype == "warmup_cosine":
        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        total_steps = int(sched_cfg.get("total_steps", 0))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))

        if total_steps <= 0:
            return None

        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            # cosine decay
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # map to [min_lr, 1.0]
            return (min_lr / optimizer.param_groups[0]["lr"]) + (1.0 - (min_lr / optimizer.param_groups[0]["lr"])) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return None


def export_results(model: DWTSModel, dataset: DWTSDataset, all_feats: torch.Tensor, run_dir: str):
    model.eval()
    results = []
    device = all_feats.device

    with torch.no_grad():
        for week in dataset.panel:
            p_fan, s_total, alpha, _ = model(week, all_feats)
            for i, tid_idx in enumerate(week["teams"]):
                results.append(
                    {
                        "season": int(week["season"]),
                        "week": int(week["week"]),
                        "team_id": dataset.teams[tid_idx],
                        "P_fan": float(p_fan[i].item()),
                        "S_total": float(s_total[i].item()),
                        "alpha": float(alpha.item()),
                    }
                )

    pd.DataFrame(results).to_csv(os.path.join(run_dir, "pred_fan_shares.csv"), index=False)


def train(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> float:
    _set_cpu_threads_leave_one_core()

    config = load_config(DEFAULT_CONFIG_PATH, config_path, overrides)

    run_dir = os.path.join(str(config["output_root"]), str(config["run_name"]))
    _ensure_dir(run_dir)
    _ensure_dir(os.path.join(run_dir, "tb"))

    setup_logger(run_dir)
    save_resolved_config(config, os.path.join(run_dir, "config_resolved.yaml"))
    write_formulas(os.path.join(run_dir, "formulas.txt"), config)

    device = get_device(str(config.get("device", "auto")))
    logging.info(f"Using device: {device}")

    dataset = DWTSDataset(str(config["data_path"]), config)
    fb = FeatureBuilder(dataset.df, config)
    all_feats = build_all_features(dataset, fb).to(device)

    model = DWTSModel(len(dataset.celebrities), len(dataset.partners), fb.dim, config).to(device)
    optimizer = _build_optimizer(model, config)

    start_epoch, best_acc, patience_counter = _maybe_resume(
        run_dir, str(config["training"].get("resume_from", "none")), device, model, optimizer
    )

    scheduler = _build_scheduler(optimizer, config)
    writer = SummaryWriter(os.path.join(run_dir, "tb"))

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    monitor = str(config["training"].get("monitor", "train_ElimTop1Acc"))

    grad_clip_norm = float(config["training"].get("grad_clip_norm", 0.0))
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)

    global_step = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        metrics_list = []

        optimizer.zero_grad(set_to_none=True)

        for wi, week in enumerate(dataset.panel):
            p_fan, s_total, alpha, _ = model(week, all_feats)
            l_val, count = compute_loss(week, p_fan, s_total, config)

            if count <= 0:
                continue

            loss = l_val + get_prior_loss(model, config)
            loss = loss / grad_accum_steps
            loss.backward()

            if (wi + 1) % grad_accum_steps == 0:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * grad_accum_steps

            m = calculate_metrics(week, s_total)
            if m:
                metrics_list.append(m)

            writer.add_scalar("Alpha/alpha", float(alpha.item()), global_step)
            writer.add_scalar("Loss/step", float(loss.item()) * grad_accum_steps, global_step)
            global_step += 1

        avg_top1 = float(sum(x["ElimTop1Acc"] for x in metrics_list) / max(1, len(metrics_list)))
        avg_bottom2 = float(sum(x["Bottom2Acc"] for x in metrics_list) / max(1, len(metrics_list)))
        avg_margin = float(sum(x["Margin"] for x in metrics_list) / max(1, len(metrics_list)))

        logging.info(
            f"Epoch {epoch:04d} | Loss {total_loss:.6f} | "
            f"ElimTop1Acc {avg_top1:.4f} | ElimInBottom2Acc {avg_bottom2:.4f} | MeanMargin {avg_margin:.4f}"
        )

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("ElimTop1Acc/train", avg_top1, epoch)
        writer.add_scalar("ElimInBottom2Acc/train", avg_bottom2, epoch)
        writer.add_scalar("MeanMargin/train", avg_margin, epoch)
        writer.add_scalar("LR/lr", optimizer.param_groups[0]["lr"], epoch)

        # Internal rule: always maximize monitor_val.
        if monitor.lower() == "train_loss":
            monitor_val = -total_loss
        else:
            monitor_val = avg_top1  # default

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_top1)
            else:
                scheduler.step()

        improved = monitor_val > best_acc

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "patience_counter": patience_counter,
            "monitor": monitor,
        }

        torch.save(ckpt, os.path.join(run_dir, "model_last.pt"))

        if improved:
            best_acc = monitor_val
            patience_counter = 0
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, os.path.join(run_dir, "model_best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered. patience={patience}")
            break

    writer.flush()
    writer.close()

    export_results(model, dataset, all_feats, run_dir)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_monitor": best_acc,
                "monitor": monitor,
                "epochs_ran": epoch + 1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return float(best_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=None,
        help="Override config entries, e.g., training.lr=0.001 (repeatable)",
    )
    args = parser.parse_args()

    overrides = None
    if args.override:
        overrides = {}
        for kv in args.override:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            # best-effort type casting
            if v.lower() in ("true", "false"):
                vv: Any = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        vv = float(v)
                    else:
                        vv = int(v)
                except ValueError:
                    vv = v
            overrides[k] = vv

    train(config_path=args.config, overrides=overrides)


if __name__ == "__main__":
    main()
