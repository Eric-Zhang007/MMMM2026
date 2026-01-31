# src/train.py
import os
import json
import logging
from typing import Any, Dict, Optional, Tuple, List

import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from src.data import DWTSDataset
from src.features import FeatureBuilder, build_all_features
from src.model import DWTSModel
from src.losses import compute_loss, get_prior_loss
from src.evaluate import calculate_metrics
from src.utils import (
    load_config,
    save_resolved_config,
    setup_logger,
    get_device,
    write_formulas,
)


def _set_cpu_threads_leave_one_core() -> None:
    cpu = os.cpu_count() or 1
    threads = max(1, cpu - 1)  # leave 1 core for other tasks
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except Exception:
        pass


def _resolve_resume_path(run_dir: str, resume_from: str) -> str:
    if resume_from is None:
        return ""
    resume_from = str(resume_from).strip()
    if resume_from == "" or resume_from.lower() == "none":
        return ""
    if resume_from.lower() == "last":
        return os.path.join(run_dir, "model_last.pt")
    if resume_from.lower() == "best":
        return os.path.join(run_dir, "model_best.pt")
    return resume_from  # treat as explicit path


def _init_optimizer(model: DWTSModel, config: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(config["training"]["lr"])
    wd = float(config["training"].get("weight_decay", 0.0))

    # Recommended: apply weight_decay only to phi; theta/u/beta use explicit priors in get_prior_loss.
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.phi.parameters()), "weight_decay": wd},
            {"params": list(model.theta.parameters()), "weight_decay": 0.0},
            {"params": list(model.u.parameters()), "weight_decay": 0.0},
            {"params": [model.beta], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    return optimizer


def _init_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    sched_cfg = config["training"]["scheduler"]
    sched_type = str(sched_cfg.get("type", "none")).lower()

    if sched_type == "none":
        return None, "none"

    if sched_type == "plateau":
        plateau_factor = float(sched_cfg.get("plateau_factor", 0.5))
        plateau_patience = int(sched_cfg.get("plateau_patience", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
            verbose=True,
        )
        return scheduler, "plateau"

    if sched_type == "warmup_cosine":
        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        total_steps = int(sched_cfg.get("total_steps", config["training"]["epochs"]))

        def lr_lambda(ep: int) -> float:
            # epoch-level warmup+cosine
            if total_steps <= 0:
                return 1.0
            if warmup_steps > 0 and ep < warmup_steps:
                return float(ep + 1) / float(max(1, warmup_steps))
            # cosine decay from 1 to 0
            denom = max(1, total_steps - warmup_steps)
            progress = float(ep - warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.1415926535))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, "warmup_cosine"

    raise ValueError(f"Unknown scheduler type: {sched_type}")


def _maybe_resume(
    model: DWTSModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    sched_type: str,
    run_dir: str,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[int, float, int]:
    resume_from = config["training"].get("resume_from", "none")
    ckpt_path = _resolve_resume_path(run_dir, resume_from)
    if not ckpt_path or not os.path.exists(ckpt_path):
        logging.info(f"No resume checkpoint loaded. resume_from={resume_from}")
        return 0, -1.0, 0

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler", None) is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            logging.info("Scheduler state not loaded; continuing with fresh scheduler.")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_acc = float(ckpt.get("best_acc", -1.0))
    patience_counter = int(ckpt.get("patience_counter", 0))

    logging.info(f"Resumed from {ckpt_path} | start_epoch={start_epoch} best_acc={best_acc}")
    return start_epoch, best_acc, patience_counter


def export_results(model: DWTSModel, dataset: DWTSDataset, all_feats: torch.Tensor, run_dir: str) -> None:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for week in dataset.panel:
            p_fan, s_total, alpha, _ = model(week, all_feats)
            for i, tid_idx in enumerate(week["teams"]):
                rows.append(
                    {
                        "season": int(week["season"]),
                        "week": int(week["week"]),
                        "team_id": dataset.teams[tid_idx],
                        "P_fan": float(p_fan[i].item()),
                        "S_total": float(s_total[i].item()),
                        "alpha": float(alpha.item()),
                    }
                )
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "pred_fan_shares.csv"), index=False)


def train(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> float:
    _set_cpu_threads_leave_one_core()

    config = load_config("configs/default.yaml", config_path, overrides)

    run_dir = os.path.join(config["output_root"], config["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)

    setup_logger(run_dir)
    save_resolved_config(config, os.path.join(run_dir, "config_resolved.yaml"))
    write_formulas(os.path.join(run_dir, "formulas.txt"), config)

    device = get_device(config["device"])
    logging.info(f"Device: {device}")

    dataset = DWTSDataset(config["data_path"], config)
    fb = FeatureBuilder(dataset.df, config)
    all_feats = build_all_features(dataset, fb).to(device)

    model = DWTSModel(len(dataset.celebrities), len(dataset.partners), fb.dim, config).to(device)
    optimizer = _init_optimizer(model, config)

    scheduler, sched_type = _init_scheduler(optimizer, config)

    start_epoch, best_acc, patience_counter = _maybe_resume(
        model, optimizer, scheduler, sched_type, run_dir, device, config
    )

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    history: List[Dict[str, Any]] = []

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
    accum_steps = int(config["training"].get("grad_accum_steps", 1))
    accum_steps = max(1, accum_steps)

    monitor = str(config["training"].get("monitor", "train_ElimTop1Acc"))
    maximize = monitor.lower() != "train_loss"

    for epoch in range(start_epoch, epochs):
        model.train()

        total_loss = 0.0
        n_updates = 0
        metrics_list: List[Dict[str, float]] = []

        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for week in dataset.panel:
            p_fan, s_total, alpha, _ = model(week, all_feats)

            loss_core, base_count = compute_loss(week, p_fan, s_total, config)
            if base_count == 0:
                continue  # no elimination -> skip updates

            prior = get_prior_loss(model, config)
            loss = loss_core + prior

            # gradient accumulation (scale loss to keep effective lr stable)
            (loss / accum_steps).backward()
            total_loss += float(loss.item())

            m = calculate_metrics(week, s_total)
            if m is not None:
                metrics_list.append(m)

            accum_counter += 1
            if accum_counter >= accum_steps:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_updates += 1
                accum_counter = 0

        # leftover grads
        if accum_counter > 0:
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_updates += 1

        if metrics_list:
            avg_top1 = sum(x["ElimTop1Acc"] for x in metrics_list) / len(metrics_list)
            avg_bottom2 = sum(x.get("Bottom2Acc", 0.0) for x in metrics_list) / len(metrics_list)
            avg_margin = sum(x.get("Margin", 0.0) for x in metrics_list) / len(metrics_list)
        else:
            avg_top1, avg_bottom2, avg_margin = 0.0, 0.0, 0.0

        lr_now = float(optimizer.param_groups[0]["lr"])

        # tensorboard scalars
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("ElimTop1Acc/train", avg_top1, epoch)
        writer.add_scalar("ElimInBottom2Acc/train", avg_bottom2, epoch)
        writer.add_scalar("MeanMargin/train", avg_margin, epoch)
        writer.add_scalar("LR/train", lr_now, epoch)

        # training log line
        logging.info(
            f"Epoch {epoch} | loss={total_loss:.6f} | top1={avg_top1:.4f} | bottom2={avg_bottom2:.4f} | "
            f"margin={avg_margin:.4f} | lr={lr_now:.6g} | updates={n_updates}"
        )

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss,
                "train_ElimTop1Acc": avg_top1,
                "train_ElimInBottom2Acc": avg_bottom2,
                "train_MeanMargin": avg_margin,
                "lr": lr_now,
                "updates": n_updates,
            }
        )

        # scheduler step
        if scheduler is not None:
            if sched_type == "plateau":
                scheduler.step(avg_top1)
            else:
                scheduler.step()

        # early stopping monitor value
        if monitor.lower() == "train_loss":
            monitor_val = -total_loss  # convert to maximize
        else:
            monitor_val = avg_top1

        improved = monitor_val > best_acc if maximize else monitor_val < best_acc
        if improved:
            best_acc = monitor_val
            patience_counter = 0
        else:
            patience_counter += 1

        # save checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": float(best_acc),
            "patience_counter": int(patience_counter),
        }
        torch.save(ckpt, os.path.join(run_dir, "model_last.pt"))
        if improved:
            torch.save(ckpt, os.path.join(run_dir, "model_best.pt"))

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    writer.flush()
    writer.close()

    # export training history and metrics
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "tb_scalars.csv"), index=False)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"best_monitor": float(best_acc)}, f, indent=2)

    export_results(model, dataset, all_feats, run_dir)
    return float(best_acc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None, help="none|last|best|path (overrides config)")
    args = parser.parse_args()

    overrides = {}
    if args.resume_from is not None:
        overrides["training.resume_from"] = args.resume_from

    train(config_path=args.config, overrides=overrides if overrides else None)
