import os
import json
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from src.utils import load_config, save_resolved_config, setup_logger, get_device, write_formulas, set_seed
from src.data import DWTSDataset
from src.features import FeatureBuilder, build_all_features
from src.model import DWTSModel
from src.losses import compute_loss, get_prior_loss
from src.evaluate import calculate_metrics


def _make_run_dir(cfg: Dict[str, Any]) -> str:
    out_root = str(cfg["output_root"])
    run_name = str(cfg["run_name"])
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    return run_dir


def _build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    AdamW param_groups:
    - weight_decay ONLY on phi
    - theta/u/beta/r no weight_decay
    """
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))

    phi_params = list(model.phi.parameters())
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("phi."):
            continue
        other_params.append(p)

    param_groups = [
        {"params": other_params, "weight_decay": 0.0},
        {"params": phi_params, "weight_decay": wd},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    sch = cfg.get("training", {}).get("scheduler", {})
    if str(sch.get("type", "none")).lower() == "plateau":
        factor = float(sch.get("plateau_factor", 0.5))
        patience = int(sch.get("plateau_patience", 5))
        min_lr = float(sch.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr
        )
    return None


def _safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    ys = [x for x in xs if x is not None and not (isinstance(x, float) and (np.isnan(x)))]
    if not ys:
        return None
    return float(np.mean(ys))


def train(config_path: Optional[str] = None,
          overrides: Optional[Dict[str, Any]] = None,
          panel_indices: Optional[List[int]] = None) -> float:
    cfg = load_config("configs/default.yaml", config_path, overrides)
    run_dir = _make_run_dir(cfg)
    setup_logger(run_dir)

    save_resolved_config(cfg, os.path.join(run_dir, "config_resolved.yaml"))
    write_formulas(os.path.join(run_dir, "formulas.txt"), cfg)

    seed = int(cfg.get("training", {}).get("seed", cfg.get("seed", 42)))
    set_seed(seed)

    device = get_device(str(cfg["device"]))

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

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    writer = SummaryWriter(os.path.join(run_dir, "tb")) if SummaryWriter is not None else None

    start_epoch = 0
    best_monitor = -1e18
    monitor_name = str(cfg.get("training", {}).get("monitor", "train_BottomKAcc"))

    # persistent epoch-level logs
    metrics_history_path = os.path.join(run_dir, "metrics_history.csv")
    if not os.path.exists(metrics_history_path):
        pd.DataFrame(columns=[
            "epoch","loss","BottomKAcc","PairwiseOrderAcc","Top1Acc","MeanMargin","AvgLogLik","RuleReproRateWeek",
            "monitor_name","monitor_val","best_monitor","improved"
        ]).to_csv(metrics_history_path, index=False)

    def _append_metrics_row(row: Dict[str, Any]) -> None:
        pd.DataFrame([row]).to_csv(metrics_history_path, mode="a", header=False, index=False)
        with open(os.path.join(run_dir, "last_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    # resume
    resume_from = str(cfg.get("training", {}).get("resume_from", "none")).lower()
    if resume_from in ["last", "best"]:
        ckpt_path = os.path.join(run_dir, f"model_{resume_from}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_monitor = float(ckpt.get("best_monitor", best_monitor))

    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["patience"])
    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0))
    accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))

    patience_counter = 0

    panels = dataset.panel
    if panel_indices is not None:
        panels = [dataset.panel[i] for i in panel_indices]

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        metrics_list: List[Dict[str, Any]] = []
        step_count = 0

        for week_data in panels:
            p_fan, s_total, alpha, id_static = model(week_data, all_feats, mc_dropout=False, dropout_p=0.0)

            l_base, used = compute_loss(week_data, p_fan, s_total, cfg)
            if used == 0:
                continue

            l_prior = get_prior_loss(model, cfg, week_data=week_data)

            loss = l_base + l_prior
            loss.backward()

            step_count += 1
            if step_count % accum_steps == 0:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())

            m = calculate_metrics(week_data, s_total.detach(), cfg)
            if m is not None:
                metrics_list.append(m)

        if step_count % accum_steps != 0:
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # epoch metrics
        if metrics_list:
            bottomk = float(np.mean([x["BottomKAcc"] for x in metrics_list if x.get("BottomKAcc") is not None]))
            pairwise = _safe_mean([x.get("PairwiseOrderAcc") for x in metrics_list])
            top1 = _safe_mean([x.get("Top1Acc") for x in metrics_list])
            mean_margin = float(np.mean([x["MeanMargin"] for x in metrics_list if x.get("MeanMargin") is not None]))
            avg_loglik = _safe_mean([x.get("AvgLogLik") for x in metrics_list])
            rule_repro = float(np.mean([x["RuleReproRateWeek"] for x in metrics_list if x.get("RuleReproRateWeek") is not None]))
        else:
            bottomk, pairwise, top1, mean_margin, avg_loglik, rule_repro = 0.0, None, None, 0.0, None, 0.0

        # choose monitor value
        monitor_map = {
            "train_BottomKAcc": bottomk,
            "train_PairwiseOrderAcc": pairwise if pairwise is not None else -1e18,
            "train_Top1Acc": top1 if top1 is not None else -1e18,
            "train_RuleReproRateWeek": rule_repro,
        }
        monitor_val = float(monitor_map.get(monitor_name, bottomk))

        if writer is not None:
            writer.add_scalar("Loss/train", total_loss, epoch)
            writer.add_scalar("BottomKAcc/train", bottomk, epoch)
            if pairwise is not None:
                writer.add_scalar("PairwiseOrderAcc/train", pairwise, epoch)
            if top1 is not None:
                writer.add_scalar("Top1Acc/train", top1, epoch)
            writer.add_scalar("MeanMargin/train", mean_margin, epoch)
            if avg_loglik is not None:
                writer.add_scalar("AvgLogLik/train", avg_loglik, epoch)
            writer.add_scalar("RuleReproRateWeek/train", rule_repro, epoch)
            writer.add_scalar("LR/lr", float(optimizer.param_groups[0]["lr"]), epoch)

        if scheduler is not None:
            scheduler.step(monitor_val)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_monitor": best_monitor,
            "monitor_name": monitor_name,
        }
        torch.save(ckpt, os.path.join(run_dir, "model_last.pt"))

        improved = monitor_val > best_monitor
        row = {
            "epoch": int(epoch),
            "loss": float(total_loss),
            "BottomKAcc": float(bottomk),
            "PairwiseOrderAcc": float(pairwise) if pairwise is not None else None,
            "Top1Acc": float(top1) if top1 is not None else None,
            "MeanMargin": float(mean_margin),
            "AvgLogLik": float(avg_loglik) if avg_loglik is not None else None,
            "RuleReproRateWeek": float(rule_repro),
            "monitor_name": str(monitor_name),
            "monitor_val": float(monitor_val),
            "best_monitor": float(best_monitor),
            "improved": bool(improved),
        }
        _append_metrics_row(row)
        if improved:
            best_monitor = monitor_val
            ckpt["best_monitor"] = best_monitor
            torch.save(ckpt, os.path.join(run_dir, "model_best.pt"))
            row_best = dict(row)
            row_best["best_monitor"] = float(best_monitor)
            with open(os.path.join(run_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(row_best, f, ensure_ascii=False, indent=2)
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:04d} | Loss {total_loss:.6f} | "
            f"BottomKAcc {bottomk:.4f} | "
            f"PairwiseOrderAcc {(pairwise if pairwise is not None else float('nan')):.4f} | "
            f"Top1Acc {(top1 if top1 is not None else float('nan')):.4f} | "
            f"MeanMargin {mean_margin:.4f} | "
            f"AvgLogLik {(avg_loglik if avg_loglik is not None else float('nan')):.4f} | "
            f"RuleRepro {rule_repro:.4f} | "
            f"monitor={monitor_name}:{monitor_val:.4f} | pat={patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            break

    # Export using the BEST checkpoint weights for consistency
    best_ckpt_path = os.path.join(run_dir, "model_best.pt")
    if os.path.exists(best_ckpt_path):
        ckpt_best = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_best["model"])
        model.eval()

    export_results(model, dataset, all_feats, fb, cfg, run_dir)

    try:
        dfp = pd.read_csv(os.path.join(run_dir, "pred_fan_shares_enriched.csv"))
        summary = {
            "best_monitor": float(best_monitor),
            "n_rows": int(len(dfp)),
            "alpha_mean": float(dfp["alpha"].mean()) if ("alpha" in dfp.columns and len(dfp)) else None,
        }
        with open(os.path.join(run_dir, "export_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return float(best_monitor)


def export_results(model: DWTSModel,
                   dataset: DWTSDataset,
                   all_feats: torch.Tensor,
                   fb: FeatureBuilder,
                   cfg: Dict[str, Any],
                   run_dir: str) -> None:
    """
    Per team-week export:
      - season, week, team_id
      - P_fan, S_total, alpha
      - p_elim = softmax(risk) where risk=-S_total
      - true_eliminated / true_withdrew / pred_eliminated / hit (Y/empty)
      - D_pf_minus_j, Z_pf_minus_j, Z_outlier_90/95
      - best_loser_worst_winner_margin

    Additional:
      - optional MC dropout CI columns for P_fan
      - team-season summary: Z_mean, Z_max_abs, frac_outlier_90/95, n_weeks
      - static params per team
    """
    device = all_feats.device
    model.eval()

    metrics_cfg = cfg.get("metrics", {})
    z90 = float(metrics_cfg.get("outlier_z_90", 1.645))
    z95 = float(metrics_cfg.get("outlier_z_95", 1.96))
    eps = float(cfg.get("features", {}).get("eps", 1e-6))

    inf_cfg = cfg.get("inference", {})
    mc_passes = int(inf_cfg.get("mc_passes", 1))
    mc_p = float(inf_cfg.get("mc_dropout_p", 0.0))
    save_mc_samples = bool(inf_cfg.get("save_mc_samples", False))
    mc_enabled = (mc_passes > 1 and mc_p > 0.0)

    rows = []
    sample_rows = []

    with torch.no_grad():
        for week_data in dataset.panel:
            # deterministic forward
            p_fan, s_total, alpha, _ = model(week_data, all_feats, mc_dropout=False, dropout_p=0.0)

            teams_idx = week_data["teams"]
            team_ids = [dataset.teams[i] for i in teams_idx]

            j_pct = week_data["j_pct"].to(device)
            D = (p_fan - j_pct)
            muD = float(D.mean().item())
            sdD = float(D.std(unbiased=False).item())
            Z = (D - muD) / (sdD + eps)

            true_elim_set = set(week_data.get("eliminated", []))
            true_wd_set = set(week_data.get("withdrew", []))

            # predicted elimination set: BottomK based on risk, excluding withdrew
            K = len(true_elim_set)
            pred_local_set = set()
            if K > 0:
                eff_local = [i for i, gid in enumerate(teams_idx) if gid not in true_wd_set]
                if eff_local:
                    risk_eff = (-s_total[torch.tensor(eff_local, device=device)])
                    order_eff = torch.argsort(risk_eff, descending=True)
                    topk_pos = order_eff[: min(K, len(eff_local))]
                    pred_local_set = set(int(eff_local[int(p.item())]) for p in topk_pos)

            risk = (-s_total)
            p_elim = torch.softmax(risk, dim=0)

            # margin computed on effective set (exclude withdrew)
            loser_local = [i for i, gid in enumerate(teams_idx) if gid in true_elim_set and gid not in true_wd_set]
            winner_local = [i for i, gid in enumerate(teams_idx) if gid not in true_elim_set and gid not in true_wd_set]
            if loser_local and winner_local:
                worst_winner = float(s_total[torch.tensor(winner_local, device=device)].min().item())
                best_loser = float(s_total[torch.tensor(loser_local, device=device)].max().item())
                margin = worst_winner - best_loser
            else:
                margin = 0.0

            # per-row export
            for local_i, tid in enumerate(team_ids):
                global_team_idx = teams_idx[local_i]

                true_elim_flag = "Y" if global_team_idx in true_elim_set else ""
                true_wd_flag = "Y" if global_team_idx in true_wd_set else ""

                pred_elim_flag = "Y" if local_i in pred_local_set else ""
                hit_flag = "Y" if (pred_elim_flag == "Y" and true_elim_flag == "Y") else ""

                z_val = float(Z[local_i].item())
                z90_flag = "Y" if abs(z_val) > z90 else ""
                z95_flag = "Y" if abs(z_val) > z95 else ""

                rows.append({
                    "season": int(week_data["season"]),
                    "week": int(week_data["week"]),
                    "team_id": tid,

                    "P_fan": float(p_fan[local_i].item()),
                    "S_total": float(s_total[local_i].item()),
                    "alpha": float(alpha.item()),

                    "p_elim": float(p_elim[local_i].item()),

                    "true_eliminated": true_elim_flag,
                    "true_withdrew": true_wd_flag,
                    "pred_eliminated": pred_elim_flag,
                    "hit": hit_flag,

                    "D_pf_minus_j": float(D[local_i].item()),
                    "Z_pf_minus_j": z_val,
                    "Z_outlier_90": z90_flag,
                    "Z_outlier_95": z95_flag,

                    "best_loser_worst_winner_margin": float(margin),
                })

            # MC dropout for CI & optional long-format samples
            if mc_enabled:
                pf_samples = []
                for s in range(mc_passes):
                    pf_s, st_s, a_s, _ = model(week_data, all_feats, mc_dropout=True, dropout_p=mc_p)
                    pf_samples.append(pf_s.unsqueeze(0))
                    if save_mc_samples:
                        for local_i, tid in enumerate(team_ids):
                            sample_rows.append({
                                "season": int(week_data["season"]),
                                "week": int(week_data["week"]),
                                "team_id": tid,
                                "sample_id": int(s),
                                "P_fan": float(pf_s[local_i].item()),
                            })

                pf_samples = torch.cat(pf_samples, dim=0)  # [S, n_team]
                lo = torch.quantile(pf_samples, 0.025, dim=0)
                hi = torch.quantile(pf_samples, 0.975, dim=0)

                n_team = len(team_ids)
                for k in range(n_team):
                    rows[-n_team + k]["P_fan_ci_low_95"] = float(lo[k].item())
                    rows[-n_team + k]["P_fan_ci_high_95"] = float(hi[k].item())

    out_csv = os.path.join(run_dir, "pred_fan_shares_enriched.csv")
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out_csv, index=False)

    if save_mc_samples and sample_rows:
        out_samp = os.path.join(run_dir, "pfan_samples_long.csv")
        pd.DataFrame(sample_rows).to_csv(out_samp, index=False)

    # team-season summary
    if len(df_rows) > 0:
        df_rows["absZ"] = df_rows["Z_pf_minus_j"].abs()
        df_rows["is_out_90"] = df_rows["absZ"] > z90
        df_rows["is_out_95"] = df_rows["absZ"] > z95

        g = df_rows.groupby(["season", "team_id"], as_index=False)
        zsum = g.agg(
            Z_mean=("Z_pf_minus_j", "mean"),
            Z_max_abs=("absZ", "max"),
            frac_outlier_90=("is_out_90", "mean"),
            frac_outlier_95=("is_out_95", "mean"),
            n_weeks=("week", "count"),
        )
        zsum.to_csv(os.path.join(run_dir, "team_season_summary.csv"), index=False)

    _export_team_params(model, dataset, all_feats, fb, run_dir)


def _export_team_params(model: DWTSModel,
                        dataset: DWTSDataset,
                        all_feats: torch.Tensor,
                        fb: FeatureBuilder,
                        run_dir: str) -> None:
    device = all_feats.device
    model.eval()

    phi_vec = model.phi.weight.detach().cpu().view(-1).tolist()

    rows = []
    with torch.no_grad():
        for tid in dataset.teams:
            row0 = dataset.df[dataset.df["team_id"] == tid].iloc[0]
            celeb = str(row0["celebrity_name"])
            partner = str(row0["ballroom_partner"])

            c_idx = torch.tensor([dataset.c_to_idx[celeb]], device=device, dtype=torch.long)
            p_idx = torch.tensor([dataset.p_to_idx[partner]], device=device, dtype=torch.long)
            t_idx = torch.tensor([dataset.team_to_idx[tid]], device=device, dtype=torch.long)

            theta = float(model.theta(c_idx).squeeze().item())
            u = float(model.u(p_idx).squeeze().item())
            x_i = all_feats[t_idx].detach().cpu().view(-1).tolist()

            rows.append({
                "team_id": tid,
                "celebrity_name": celeb,
                "ballroom_partner": partner,
                "theta": theta,
                "u_partner": u,
                "x_i": json.dumps(x_i),
                "phi": json.dumps(phi_vec),
            })

    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "team_static_params.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    train(config_path=args.config)
