import torch
import torch.nn.functional as F
from typing import Dict, Optional


def get_prior_loss(model, config: Dict, week_data: Optional[Dict] = None) -> torch.Tensor:
    """
    Explicit priors / regularizers.

    - L2 on theta/u: lambda_theta * mean(theta^2) + lambda_u * mean(u^2)
    - phi regularization: l1/l2/elasticnet with lambda_phi_l1/l2
    - beta center prior: lambda_beta * sum((beta - beta_center)^2)

    Random-walk shock r_{i,t}:
    - lambda_r  * mean(r_t^2) on the obs_ids present in this week (amplitude)
    - lambda_rw * mean((r_t - r_{t-1})^2) using prev_obs_ids (smoothness)
      If prev_obs_id == -1, that term is skipped.
    """
    mcfg = config.get("model", {})

    device = model.theta.weight.device

    lambda_theta = float(mcfg.get("lambda_theta", 0.01))
    lambda_u = float(mcfg.get("lambda_u", 0.01))

    l_theta = lambda_theta * (model.theta.weight.pow(2).mean())
    l_u = lambda_u * (model.u.weight.pow(2).mean())

    phi_reg = str(mcfg.get("phi_reg", "none")).lower()
    lam_phi_l1 = float(mcfg.get("lambda_phi_l1", 0.0))
    lam_phi_l2 = float(mcfg.get("lambda_phi_l2", 0.0))

    w = model.phi.weight
    if phi_reg == "l1":
        l_phi = lam_phi_l1 * torch.norm(w, 1)
    elif phi_reg == "l2":
        l_phi = lam_phi_l2 * torch.norm(w, 2).pow(2)
    elif phi_reg == "elasticnet":
        l_phi = lam_phi_l1 * torch.norm(w, 1) + lam_phi_l2 * torch.norm(w, 2).pow(2)
    else:
        l_phi = torch.tensor(0.0, device=w.device)

    lambda_beta = float(mcfg.get("lambda_beta", 0.0))
    beta_center = float(mcfg.get("beta_center", 1.0))
    l_beta = lambda_beta * torch.sum((model.beta - beta_center).pow(2))

    # r regularization (week-local)
    l_r = torch.tensor(0.0, device=device)
    l_rw = torch.tensor(0.0, device=device)

    lam_r = float(mcfg.get("lambda_r", 0.0))
    lam_rw = float(mcfg.get("lambda_rw", 0.0))

    if week_data is not None and (lam_r > 0.0 or lam_rw > 0.0):
        obs_ids = week_data.get("obs_ids", [])
        prev_obs_ids = week_data.get("prev_obs_ids", [])
        if obs_ids and len(obs_ids) == len(prev_obs_ids):
            obs = torch.tensor(obs_ids, device=device, dtype=torch.long)
            r_t = model.r(obs).squeeze(-1)
            if lam_r > 0.0:
                l_r = lam_r * r_t.pow(2).mean()

            if lam_rw > 0.0:
                prev = torch.tensor(prev_obs_ids, device=device, dtype=torch.long)
                mask = prev >= 0
                if mask.any():
                    r_prev = model.r(prev[mask]).squeeze(-1)
                    l_rw = lam_rw * (r_t[mask] - r_prev).pow(2).mean()

    return l_theta + l_u + l_phi + l_beta + l_r + l_rw


def compute_loss(week_data: Dict, p_fan: torch.Tensor, s_total: torch.Tensor, config: Dict):
    """
    Pairwise hinge loss on winner-loser pairs.

    Important:
    - true eliminated come from week_data["eliminated"] (global team idx list)
    - withdrew teams are excluded from the effective set (A_eff = A \ withdrew) for supervision

    Returns:
      (loss_tensor, used_flag)
    """
    elim = set(week_data.get("eliminated", []))
    if not elim:
        return torch.tensor(0.0, device=s_total.device), 0

    withdrew = set(week_data.get("withdrew", []))
    teams_global = week_data["teams"]  # global team ids in local order

    # Effective local indices: exclude withdrew
    eff_local = [i for i, tid in enumerate(teams_global) if tid not in withdrew]
    if len(eff_local) == 0:
        return torch.tensor(0.0, device=s_total.device), 0

    loser_local = [i for i in eff_local if teams_global[i] in elim]
    winner_local = [i for i in eff_local if teams_global[i] not in elim]

    if len(loser_local) == 0 or len(winner_local) == 0:
        return torch.tensor(0.0, device=s_total.device), 0

    wi = torch.tensor(winner_local, device=s_total.device, dtype=torch.long)
    li = torch.tensor(loser_local, device=s_total.device, dtype=torch.long)

    s_diff = s_total[wi].unsqueeze(1) - s_total[li].unsqueeze(0)
    xi_m = float(config.get("loss", {}).get("xi_margin", 0.05))
    l_base = F.relu(xi_m - s_diff).mean()

    # Optional twist constraint (kept as-is, but applied on effective set only)
    l_twist = torch.tensor(0.0, device=s_total.device)
    loss_cfg = config.get("loss", {})
    twist_mode = str(loss_cfg.get("twist_mode", "hinge")).lower()

    season = int(week_data.get("season", 0))
    if season >= 28 and twist_mode == "hinge":
        # Only meaningful for single-elimination weeks (after excluding withdrew)
        if len(loser_local) == 1 and len(eff_local) > 2 and int(week_data["week"]) < int(week_data["max_week"]):
            e_idx = int(loser_local[0])
            # proxy survivor is the worst among survivors
            s_proxy_idx = int(min(winner_local, key=lambda j: float(s_total[j].item())))

            xi_t = float(loss_cfg.get("xi_twist", 0.05))
            xi_tb = float(loss_cfg.get("xi_tb", 0.05))
            lamA = float(loss_cfg.get("lambdaA", 0.2))
            lamB = float(loss_cfg.get("lambdaB", 0.2))

            eff_t = torch.tensor(eff_local, device=s_total.device, dtype=torch.long)
            for kk in eff_local:
                if kk != e_idx and kk != s_proxy_idx:
                    l_twist = l_twist + F.relu(xi_t - (s_total[kk] - s_total[e_idx]))                                       + F.relu(xi_t - (s_total[kk] - s_total[s_proxy_idx]))
            l_twist = l_twist / float(len(eff_local) - 2)

            j_total = week_data["j_total"].to(s_total.device)
            l_b = F.relu(xi_tb - (j_total[s_proxy_idx] - j_total[e_idx]))
            l_twist = lamA * l_twist + lamB * l_b

    return l_base + l_twist, 1
