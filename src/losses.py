# src/losses.py
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def get_prior_loss(model, config: Dict) -> torch.Tensor:
    # theta/u Gaussian priors -> squared L2 (mean for scale stability)
    lambda_theta = float(config["model"]["lambda_theta"])
    lambda_u = float(config["model"]["lambda_u"])

    theta_w = model.theta.weight.squeeze(-1)  # [num_celebs]
    u_w = model.u.weight.squeeze(-1)          # [num_partners]

    l_theta = lambda_theta * (theta_w ** 2).mean()
    l_u = lambda_u * (u_w ** 2).mean()

    # phi regularization (supports none, l1, l2, elasticnet)
    phi_reg = str(config["model"]["phi_reg"]).lower()
    lam_l1 = float(config["model"].get("lambda_phi_l1", 0.0))
    lam_l2 = float(config["model"].get("lambda_phi_l2", 0.0))
    w = model.phi.weight  # shape [1, feat_dim]

    if phi_reg == "none":
        l_phi = 0.0 * w.sum()  # keep it a tensor on-graph
    elif phi_reg == "l1":
        l_phi = lam_l1 * torch.norm(w, p=1)
    elif phi_reg == "l2":
        l_phi = lam_l2 * (torch.norm(w, p=2) ** 2)
    elif phi_reg == "elasticnet":
        l_phi = lam_l1 * torch.norm(w, p=1) + lam_l2 * (torch.norm(w, p=2) ** 2)
    else:
        raise ValueError(f"Unknown phi_reg='{phi_reg}'. Use: none, l1, l2, elasticnet")

    # beta prior around beta_center
    lambda_beta = float(config["model"]["lambda_beta"])
    beta_center = float(config["model"]["beta_center"])
    l_beta = lambda_beta * torch.sum((model.beta - beta_center) ** 2)

    return l_theta + l_u + l_phi + l_beta


def compute_loss(
    week_data: Dict,
    p_fan: torch.Tensor,
    s_total: torch.Tensor,
    config: Dict
) -> Tuple[torch.Tensor, int]:
    """
    Base pairwise hinge is normalized by the number of (winner, loser) pairs.
    Twist L_A is normalized by the number of k candidates (|A|-2).
    Returns:
      loss: scalar tensor
      base_count: 1 if this week has elimination (base loss active), else 0
    """
    elim = week_data["eliminated"]
    if not elim:
        return torch.tensor(0.0, device=s_total.device), 0

    n = len(week_data["teams"])
    all_pos = torch.arange(n, device=s_total.device)

    elim_set = set(elim)  # elim contains global team indices
    loser_mask = torch.tensor([tid in elim_set for tid in week_data["teams"]], device=s_total.device)

    winners = all_pos[~loser_mask]
    losers = all_pos[loser_mask]

    # -------------------------
    # Base pairwise hinge (mean over |W|*|L|)
    # -------------------------
    xi = float(config["loss"]["xi_margin"])
    if winners.numel() == 0 or losers.numel() == 0:
        l_base = torch.tensor(0.0, device=s_total.device)
        base_count = 0
    else:
        s_diff = s_total[winners].unsqueeze(1) - s_total[losers].unsqueeze(0)  # [|W|, |L|]
        l_base = F.relu(xi - s_diff).mean()
        base_count = 1

    # -------------------------
    # Twist loss (Season >= 28, exactly 1 elimination, enough teams, not last week)
    # -------------------------
    l_twist = torch.tensor(0.0, device=s_total.device)

    season = int(week_data["season"])
    week = int(week_data["week"])
    max_week = int(week_data["max_week"])

    lamA = float(config["loss"]["lambdaA"])
    lamB = float(config["loss"]["lambdaB"])

    if (
        season >= 28
        and losers.numel() == 1
        and n > 2
        and week < max_week
        and (lamA > 0.0 or lamB > 0.0)
    ):
        e_pos = int(losers.item())  # position inside this week's arrays
        s_proxy_pos = int(winners[torch.argmin(s_total[winners])].item())

        mask = torch.ones(n, dtype=torch.bool, device=s_total.device)
        mask[e_pos] = False
        mask[s_proxy_pos] = False
        ks = all_pos[mask]  # all other positions

        # L_A: encourage {e, s_proxy} to be bottom-2 (hinge, mean over ks)
        if lamA > 0.0 and ks.numel() > 0:
            xi_t = float(config["loss"]["xi_twist"])
            l_a = F.relu(xi_t - (s_total[ks] - s_total[e_pos])).mean() + \
                  F.relu(xi_t - (s_total[ks] - s_total[s_proxy_pos])).mean()
        else:
            l_a = 0.0 * s_total.sum()

        # L_B: judges pick which to eliminate among bottom two (hinge proxy)
        if lamB > 0.0:
            j_total = week_data["j_total"].to(s_total.device)
            xi_tb = float(config["loss"]["xi_tb"])
            l_b = F.relu(xi_tb - (j_total[s_proxy_pos] - j_total[e_pos]))
        else:
            l_b = 0.0 * s_total.sum()

        l_twist = lamA * l_a + lamB * l_b

    return l_base + l_twist, base_count
