import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DWTSModel(nn.Module):
    """
    Identity (static):
        id_static = theta_c + u_p + phi^T x_i

    Random-walk shock (fan-side only):
        r_{i,t} ~ RW over (season, team, week) observations

    Identity (dynamic for fan utility):
        id_dyn = id_static + r_{i,t}

    Alpha uses ONLY static identity dispersion:
        sigma_fan^2 = Var(id_static)
        sigma_judge^2 = Var(j_pct)

        alpha = sigma_judge^2 / (sigma_judge^2 + k * sigma_fan^2 + eps)

    Utility:
        eta = (1-alpha)*id_dyn + alpha*lambda_perf*perf_in
        p_fan = softmax(eta)

    Optional MC dropout for inference:
      forward(..., mc_dropout=True, dropout_p=...) applies dropout to eta even in eval mode.
    """

    def __init__(self, num_celebs: int, num_partners: int, feat_dim: int, num_obs: int, config: Dict):
        super().__init__()
        self.config = config

        self.theta = nn.Embedding(num_celebs, 1)
        self.u = nn.Embedding(num_partners, 1)
        self.phi = nn.Linear(feat_dim, 1, bias=False)

        self.r = nn.Embedding(int(num_obs), 1)
        nn.init.zeros_(self.r.weight)

        beta_center = float(config.get("model", {}).get("beta_center", 1.0))
        self.beta = nn.Parameter(torch.tensor([beta_center, beta_center], dtype=torch.float32))

        nn.init.normal_(self.theta.weight, std=0.01)
        nn.init.normal_(self.u.weight, std=0.01)
        nn.init.normal_(self.phi.weight, std=0.01)

    def forward(self, week_data: Dict, all_feats: torch.Tensor, *, mc_dropout: bool = False, dropout_p: float = 0.0):
        device = all_feats.device
        eps = float(self.config.get("features", {}).get("eps", 1e-6))

        c_idx = torch.tensor(week_data["celebrities"], device=device, dtype=torch.long)
        p_idx = torch.tensor(week_data["partners"], device=device, dtype=torch.long)
        t_idx = torch.tensor(week_data["teams"], device=device, dtype=torch.long)
        obs_idx = torch.tensor(week_data.get("obs_ids", []), device=device, dtype=torch.long)

        theta_base = self.theta(c_idx).squeeze(-1)
        u_base = self.u(p_idx).squeeze(-1)
        phi_x = self.phi(all_feats[t_idx]).squeeze(-1)

        id_static = theta_base + u_base + phi_x

        if obs_idx.numel() == id_static.numel():
            r_t = self.r(obs_idx).squeeze(-1)
        else:
            r_t = torch.zeros_like(id_static)

        id_dyn = id_static + r_t

        zj = week_data["zj"].to(device)
        dzj = week_data["dzj"].to(device)
        perf_in = self.beta[0] * zj + self.beta[1] * dzj

        sigma_fan2 = torch.var(id_static, unbiased=False)
        sigma_judge2 = torch.var(week_data["j_pct"].to(device), unbiased=False)

        k = float(self.config.get("model", {}).get("k_variance_ratio", 1.0))
        alpha = sigma_judge2 / (sigma_judge2 + k * sigma_fan2 + eps)

        lambda_perf = float(self.config.get("model", {}).get("lambda_perf", 1.0))
        eta = (1.0 - alpha) * id_dyn + alpha * lambda_perf * perf_in

        p = float(dropout_p or 0.0)
        if p > 0.0:
            eta = F.dropout(eta, p=p, training=(self.training or mc_dropout))

        p_fan = F.softmax(eta, dim=0)

        season = int(week_data["season"])
        if 3 <= season <= 27:
            s_total = week_data["j_pct"].to(device) + p_fan
        else:
            rj = week_data["rj"].to(device)
            tau = float(self.config.get("model", {}).get("tau_rank", 0.7))
            diff = (eta.unsqueeze(0) - eta.unsqueeze(1)) / tau
            soft_rank = 1.0 + torch.sum(torch.sigmoid(diff), dim=1) - 0.5
            s_total = -(rj + soft_rank)

        return p_fan, s_total, alpha, id_static
