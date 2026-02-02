import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def _safe_normalize(p: torch.Tensor, eps: float) -> torch.Tensor:
    p2 = p + eps
    return p2 / (p2.sum() + eps)


class DWTSModel(nn.Module):
    """
    Identity (static):
        id_static = theta_c + u_p + phi^T x_i

    Random-walk shock (fan-side only):
        r_{i,t} ~ RW over (season, team, week) observations

    Identity (dynamic for fan utility):
        id_dyn = id_static + r_{i,t}

    Alpha uses ONLY static identity dispersion (default: variance).
    You can switch alpha_mode to 'entropy' or 'hhi' (concentration-based),
    and optionally force alpha at inference time.

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

    def _compute_alpha(self, id_static: torch.Tensor, j_pct: torch.Tensor, eps: float) -> torch.Tensor:
        mcfg = self.config.get("model", {})
        mode = str(mcfg.get("alpha_mode", "variance")).lower()
        k = float(mcfg.get("k_variance_ratio", 1.0))

        # Optional: inference-only alpha override (used for alpha=0/1 stress tests)
        inf_cfg = self.config.get("inference", {})
        force_alpha = inf_cfg.get("force_alpha", None)
        if force_alpha is not None:
            fa = float(force_alpha)
            fa = max(0.0, min(1.0, fa))
            return id_static.new_tensor(fa)

        if mode == "variance":
            sigma_fan2 = torch.var(id_static, unbiased=False)
            sigma_judge2 = torch.var(j_pct, unbiased=False)
            return sigma_judge2 / (sigma_judge2 + k * sigma_fan2 + eps)

        # concentration-based modes: compare fan vs judge "peakedness"
        tau_rank = float(mcfg.get("tau_rank", 0.7))
        tau_alpha = float(mcfg.get("alpha_tau", tau_rank))
        alpha_eps = float(mcfg.get("alpha_eps", eps))

        # fan distribution from static identity (NOT using r_t)
        p_fan_static = F.softmax(id_static / max(tau_alpha, 1e-6), dim=0)

        # judge distribution: j_pct is already a distribution, but normalize safely
        p_j = _safe_normalize(j_pct, alpha_eps)

        if mode == "hhi":
            c_fan = torch.sum(p_fan_static.pow(2))
            c_j = torch.sum(p_j.pow(2))
            return c_j / (c_j + k * c_fan + alpha_eps)

        if mode == "entropy":
            h_fan = -torch.sum(p_fan_static * torch.log(p_fan_static + alpha_eps))
            h_j = -torch.sum(p_j * torch.log(p_j + alpha_eps))
            neff_fan = torch.exp(h_fan)
            neff_j = torch.exp(h_j)
            c_fan = 1.0 / (neff_fan + alpha_eps)
            c_j = 1.0 / (neff_j + alpha_eps)
            return c_j / (c_j + k * c_fan + alpha_eps)

        sigma_fan2 = torch.var(id_static, unbiased=False)
        sigma_judge2 = torch.var(j_pct, unbiased=False)
        return sigma_judge2 / (sigma_judge2 + k * sigma_fan2 + eps)

    def forward(self, week_data: Dict, all_feats: torch.Tensor, *, mc_dropout: bool = False, dropout_p: float = 0.0):
        device = all_feats.device
        eps = float(self.config.get("features", {}).get("eps", 1e-6))

        c_idx = torch.tensor(week_data["celebrities"], device=device, dtype=torch.long)
        p_idx = torch.tensor(week_data["partners"], device=device, dtype=torch.long)
        t_idx = torch.tensor(week_data["teams"], device=device, dtype=torch.long)
        obs_idx = torch.tensor(week_data.get("obs_ids", []), device=device, dtype=torch.long)

        theta_base = self.theta(c_idx).squeeze(-1)
        u_base = self.u(p_idx).squeeze(-1)
        mcfg = self.config.get("model", {})
        use_phi = bool(mcfg.get("use_phi", True))
        use_r = bool(mcfg.get("use_r", True))
        perf_use_dzj = bool(mcfg.get("perf_use_dzj", True))

        if use_phi:
            phi_x = self.phi(all_feats[t_idx]).squeeze(-1)
        else:
            phi_x = torch.zeros_like(theta_base)

        id_static = theta_base + u_base + phi_x

        if use_r and obs_idx.numel() == id_static.numel():
            r_t = self.r(obs_idx).squeeze(-1)
        else:
            r_t = torch.zeros_like(id_static)

        id_dyn = id_static + r_t

        zj = week_data["zj"].to(device)
        dzj = week_data["dzj"].to(device)
        if not perf_use_dzj:
            dzj = torch.zeros_like(dzj)

        # Performance decomposition
        perf_zj = self.beta[0] * zj
        perf_dzj = self.beta[1] * dzj

        dzj_side = str(mcfg.get("dzj_side", "fan")).lower()
        lambda_dzj_fan = float(mcfg.get("lambda_dzj_fan", 1.0))

        j_pct = week_data["j_pct"].to(device)
        alpha = self._compute_alpha(id_static=id_static, j_pct=j_pct, eps=eps)

        lambda_perf = float(mcfg.get("lambda_perf", 1.0))
        if dzj_side == "judge":
            perf_in = perf_zj + perf_dzj
            eta = (1.0 - alpha) * id_dyn + alpha * lambda_perf * perf_in
        else:
            # default: move ΔzJ to fan-side utility only (does not affect alpha dispersion)
            eta = (1.0 - alpha) * (id_dyn + lambda_dzj_fan * perf_dzj) + alpha * lambda_perf * perf_zj

        p = float(dropout_p or 0.0)
        if p > 0.0:
            eta = F.dropout(eta, p=p, training=(self.training or mc_dropout))

        p_fan = F.softmax(eta, dim=0)

        season = int(week_data["season"])
        if 3 <= season <= 27:
            s_total = j_pct + p_fan
        else:
            rj = week_data["rj"].to(device)
            tau = float(mcfg.get("tau_rank", 0.7))
            diff = (eta.unsqueeze(0) - eta.unsqueeze(1)) / max(tau, 1e-6)
            soft_rank = 1.0 + torch.sum(torch.sigmoid(diff), dim=1) - 0.5
            s_total = -(rj + soft_rank)

        return p_fan, s_total, alpha, id_static
