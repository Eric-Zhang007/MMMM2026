import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class DWTSModel(nn.Module):
    def __init__(self, num_celebs: int, num_partners: int, feat_dim: int, config: Dict):
        super().__init__()
        self.config = config
        self.theta = nn.Embedding(num_celebs, 1)
        self.u = nn.Embedding(num_partners, 1)
        self.phi = nn.Linear(feat_dim, 1, bias=False)
        self.beta = nn.Parameter(torch.tensor([config["model"]["beta_center"], config["model"]["beta_center"]]))
        nn.init.normal_(self.theta.weight, std=0.01)
        nn.init.normal_(self.u.weight, std=0.01)
        nn.init.normal_(self.phi.weight, std=0.01)

    def forward(self, week_data: Dict, all_feats: torch.Tensor):
        device = all_feats.device
        eps = self.config["features"]["eps"]
        c_idx = torch.tensor(week_data["celebrities"], device=device)
        p_idx = torch.tensor(week_data["partners"], device=device)
        t_idx = torch.tensor(week_data["teams"], device=device)
        
        id_scores = self.theta(c_idx).squeeze(-1) + self.u(p_idx).squeeze(-1) + self.phi(all_feats[t_idx]).squeeze(-1)
        perf_in = self.beta[0] * week_data["zj"].to(device) + self.beta[1] * week_data["dzj"].to(device)
        
        sigma_fan2 = torch.var(id_scores, unbiased=False)
        sigma_judge2 = torch.var(week_data["j_pct"].to(device), unbiased=False)
        k = self.config["model"]["k_variance_ratio"]
        alpha = sigma_judge2 / (sigma_judge2 + k * sigma_fan2 + eps)
        
        eta = (1 - alpha) * id_scores + alpha * self.config["model"]["lambda_perf"] * perf_in
        p_fan = F.softmax(eta, dim=0)
        
        if 3 <= week_data["season"] <= 27:
            s_total = week_data["j_pct"].to(device) + p_fan
        else:
            diff = (eta.unsqueeze(0) - eta.unsqueeze(1)) / self.config["model"]["tau_rank"]
            soft_rank = 1.0 + torch.sum(torch.sigmoid(diff), dim=1) - 0.5
            s_total = -(week_data["rj"].to(device) + soft_rank)
            
        return p_fan, s_total, alpha, id_scores