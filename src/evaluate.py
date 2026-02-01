import math
import torch
from typing import Dict, Optional


def calculate_metrics(week_data: Dict, s_total: torch.Tensor, config: Optional[Dict] = None):
    """
    Metrics under the unified symbol system.

    Let A be active teams this week (week_data['teams'] local list).
    Let X be withdrew teams this week (week_data['withdrew'] global ids).
    Use effective set A_eff = A \ X for metrics that compare eliminated vs survivors.

    Define risk_i so that larger means more dangerous:
        risk = -S_total

    Primary metrics:
      - BottomKAcc: K=|L|, BottomK(risk) vs L (supports multi-elimination weeks)
      - Top1Acc: only when K==1 (otherwise None)
      - PairwiseOrderAcc: average 1{risk_l > risk_w} over l in L, w in W

    Keep:
      - MeanMargin (worst winner - best loser, in S_total space, on A_eff)

    Extra (useful but not "main" in the paper):
      - AvgLogLik / Perplexity: pairwise BT likelihood on winner-loser pairs (A_eff)
      - RuleReproRateWeek: 1 iff all eliminated are inside bottom-K (K=|L|) on A_eff
    """
    elim_global = set(week_data.get("eliminated", []))
    if not elim_global:
        return None

    withdrew_global = set(week_data.get("withdrew", []))
    teams_global = week_data["teams"]  # local order, global ids

    # Effective local indices (exclude withdrew)
    eff_local = [i for i, tid in enumerate(teams_global) if tid not in withdrew_global]
    if not eff_local:
        return None

    loser_local = [i for i in eff_local if teams_global[i] in elim_global]
    K = len(loser_local)
    if K == 0:
        return None

    winner_local = [i for i in eff_local if teams_global[i] not in elim_global]
    if not winner_local:
        return None

    # risk ordering within effective set
    risk = -s_total
    eff_t = torch.tensor(eff_local, device=s_total.device, dtype=torch.long)
    risk_eff = risk[eff_t]
    order_eff = torch.argsort(risk_eff, descending=True)  # most dangerous first, indices in eff-local list
    bottomk_eff_pos = set(int(x.item()) for x in order_eff[: min(K, len(eff_local))])
    bottomk_local = set(eff_local[pos] for pos in bottomk_eff_pos)

    bottomk_hit = sum(1.0 for x in loser_local if x in bottomk_local) / float(K)
    rule_repro = 1.0 if all(x in bottomk_local for x in loser_local) else 0.0

    top1_acc = None
    if K == 1:
        top1_local = int(eff_local[int(order_eff[0].item())])
        top1_acc = 1.0 if top1_local in loser_local else 0.0

    # Pairwise order acc
    wl = torch.tensor(winner_local, device=s_total.device, dtype=torch.long)
    ll = torch.tensor(loser_local, device=s_total.device, dtype=torch.long)
    pairwise_acc = None
    if wl.numel() > 0 and ll.numel() > 0:
        # compare every (l,w): risk_l > risk_w ?
        comp = (risk[ll].unsqueeze(1) > risk[wl].unsqueeze(0)).float()
        pairwise_acc = float(comp.mean().item())

    # Mean margin: worst winner - best loser in S_total space
    mean_margin = 0.0
    if winner_local and loser_local:
        worst_winner = float(s_total[torch.tensor(winner_local, device=s_total.device)].min().item())
        best_loser = float(s_total[torch.tensor(loser_local, device=s_total.device)].max().item())
        mean_margin = worst_winner - best_loser

    # Avg loglik / perplexity
    tau_ll = 1.0
    if config is not None:
        tau_ll = float(config.get("metrics", {}).get("tau_ll", 1.0))

    avg_loglik = None
    perplexity = None
    if winner_local and loser_local:
        s_w = s_total[torch.tensor(winner_local, device=s_total.device)]
        s_l = s_total[torch.tensor(loser_local, device=s_total.device)]
        diff = (s_w.unsqueeze(1) - s_l.unsqueeze(0)) / tau_ll
        logp = torch.log(torch.sigmoid(diff).clamp_min(1e-12))
        avg_loglik = float(logp.mean().item())
        perplexity = float(math.exp(-avg_loglik))

    return {
        "BottomKAcc": float(bottomk_hit),
        "Top1Acc": top1_acc,
        "PairwiseOrderAcc": pairwise_acc,
        "MeanMargin": float(mean_margin),
        "RuleReproRateWeek": float(rule_repro),
        "AvgLogLik": avg_loglik,
        "Perplexity": perplexity,
        "K": int(K),
    }
