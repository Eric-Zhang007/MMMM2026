import torch

def calculate_metrics(week_data, s_total):
    elim = week_data["eliminated"]
    if not elim: return None
    order = torch.argsort(s_total, descending=False)
    loser_indices = [i for i, tid in enumerate(week_data["teams"]) if tid in elim]
    top1_hit = 1.0 if order[0].item() in loser_indices else 0.0
    bottom2_hit = sum([1.0 for idx in order[:2] if idx.item() in loser_indices]) / len(loser_indices)
    
    margin = 0.0
    if len(week_data["teams"]) > len(loser_indices):
        winner_min = s_total[[i for i in range(len(s_total)) if i not in loser_indices]].min()
        loser_max = s_total[loser_indices].max()
        margin = (winner_min - loser_max).item()
        
    return {"ElimTop1Acc": top1_hit, "Bottom2Acc": bottom2_hit, "Margin": margin}