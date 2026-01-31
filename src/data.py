import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple

class DWTSDataset:
    def __init__(self, csv_path: str, config: Dict):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.eps = float(config["features"]["eps"])
        self.prepare_data()

    def prepare_data(self):
        self.df["team_id"] = self.df["celebrity_name"] + "__" + self.df["ballroom_partner"]
        self.teams = self.df["team_id"].unique().tolist()
        self.partners = self.df["ballroom_partner"].unique().tolist()
        self.celebrities = self.df["celebrity_name"].unique().tolist()
        
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.p_to_idx = {p: i for i, p in enumerate(self.partners)}
        self.c_to_idx = {c: i for i, c in enumerate(self.celebrities)}

        self.panel = []
        seasons = sorted(self.df["season"].unique())
        for s in seasons:
            sdf = self.df[self.df["season"] == s].copy()
            week_scores = {}
            max_w = 0
            for w in range(1, 12):
                cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
                scores = sdf[cols].replace("N/A", np.nan).astype(float).sum(axis=1, skipna=True)
                if scores.sum() > 0:
                    week_scores[w] = scores
                    max_w = w

            prev_zj = {}
            prev_active = {}
            for w in range(1, max_w + 1):
                if w not in week_scores: continue
                scores = week_scores[w]
                active_mask = scores > 0
                a_sdf = sdf[active_mask].copy()
                a_scores = scores[active_mask].values
                
                j_pct = a_scores / (a_scores.sum() + self.eps)
                mean_s, std_s = a_scores.mean(), a_scores.std()
                z_j = (a_scores - mean_s) / (std_s + self.eps)
                r_j = pd.Series(a_scores).rank(ascending=False, method="min").values
                
                dzj = np.zeros_like(z_j)
                curr_tids = a_sdf["team_id"].tolist()
                for i, tid in enumerate(curr_tids):
                    if prev_active.get(tid, False):
                        dzj[i] = z_j[i] - prev_zj.get(tid, 0.0)
                    else:
                        dzj[i] = 0.0
                    prev_zj[tid] = z_j[i]
                    prev_active[tid] = True

                for tid in sdf["team_id"].tolist():
                    if tid not in set(curr_tids):
                        prev_active[tid] = False

                eliminated = []
                if w < max_w:
                    next_scores = week_scores.get(w+1, pd.Series(0, index=sdf.index))
                    for idx, row in a_sdf.iterrows():
                        if next_scores.loc[idx] <= 0:
                            eliminated.append(self.team_to_idx[row["team_id"]])

                self.panel.append({
                    "season": s, "week": w, "max_week": max_w,
                    "teams": [self.team_to_idx[tid] for tid in a_sdf["team_id"]],
                    "partners": [self.p_to_idx[p] for p in a_sdf["ballroom_partner"]],
                    "celebrities": [self.c_to_idx[c] for c in a_sdf["celebrity_name"]],
                    "j_pct": torch.tensor(j_pct, dtype=torch.float32),
                    "zj": torch.tensor(z_j, dtype=torch.float32),
                    "dzj": torch.tensor(dzj, dtype=torch.float32),
                    "rj": torch.tensor(r_j, dtype=torch.float32),
                    "eliminated": eliminated,
                    "j_total": torch.tensor(a_scores, dtype=torch.float32)
                })

    def __len__(self):
        return len(self.panel)