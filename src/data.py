import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Any, Tuple


def _to_float(x: Any, default: float = 0.0) -> float:
    """Robust float conversion for YAML values that might be strings."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


class DWTSDataset:
    """
    Build a season-week panel from the wide CSV.

    Symbols:
      - A: active teams this week (as rows in week_data["teams"])
      - L: true eliminated teams this week (global indices in week_data["eliminated"])
      - X: withdrew teams this week (global indices in week_data["withdrew"])
      - W = A \ (L ∪ X): survivors among effective set used for supervision/metrics

    Key design choices:
    - Elimination inferred by: active this week, inactive next week (same season).
    - Withdrew handled separately:
        * is_withdrew determined by row-level `results` contains "Withdrew" (case-insensitive).
        * withdrew week inferred by first transition to inactive (scores become 0).
        * withdrew teams are NOT included in `eliminated` (so do not affect loss/metrics).
    - Multiple eliminations supported (|L| can be > 1).
    - Random-walk observation ids:
        * Each active (season, team, week) gets a unique obs_id.
        * prev_obs_id points to the previous active week for the same (season, team), or -1 if none.
    """

    def __init__(self, csv_path: str, config: Dict):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.eps = _to_float(config.get("features", {}).get("eps", 1e-6), default=1e-6)
        self.panel: List[Dict[str, Any]] = []
        self.num_obs: int = 0
        self.prepare_data()

    def prepare_data(self) -> None:
        df = self.df
        df["team_id"] = df["celebrity_name"].astype(str) + "__" + df["ballroom_partner"].astype(str)

        self.teams = df["team_id"].unique().tolist()
        self.partners = df["ballroom_partner"].unique().tolist()
        self.celebrities = df["celebrity_name"].unique().tolist()

        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.p_to_idx = {p: i for i, p in enumerate(self.partners)}
        self.c_to_idx = {c: i for i, c in enumerate(self.celebrities)}

        seasons = sorted(df["season"].unique())

        # Pre-detect withdrew at row level
        if "results" in df.columns:
            df["_results_str"] = df["results"].fillna("").astype(str)
            df["_is_withdrew"] = df["_results_str"].str.contains("withdrew", case=False, regex=False)
        else:
            df["_is_withdrew"] = False

        obs_counter = 0
        last_obs: Dict[Tuple[int, int], int] = {}  # (season, team_global) -> obs_id

        for s in seasons:
            sdf = df[df["season"] == s].copy()

            # Precompute weekly total scores per row
            week_scores: Dict[int, pd.Series] = {}
            max_w = 0
            for w in range(1, 12):
                cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
                if not set(cols).issubset(sdf.columns):
                    continue
                scores = (
                    sdf[cols]
                    .replace("N/A", np.nan)
                    .astype(float)
                    .sum(axis=1, skipna=True)
                )
                if scores.sum() > 0:
                    week_scores[w] = scores
                    max_w = w

            if max_w == 0:
                continue

            prev_zj: Dict[str, float] = {}
            withdrew_row = sdf["_is_withdrew"].to_dict()  # index->bool

            for w in range(1, max_w + 1):
                if w not in week_scores:
                    continue

                scores = week_scores[w]
                active_mask = scores > 0
                if not active_mask.any():
                    continue

                a_sdf = sdf.loc[active_mask].copy()
                a_scores = scores.loc[active_mask].values.astype(float)

                j_pct = a_scores / (a_scores.sum() + self.eps)

                mean_s = float(a_scores.mean())
                std_s = float(a_scores.std())
                z_j = (a_scores - mean_s) / (std_s + self.eps)

                r_j = pd.Series(a_scores).rank(ascending=False, method="min").values.astype(float)

                dzj = np.zeros_like(z_j, dtype=float)
                for i, (idx, row) in enumerate(a_sdf.iterrows()):
                    tid = str(row["team_id"])
                    if tid in prev_zj:
                        dzj[i] = z_j[i] - prev_zj[tid]
                    prev_zj[tid] = float(z_j[i])

                eliminated: List[int] = []
                withdrew: List[int] = []
                if w < max_w:
                    next_scores = week_scores.get(w + 1, pd.Series(0.0, index=sdf.index))
                    for idx, row in a_sdf.iterrows():
                        if float(next_scores.loc[idx]) <= 0.0:
                            tid = str(row["team_id"])
                            t_global = self.team_to_idx[tid]
                            if bool(withdrew_row.get(idx, False)):
                                withdrew.append(t_global)
                            else:
                                eliminated.append(t_global)

                # obs ids aligned with the active list order
                teams_global = [self.team_to_idx[str(tid)] for tid in a_sdf["team_id"].tolist()]
                obs_ids: List[int] = []
                prev_obs_ids: List[int] = []
                for t_global in teams_global:
                    key = (int(s), int(t_global))
                    prev = last_obs.get(key, -1)
                    obs_ids.append(obs_counter)
                    prev_obs_ids.append(prev)
                    last_obs[key] = obs_counter
                    obs_counter += 1

                week_data: Dict[str, Any] = {
                    "season": int(s),
                    "week": int(w),
                    "max_week": int(max_w),
                    "teams": teams_global,
                    "partners": [self.p_to_idx[str(p)] for p in a_sdf["ballroom_partner"].tolist()],
                    "celebrities": [self.c_to_idx[str(c)] for c in a_sdf["celebrity_name"].tolist()],
                    "obs_ids": obs_ids,
                    "prev_obs_ids": prev_obs_ids,
                    "j_pct": torch.tensor(j_pct, dtype=torch.float32),
                    "zj": torch.tensor(z_j, dtype=torch.float32),
                    "dzj": torch.tensor(dzj, dtype=torch.float32),
                    "rj": torch.tensor(r_j, dtype=torch.float32),
                    "j_total": torch.tensor(a_scores, dtype=torch.float32),
                    "eliminated": eliminated,      # excludes withdrew
                    "withdrew": withdrew,          # separate
                }
                self.panel.append(week_data)

        self.num_obs = obs_counter

    def __len__(self) -> int:
        return len(self.panel)
