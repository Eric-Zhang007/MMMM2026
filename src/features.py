from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data import DWTSDataset


class FeatureBuilder:
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.config = config
        self.eps = config["features"]["eps"]
        
        self.industries = self._get_unique(df, "celebrity_industry")
        self.countries = self._get_unique(df, "celebrity_homecountry/region")
        self.states = self._get_unique(df, "celebrity_homestate")
        
        ages = df["celebrity_age_during_season"].astype(float)
        log_ages = np.log(ages + self.eps)
        self.age_mean = log_ages.mean()
        self.age_std = log_ages.std()
        
        self.ind_map = {v: i for i, v in enumerate(self.industries)}
        self.cnt_map = {v: i for i, v in enumerate(self.countries)}
        self.st_map = {v: i for i, v in enumerate(self.states)}
        self.dim = 1 + len(self.industries) + len(self.countries)
        if config["features"]["use_homestate"]: self.dim += len(self.states)

    def _get_unique(self, df, col):
        return sorted(df[col].apply(self._norm_cat).unique().tolist())

    def _norm_cat(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)): return "Unknown"
        return str(val)

    def get_features(self, row: pd.Series) -> torch.Tensor:
        age = (np.log(float(row["celebrity_age_during_season"]) + self.eps) - self.age_mean) / (self.age_std + self.eps)
        ind_v = torch.zeros(len(self.industries))
        ind_v[self.ind_map[self._norm_cat(row["celebrity_industry"])]] = 1.0
        cnt_v = torch.zeros(len(self.countries))
        cnt_v[self.cnt_map[self._norm_cat(row.get("celebrity_homecountry/region"))]] = 1.0
        feats = [torch.tensor([age]), ind_v, cnt_v]
        if self.config["features"]["use_homestate"]:
            st_v = torch.zeros(len(self.states))
            st_v[self.st_map[self._norm_cat(row.get("celebrity_homestate"))]] = 1.0
            feats.append(st_v)
        return torch.cat(feats).float()
    
def build_all_features(dataset: DWTSDataset, feature_builder: FeatureBuilder) -> torch.Tensor:
    return torch.stack(
        [feature_builder.get_features(dataset.df[dataset.df["team_id"] == tid].iloc[0]) for tid in dataset.teams]
    )