from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data import DWTSDataset


class FeatureBuilder:
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.config = config
        self.eps = float(config["features"]["eps"])
        # Supported: log_zscore (default), zscore, none
        self.age_transform = str(config["features"].get("age_transform", "log_zscore")).lower()

        self.industries = self._get_unique(df, "celebrity_industry")
        self.countries = self._get_unique(df, "celebrity_homecountry/region")
        self.states = self._get_unique(df, "celebrity_homestate")

        # Age stats for z-score (after optional log-transform)
        ages = df["celebrity_age_during_season"].astype(float).to_numpy()
        if self.age_transform.startswith("log"):
            ages_t = np.log(ages + self.eps)
        else:
            ages_t = ages
        self.age_mean = float(np.mean(ages_t))
        self.age_std = float(np.std(ages_t))

        self.ind_map = {v: i for i, v in enumerate(self.industries)}
        self.cnt_map = {v: i for i, v in enumerate(self.countries)}
        self.st_map = {v: i for i, v in enumerate(self.states)}
        self.dim = 1 + len(self.industries) + len(self.countries)
        if bool(config["features"].get("use_homestate", True)):
            self.dim += len(self.states)

    def _get_unique(self, df: pd.DataFrame, col: str):
        # Keep every category value (including Unknown) so one-hot has full coverage.
        return sorted(df[col].apply(self._norm_cat).unique().tolist())

    def _norm_cat(self, val):
        if val is None:
            return "Unknown"
        if isinstance(val, float) and np.isnan(val):
            return "Unknown"
        s = str(val).strip()
        return s if s else "Unknown"

    def _age_feature(self, age_value: float) -> float:
        if self.age_transform in ("none", "raw"):
            return float(age_value)

        if self.age_transform in ("zscore", "raw_zscore"):
            base = float(age_value)
            return (base - self.age_mean) / (self.age_std + self.eps)

        if self.age_transform in ("log_zscore", "logzscore", "log+zscore"):
            base = float(np.log(age_value + self.eps))
            return (base - self.age_mean) / (self.age_std + self.eps)

        # Fallback: be strict to avoid silently violating your modeling spec.
        raise ValueError(f"Unknown features.age_transform: {self.age_transform}")

    def get_features(self, row: pd.Series) -> torch.Tensor:
        age_val = float(row["celebrity_age_during_season"])
        age_feat = self._age_feature(age_val)

        ind_v = torch.zeros(len(self.industries))
        ind_v[self.ind_map[self._norm_cat(row["celebrity_industry"])]] = 1.0

        cnt_v = torch.zeros(len(self.countries))
        cnt_v[self.cnt_map[self._norm_cat(row.get("celebrity_homecountry/region"))]] = 1.0

        feats = [torch.tensor([age_feat], dtype=torch.float32), ind_v, cnt_v]

        if bool(self.config["features"].get("use_homestate", True)):
            st_v = torch.zeros(len(self.states))
            st_v[self.st_map[self._norm_cat(row.get("celebrity_homestate"))]] = 1.0
            feats.append(st_v)

        return torch.cat(feats).float()


def build_all_features(dataset: "DWTSDataset", feature_builder: FeatureBuilder) -> torch.Tensor:
    return torch.stack(
        [feature_builder.get_features(dataset.df[dataset.df["team_id"] == tid].iloc[0]) for tid in dataset.teams]
    )
