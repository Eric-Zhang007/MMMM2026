import os
import yaml
import json
import time
import math
import random
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce_known_numerics(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # 防止 YAML/CLI 覆盖导致 eps 变成字符串，从而出现 numpy dtype('<U4') 之类错误
    def _to_float_if_possible(x):
        if isinstance(x, str):
            try:
                return float(x)
            except Exception:
                return x
        return x

    # 常见数值字段
    if "features" in cfg and "eps" in cfg["features"]:
        cfg["features"]["eps"] = _to_float_if_possible(cfg["features"]["eps"])
    if "training" in cfg:
        for key in ["lr", "weight_decay", "grad_clip_norm"]:
            if key in cfg["training"]:
                cfg["training"][key] = _to_float_if_possible(cfg["training"][key])
    if "model" in cfg:
        for key in ["k_variance_ratio", "lambda_perf", "tau_rank",
                    "lambda_beta", "beta_center",
                    "lambda_phi_l1", "lambda_phi_l2",
                    "lambda_theta", "lambda_u", "lambda_rw"]:
            if key in cfg["model"]:
                cfg["model"][key] = _to_float_if_possible(cfg["model"][key])
    if "metrics" in cfg and "pairwise_tau" in cfg["metrics"]:
        cfg["metrics"]["pairwise_tau"] = _to_float_if_possible(cfg["metrics"]["pairwise_tau"])
    return cfg


def load_config(default_path: str, config_path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if config_path and config_path != default_path:
        with open(config_path, "r", encoding="utf-8") as f:
            custom = yaml.safe_load(f)
        cfg = deep_update(cfg, custom)

    if overrides:
        # overrides: {"a.b.c": value}
        for k, v in overrides.items():
            keys = k.split(".")
            d = cfg
            for kk in keys[:-1]:
                d = d.setdefault(kk, {})
            d[keys[-1]] = v

    cfg = _coerce_known_numerics(cfg)
    return cfg


def save_resolved_config(cfg: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def setup_logger(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8"),
            logging.StreamHandler()
        ],
    )


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_formulas(path: str, cfg: Dict[str, Any]) -> None:
    eps = float(cfg["features"]["eps"])
    k = float(cfg["model"]["k_variance_ratio"])
    lam_r = float(cfg["model"].get("lambda_r", 0.0))
    lam_rw = float(cfg["model"].get("lambda_rw", 0.0))
    phi_reg = str(cfg["model"].get("phi_reg", "none")).lower()
    lam_phi_l1 = float(cfg["model"].get("lambda_phi_l1", 0.0))
    lam_phi_l2 = float(cfg["model"].get("lambda_phi_l2", 0.0))
    dzj_side = str(cfg["model"].get("dzj_side", "fan")).lower()
    lam_dzj_fan = float(cfg["model"].get("lambda_dzj_fan", 1.0))

    formulas = rf"""
1. IdentityScore_i = \theta_{{celebrity(i)}} + u_{{partner(i)}} + \phi^T x_i
2. JudgePerf_{{i,t}} = \beta_1 zJ_{{i,t}}
3. DeltaJudge_{{i,t}} = \beta_2 \Delta zJ_{{i,t}}
4. \sigma_{{fan}}^2[s,t] = Var_{{i \in A_{{s,t}}}}(IdentityScore_i)
5. \sigma_{{judge}}^2[s,t] = Var_{{i \in A_{{s,t}}}}(J\_pct_{{i,t}})
6. \alpha_{{s,t}} = \frac{{\sigma_{{judge}}^2}}{{\sigma_{{judge}}^2 + {k}\cdot\sigma_{{fan}}^2 + {eps}}}

7. Random-walk shock (fan-side): IdentityDyn_{{i,t}} = IdentityScore_i + r_{{i,t}}

8. Utility:
   if dzj\_side = judge:
     \eta_{{i,t}} = (1-\alpha)\cdot IdentityDyn_{{i,t}} + \alpha\cdot \lambda_{{perf}}\cdot(\beta_1 zJ_{{i,t}} + \beta_2 \Delta zJ_{{i,t}})
   if dzj\_side = fan:
     \eta_{{i,t}} = (1-\alpha)\cdot\big(IdentityDyn_{{i,t}} + {lam_dzj_fan}\cdot \beta_2 \Delta zJ_{{i,t}}\big)
                  + \alpha\cdot \lambda_{{perf}}\cdot(\beta_1 zJ_{{i,t}})

9. P\_fan_{{i,t}} = softmax(\eta_{{i,t}})

10. S\_total (Percent seasons) = J\_pct + P\_fan

11. Priors and total loss:
    L\_r  = {lam_r} * mean(r_{{i,t}}^2)
    L\_rw = {lam_rw} * mean((r_{{i,t}} - r_{{i,t-1}})^2)
    L\_\phi follows {phi_reg} with (\lambda_{{\phi,1}}, \lambda_{{\phi,2}}) = ({lam_phi_l1}, {lam_phi_l2})
    L\_base = mean_{{w \in W, \ell \in L}} ReLU(\xi - (S_w - S_\ell))
    L = L\_base + L\_twist + L\_\theta + L\_u + L\_\phi + L\_\beta + L\_r + L\_rw
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(formulas.strip() + "
")
