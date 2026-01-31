import os
import torch
import yaml
import logging
from typing import Any, Dict

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_config(default_path: str, config_path: str = None, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)
    if config_path and config_path != default_path:
        with open(config_path, "r") as f:
            cfg = deep_update(cfg, yaml.safe_load(f))
    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
    return cfg

def save_resolved_config(cfg: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

def setup_logger(output_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )

def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device_name)

def write_formulas(path: str, config: Dict):
    eps = config["features"]["eps"]
    k = config["model"]["k_variance_ratio"]
    formulas = rf"""
1. IdentityScore_i = \theta_{{celebrity(i)}} + u_{{partner(i)}} + \phi^T x_i
2. PerformanceInput_{{i,t}} = \beta_1 zJ_{{i,t}} + \beta_2 \Delta zJ_{{i,t}}
3. \sigma_{{fan}}^2[s,t] = Var_{{i \in A_{{s,t}}}}(IdentityScore_i)
4. \sigma_{{judge}}^2[s,t] = Var_{{i \in A_{{s,t}}}}(J\_pct_{{i,t}})
5. \alpha_{{s,t}} = \frac{{\sigma_{{judge}}^2}}{{\sigma_{{judge}}^2 + {k} \cdot \sigma_{{fan}}^2 + {eps}}}
6. \eta_{{i,t}} = (1 - \alpha_{{s,t}}) \cdot IdentityScore_i + \alpha_{{s,t}} \cdot \lambda_{{perf}} \cdot PerformanceInput_{{i,t}}
7. P\_fan_{{i,t}} = \text{{softmax}}_{{i \in A_{{s,t}}}}(\eta_{{i,t}})
8. S\_total_{{i,t}} (Percent) = J\_pct_{{i,t}} + P\_fan_{{i,t}}
9. S\_total_{{i,t}} (Rank) = -(rJ_{{i,t}} + 1 + \sum_{{k \neq i}} \sigma(\frac{{\eta_{{k,t}} - \eta_{{i,t}}}}{{\tau_{{rank}}}}))
10. L\_base = \frac{{1}}{{|W||L|}} \sum_{{i \in W, j \in L}} \text{{ReLU}}(\xi_m - (S\_total_i - S\_total_j))
11. Twist Loss (S28+): I_{{twist}} \cdot (\lambda_A L_A + \lambda_B L_B)
"""
    with open(path, "w") as f:
        f.write(formulas)