"""Optuna tuning entrypoint.

This version reads ALL parameter ranges from a YAML file (default: configs/tune_cpu.yaml),
so you can adjust ranges without touching code.

Usage:
  python -m src.tune --tune_config configs/tune_cpu.yaml
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import yaml

try:
    import optuna
except Exception as e:  # pragma: no cover
    optuna = None  # type: ignore

from src.train import train
from src.utils import load_config, setup_logger, save_resolved_config


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _flatten_overrides(overrides: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in overrides.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_overrides(v, kk))
        else:
            out[kk] = v
    return out


def _suggest_from_space(trial: "optuna.Trial", space: Dict[str, Any]) -> Any:
    t = str(space.get("type", "float")).lower()
    if t == "categorical":
        return trial.suggest_categorical(space["name"], space["choices"])
    if t == "int":
        low = int(space["low"])
        high = int(space["high"])
        step = int(space.get("step", 1))
        log = bool(space.get("log", False))
        return trial.suggest_int(space["name"], low, high, step=step, log=log)
    # float
    low = float(space["low"])
    high = float(space["high"])
    log = bool(space.get("log", False))
    return trial.suggest_float(space["name"], low, high, log=log)


def run_tune(tune_config_path: str) -> None:
    if optuna is None:
        raise RuntimeError("optuna is not available in this environment.")

    with open(tune_config_path, "r", encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)

    seed = int(tcfg.get("seed", 42))
    study_name = str(tcfg.get("study_name", "dwts_tune"))
    direction = str(tcfg.get("direction", "maximize"))
    n_trials = int(tcfg.get("n_trials", 50))
    n_jobs = int(tcfg.get("n_jobs", 1))
    storage = str(tcfg.get("storage", "sqlite:///outputs/optuna.db"))
    output_root = str(tcfg.get("output_root", "outputs"))
    run_prefix = str(tcfg.get("run_prefix", "tune"))
    base_config_path = str(tcfg.get("base_config_path", "configs/default.yaml"))

    fixed_overrides = tcfg.get("fixed_overrides", {}) or {}
    fixed_flat = _flatten_overrides(fixed_overrides)

    search_space = tcfg.get("search_space", {}) or {}

    tune_root = os.path.join(output_root, f"{study_name}__tune")
    os.makedirs(tune_root, exist_ok=True)
    setup_logger(tune_root)

    base_cfg = load_config(base_config_path, None, overrides=None)
    save_resolved_config(base_cfg, os.path.join(tune_root, "base_config_resolved.yaml"))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner_startup = int(tcfg.get("pruner_startup", 10))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: "optuna.Trial") -> float:
        overrides: Dict[str, Any] = {}
        overrides.update(fixed_flat)

        for dotted_key, spec in search_space.items():
            spec = dict(spec)
            spec["name"] = dotted_key
            val = _suggest_from_space(trial, spec)
            overrides[dotted_key] = val

        overrides["output_root"] = tune_root
        overrides["run_name"] = f"{run_prefix}_trial_{trial.number:04d}"
        overrides["training.resume_from"] = "none"

        val = float(train(config_path=base_config_path, overrides=overrides))
        return val

    with open(os.path.join(tune_root, "tune_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_name": study_name,
                "direction": direction,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "storage": storage,
                "tune_config": tune_config_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_config", type=str, default="configs/tune_cpu.yaml")
    args = parser.parse_args()
    run_tune(args.tune_config)
