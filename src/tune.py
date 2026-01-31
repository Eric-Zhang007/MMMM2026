"""
Optuna hyperparameter tuning for the DWTS model.

Design goals:
1) Objective maximizes the same monitor used in training (default: train_ElimTop1Acc).
2) Each trial runs a full end-to-end training and returns the best in-trial monitor value.
3) Uses multi-process parallelism on CPU (sqlite storage) to avoid PyTorch+threads conflicts.
4) Keeps ONE CPU core free for other tasks by default.
5) Avoids parallel GPU trials unless explicitly allowed.

Usage:
  python -m src.tune --config configs/default.yaml
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import optuna
import yaml

from src.train import train
from src.utils import deep_update, load_config, save_resolved_config, setup_logger

DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")

_TUNE_CTX: Dict[str, Any] = {}


def _cpu_minus_one() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _is_gpu_device_name(device_name: str) -> bool:
    d = (device_name or "").lower()
    return ("cuda" in d) or ("xpu" in d) or ("mps" in d)


def _get_tune_root(output_root: str, base_run_name: str) -> str:
    return os.path.join(str(output_root), f"{base_run_name}__tune")


def _suggest_overrides(trial: optuna.Trial) -> Dict[str, Any]:
    # Core sensitive hyperparams (aligned with your model):
    overrides: Dict[str, Any] = {}

    # alpha variance tradeoff
    overrides["model.k_variance_ratio"] = trial.suggest_float("k_variance_ratio", 0.05, 20.0, log=True)

    # rank soft-temperature
    overrides["model.tau_rank"] = trial.suggest_float("tau_rank", 0.05, 1.5, log=True)

    # priors / regularization
    overrides["model.lambda_theta"] = trial.suggest_float("lambda_theta", 1e-4, 5e-2, log=True)
    overrides["model.lambda_u"] = trial.suggest_float("lambda_u", 1e-4, 5e-2, log=True)
    overrides["model.lambda_beta"] = trial.suggest_float("lambda_beta", 1e-4, 5e-2, log=True)

    # elasticnet on phi (keep both when phi_reg=elasticnet)
    overrides["model.lambda_phi_l1"] = trial.suggest_float("lambda_phi_l1", 1e-6, 5e-3, log=True)
    overrides["model.lambda_phi_l2"] = trial.suggest_float("lambda_phi_l2", 1e-6, 5e-3, log=True)

    # hinge margins
    overrides["loss.xi_margin"] = trial.suggest_float("xi_margin", 0.01, 0.3, log=True)
    overrides["loss.xi_twist"] = trial.suggest_float("xi_twist", 0.01, 0.3, log=True)
    overrides["loss.xi_tb"] = trial.suggest_float("xi_tb", 0.01, 0.3, log=True)

    # twist weights
    overrides["loss.lambdaA"] = trial.suggest_float("lambdaA", 1e-3, 1.0, log=True)
    overrides["loss.lambdaB"] = trial.suggest_float("lambdaB", 1e-3, 1.0, log=True)

    # optimizer knobs
    overrides["training.lr"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    overrides["training.weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.05)

    return overrides


def objective(trial: optuna.Trial) -> float:
    ctx = _TUNE_CTX
    config_path = ctx["config_path"]
    tune_root = ctx["tune_root"]
    base_overrides = ctx["base_overrides"]

    # Trial directory: outputs/<run_name>__tune/trial_0001/...
    trial_run_name = f"trial_{trial.number:04d}"

    trial_overrides = dict(base_overrides)
    trial_overrides.update(_suggest_overrides(trial))

    # Put each trial under the tune_root for tidy structure
    trial_overrides["output_root"] = tune_root
    trial_overrides["run_name"] = trial_run_name

    # Speed up tuning by using fewer epochs/patience than the final training (if configured)
    trial_overrides["training.epochs"] = ctx["tune_epochs"]
    trial_overrides["training.patience"] = ctx["tune_patience"]

    # Never resume inside tuning trials
    trial_overrides["training.resume_from"] = "none"

    # Actually run training; return best monitor value (train() already handles early stop)
    try:
        val = float(train(config_path=config_path, overrides=trial_overrides))
    except Exception as e:
        # Mark failure and re-raise so Optuna records it properly
        raise e

    return val


def _create_study(study_name: str, storage_url: str, seed: int) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


def _split_trials(n_trials: int, n_workers: int) -> Tuple[int, ...]:
    base = n_trials // n_workers
    rem = n_trials % n_workers
    counts = []
    for i in range(n_workers):
        counts.append(base + (1 if i < rem else 0))
    return tuple(counts)


def _run_worker(
    worker_id: int,
    n_trials: int,
    study_name: str,
    storage_url: str,
    config_path: str,
    tune_root: str,
    base_overrides: Dict[str, Any],
    tune_epochs: int,
    tune_patience: int,
    seed: int,
):
    # Each worker is a separate process: enforce single-thread PyTorch to avoid oversubscription.
    os.environ["DWTS_TORCH_THREADS"] = "1"

    # Set global context for objective()
    global _TUNE_CTX
    _TUNE_CTX = {
        "config_path": config_path,
        "tune_root": tune_root,
        "base_overrides": base_overrides,
        "tune_epochs": tune_epochs,
        "tune_patience": tune_patience,
    }

    study = _create_study(study_name, storage_url, seed + worker_id * 1000)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)


def run_tune(config_path: Optional[str], n_trials: Optional[int], n_jobs: Optional[int], seed: int):
    base_cfg = load_config(DEFAULT_CONFIG_PATH, config_path, overrides=None)

    base_run_name = str(base_cfg["run_name"])
    output_root = str(base_cfg["output_root"])
    tune_root = _get_tune_root(output_root, base_run_name)
    os.makedirs(tune_root, exist_ok=True)

    setup_logger(tune_root)
    save_resolved_config(base_cfg, os.path.join(tune_root, "base_config_resolved.yaml"))

    # Resolve n_trials / n_jobs (workers)
    cfg_tune = base_cfg.get("tune", {}) or {}
    if n_trials is None:
        n_trials = int(cfg_tune.get("n_trials", 50))

    # Default: keep 1 core free
    if n_jobs is None:
        cfg_n_jobs = cfg_tune.get("n_jobs", None)
        if cfg_n_jobs is None or (isinstance(cfg_n_jobs, int) and cfg_n_jobs <= 0) or str(cfg_n_jobs).lower() == "auto":
            n_jobs = _cpu_minus_one()
        else:
            n_jobs = max(1, min(int(cfg_n_jobs), _cpu_minus_one()))

    # GPU safety: avoid multiple parallel GPU trials unless explicitly allowed
    device_name = str(base_cfg.get("device", "auto"))
    allow_parallel_gpu = bool(cfg_tune.get("allow_parallel_gpu", False))
    if _is_gpu_device_name(device_name) and n_jobs > 1 and not allow_parallel_gpu:
        n_jobs = 1

    # Tuning epochs/patience (defaults: smaller than final)
    tune_epochs = int(cfg_tune.get("epochs", min(int(base_cfg["training"]["epochs"]), 80)))
    tune_patience = int(cfg_tune.get("patience", min(int(base_cfg["training"]["patience"]), 10)))

    # Optuna study storage
    study_name = str(cfg_tune.get("study_name", f"{base_run_name}_study"))
    storage_url = f"sqlite:///{os.path.join(tune_root, 'optuna.db')}"

    # Base overrides that should hold across trials
    base_overrides: Dict[str, Any] = {}
    # Keep your modeling constants; beta_center is already in config.
    base_overrides["training.monitor"] = str(base_cfg["training"].get("monitor", "train_ElimTop1Acc"))

    # Save tune meta
    with open(os.path.join(tune_root, "tune_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_name": study_name,
                "storage_url": storage_url,
                "n_trials": n_trials,
                "n_workers": n_jobs,
                "tune_epochs": tune_epochs,
                "tune_patience": tune_patience,
                "seed": seed,
                "device": device_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Run workers
    trial_counts = _split_trials(n_trials, n_jobs)

    if n_jobs == 1:
        _run_worker(
            worker_id=0,
            n_trials=trial_counts[0],
            study_name=study_name,
            storage_url=storage_url,
            config_path=config_path,
            tune_root=tune_root,
            base_overrides=base_overrides,
            tune_epochs=tune_epochs,
            tune_patience=tune_patience,
            seed=seed,
        )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        procs = []
        for wid, cnt in enumerate(trial_counts):
            if cnt <= 0:
                continue
            p = ctx.Process(
                target=_run_worker,
                args=(
                    wid,
                    cnt,
                    study_name,
                    storage_url,
                    config_path,
                    tune_root,
                    base_overrides,
                    tune_epochs,
                    tune_patience,
                    seed,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    # Summarize best results
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        print("[TUNE ERROR] No successful trials. All trials failed.")
        print("[TUNE ERROR] Please check the first failure stack trace above (often data parsing / config types).")
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        print(f"[TUNE ERROR] failed_trials={len(failed)} total_trials={len(study.trials)}")
        return

    best_params = dict(study.best_params)
    best_value = float(study.best_value)


    with open(os.path.join(tune_root, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"best_value": best_value, "best_params": best_params}, f, ensure_ascii=False, indent=2)

    # Build best overrides yaml (dot-keys) and a fully resolved config yaml
    best_overrides: Dict[str, Any] = {}
    # Map trial param names back to dot-keys
    mapping = {
        "k_variance_ratio": "model.k_variance_ratio",
        "tau_rank": "model.tau_rank",
        "lambda_theta": "model.lambda_theta",
        "lambda_u": "model.lambda_u",
        "lambda_beta": "model.lambda_beta",
        "lambda_phi_l1": "model.lambda_phi_l1",
        "lambda_phi_l2": "model.lambda_phi_l2",
        "xi_margin": "loss.xi_margin",
        "xi_twist": "loss.xi_twist",
        "xi_tb": "loss.xi_tb",
        "lambdaA": "loss.lambdaA",
        "lambdaB": "loss.lambdaB",
        "lr": "training.lr",
        "weight_decay": "training.weight_decay",
    }
    for k, v in best_params.items():
        if k in mapping:
            best_overrides[mapping[k]] = v

    # Also enforce the tune-root structure for a “best run” if you want to re-train later
    # (Optional; you can comment out if you prefer running best config elsewhere.)
    with open(os.path.join(tune_root, "best_overrides.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_overrides, f, sort_keys=False, allow_unicode=True)

    # Produce fully merged best config for convenience
    best_cfg = load_config(DEFAULT_CONFIG_PATH, config_path, overrides=None)
    # Apply dot-key overrides into nested dict
    nested_best = {}
    for dk, dv in best_overrides.items():
        keys = dk.split(".")
        d = nested_best
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = dv
    best_cfg = deep_update(best_cfg, nested_best)

    save_resolved_config(best_cfg, os.path.join(tune_root, "best_config_resolved.yaml"))

    print(f"[TUNE DONE] best_value={best_value:.6f}")
    print(f"[TUNE DONE] best_params={best_params}")
    print(f"[TUNE DONE] tune_root={tune_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of worker processes (CPU). Default: cpu-1.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tune(config_path=args.config, n_trials=args.n_trials, n_jobs=args.n_jobs, seed=args.seed)


if __name__ == "__main__":
    main()
