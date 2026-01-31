import optuna
from src.train import train

def objective(trial):
    overrides = {
        "model.lambda_theta_u": trial.suggest_float("lambda_theta_u", 1e-5, 1e-2, log=True),
        "model.k_variance_ratio": trial.suggest_float("k_variance_ratio", 0.1, 10.0),
        "training.lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    }
    # This is a simplified call; in practice, return the best metric from a validation split
    return 0.0 

def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)