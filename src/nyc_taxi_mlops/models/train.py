import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


# -----------------------------
# Configuration (can later be CLI / YAML)
# -----------------------------
CONFIG = {
    "model_type": "linear_regression",
    "random_seed": 42,
    "n_samples": 1000,
    "n_features": 5,
    "noise": 0.5,
    "test_size": 0.2,
}


# -----------------------------
# Data generation
# -----------------------------
def make_fake_data(cfg):
    rng = np.random.default_rng(cfg["random_seed"])

    X = rng.normal(size=(cfg["n_samples"], cfg["n_features"]))
    true_coef = rng.normal(size=cfg["n_features"])
    y = X @ true_coef + rng.normal(scale=cfg["noise"], size=cfg["n_samples"])

    return X, y


# -----------------------------
# Training logic
# -----------------------------
def train(cfg):
    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc_taxi_mlops_experiment")

    X, y = make_fake_data(cfg)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
    )

    with mlflow.start_run(run_name="train_script_test") as run:
        # log config as params
        mlflow.log_params(cfg)

        if cfg["model_type"] == "linear_regression":
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {cfg['model_type']}")

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Run {run.info.run_id} finished | RMSE={rmse:.4f}")


# -----------------------------
# Script entrypoint
# -----------------------------
if __name__ == "__main__":
    train(CONFIG)
