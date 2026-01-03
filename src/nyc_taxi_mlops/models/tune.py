from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from nyc_taxi_mlops.features.build_features import make_features


# -------------------------
# Paths & constants
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"

EXPERIMENT_NAME = "nyc_taxi_mlops_experiment"
COMPARISON_GROUP = "xgb_tuning_v2"
SPLIT_STRATEGY = "standard_random_split_v1"  # tag to mark how splits were created


# -------------------------
# Hyperparameter search space
# -------------------------
def sample_xgb_params(rng: random.Random) -> dict:
    return {
        "max_depth": rng.choice([3, 4, 5, 6]),
        "learning_rate": rng.choice([0.03, 0.05, 0.1]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "min_child_weight": rng.choice([1, 3, 5, 10]),
        "reg_lambda": rng.choice([1.0, 5.0, 10.0]),
        "reg_alpha": rng.choice([0.0, 0.1, 0.5]),
    }


# -------------------------
# XGBoost tuning (train/val loaded from split files)
# -------------------------
def tune_xgboost(
    *,
    n_trials: int,
    random_seed: int,
    early_stopping_rounds: int,
):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    rng = random.Random(random_seed)

    df_train = pd.read_parquet(TRAIN_PATH)
    df_val = pd.read_parquet(VAL_PATH)

    X_train, y_train, feature_cols = make_features(df_train)
    X_val, y_val, feature_cols_val = make_features(df_val)

    if feature_cols_val != feature_cols:
        raise ValueError("Feature columns mismatch between train and val splits.")

    best_rmse = float("inf")

    # Parent run = tuning session
    with mlflow.start_run(run_name="xgb_tuning_standard_split"):
        mlflow.set_tag("comparison_group", COMPARISON_GROUP)
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("variant", "tuning")
        mlflow.set_tag("split_strategy", SPLIT_STRATEGY)

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        mlflow.log_param("train_rows", len(df_train))
        mlflow.log_param("val_rows", len(df_val))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("train_path", str(TRAIN_PATH))
        mlflow.log_param("val_path", str(VAL_PATH))

        for i in range(n_trials):
            params = sample_xgb_params(rng)

            model = XGBRegressor(
                n_estimators=5000,  # big cap; early stopping finds best iteration
                objective="reg:squarederror",
                random_state=random_seed,
                n_jobs=-1,
                early_stopping_rounds=early_stopping_rounds,  # (your version needs this here)
                **params,
            )

            # Child run = one trial
            with mlflow.start_run(run_name=f"xgb_trial_{i:02d}", nested=True):
                mlflow.set_tag("variant", "xgb_trial")
                mlflow.set_tag("split_strategy", SPLIT_STRATEGY)
                mlflow.log_params(params)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                preds = model.predict(X_val)
                rmse = root_mean_squared_error(y_val, preds)
                mae = mean_absolute_error(y_val, preds)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                if rmse < best_rmse:
                    best_rmse = rmse
                    mlflow.set_tag("best_so_far", "true")

        print(f"Best RMSE observed during tuning: {best_rmse:.3f}")


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--model", required=True, choices=["xgboost"])
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)

    args = parser.parse_args()

    if args.model == "xgboost":
        tune_xgboost(
            n_trials=args.n_trials,
            random_seed=args.random_seed,
            early_stopping_rounds=args.early_stopping_rounds,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
