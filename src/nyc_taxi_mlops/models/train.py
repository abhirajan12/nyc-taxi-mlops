from __future__ import annotations

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
TEST_PATH = SPLITS_DIR / "test.parquet"

EXPERIMENT_NAME = "nyc_taxi_mlops_experiment"
COMPARISON_GROUP = "final_train_v2"
SPLIT_STRATEGY = "standard_random_split_v1"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


# ------------------------------------------------------------------
# LOCKED XGBOOST PARAMETERS
#
# Chosen from MLflow tuning:
#   experiment: nyc_taxi_mlops_experiment
#   comparison_group: xgb_tuning_v2
#   run_id: 9011d38edc3d4b20af77d290795421ab
#   metric: rmse (best on validation)

# TODO: change these to your best parameters if model changed

# ------------------------------------------------------------------
BEST_XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}



CONFIG = {
    "target": "fare_amount",
    "model_type": "xgboost",
    "random_seed": 42,
    "n_estimators": 5000,          # large cap; no early stopping in final fit
    "run_name": "xgb_final_train_standard_split",
}


def train(cfg: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -------------------------
    # Load fixed splits
    # -------------------------
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val = pd.read_parquet(VAL_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    X_train, y_train, feature_cols = make_features(df_train)
    X_val, y_val, feature_cols_val = make_features(df_val)
    X_test, y_test, feature_cols_test = make_features(df_test)

    if feature_cols_val != feature_cols or feature_cols_test != feature_cols:
        raise ValueError("Feature columns mismatch across train/val/test splits.")

    # Combine train + val for final training
    X_train_full = pd.concat(
        [pd.DataFrame(X_train), pd.DataFrame(X_val)], ignore_index=True
    ).to_numpy()
    y_train_full = pd.concat(
        [pd.Series(y_train), pd.Series(y_val)], ignore_index=True
    ).to_numpy()

    model = XGBRegressor(
        n_estimators=cfg["n_estimators"],
        objective="reg:squarederror",
        random_state=cfg["random_seed"],
        n_jobs=-1,
        **BEST_XGB_PARAMS,
    )

    run_name = cfg.get("run_name", "final_train")

    with mlflow.start_run(run_name=run_name):
        # Tags for filtering
        mlflow.set_tag("comparison_group", COMPARISON_GROUP)
        mlflow.set_tag("split_strategy", SPLIT_STRATEGY)
        mlflow.set_tag("model_type", cfg["model_type"])
        mlflow.set_tag("variant", "final_train")
        mlflow.set_tag("target", cfg["target"])

        # Log config + locked params
        mlflow.log_params(cfg)
        mlflow.log_params({f"best_{k}": v for k, v in BEST_XGB_PARAMS.items()})

        # Data + feature metadata
        mlflow.log_param("train_rows", len(df_train))
        mlflow.log_param("val_rows", len(df_val))
        mlflow.log_param("test_rows", len(df_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("train_path", str(TRAIN_PATH))
        mlflow.log_param("val_path", str(VAL_PATH))
        mlflow.log_param("test_path", str(TEST_PATH))

        # Fit final model on train+val
        model.fit(X_train_full, y_train_full)

        # Evaluate once on test (this is the real final metric)
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_metric("rmse_test", rmse)
        mlflow.log_metric("mae_test", mae)

        # (Optional) sanity metrics on val, not for final claims
        preds_val = model.predict(X_val)
        rmse_val = root_mean_squared_error(y_val, preds_val)
        mae_val = mean_absolute_error(y_val, preds_val)
        mlflow.log_metric("rmse_val", rmse_val)
        mlflow.log_metric("mae_val", mae_val)

        # Log model artifact
        try:
            mlflow.xgboost.log_model(model, artifact_path="model")
        except Exception:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[FINAL] RMSE(test)={rmse:.3f} | MAE(test)={mae:.3f}")
        print(f"[SANITY] RMSE(val)={rmse_val:.3f} | MAE(val)={mae_val:.3f}")


if __name__ == "__main__":
    train(CONFIG)
