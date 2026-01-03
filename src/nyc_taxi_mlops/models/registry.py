# src/nyc_taxi_mlops/models/registry.py
from __future__ import annotations

import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "nyc_taxi_fare_model"
PROD_ALIAS = "prod"


def load_production_model():
    """
    Loads the current production model from MLflow Model Registry using the `prod` alias.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{PROD_ALIAS}"
    return mlflow.pyfunc.load_model(model_uri)


