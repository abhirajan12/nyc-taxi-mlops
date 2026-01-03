from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

TARGET_COL = "fare_amount"

def make_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    # minimal feature set to get pipeline working
    feature_cols = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "trip_duration_min",
    ]
    missing = [c for c in [TARGET_COL] + feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_cols].fillna(0).astype(float).to_numpy()
    y = df[TARGET_COL].astype(float).to_numpy()
    return X, y, feature_cols
