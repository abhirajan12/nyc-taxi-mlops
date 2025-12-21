from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# Deterministic project root from this file location:
# repo_root/
# src/nyc_taxi_mlops/data/make_dataset.py  <-- here
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class Config:
    year: int
    month: int
    n_rows: int
    dataset: str  # "yellow" (default)


def _build_tlc_url(dataset: str, year: int, month: int) -> str:
    # NYC TLC trip-data parquet files are commonly hosted on this CloudFront path.
    # Example pattern: .../yellow_tripdata_2024-01.parquet
    mm = f"{month:02d}"
    return f"https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset}_tripdata_{year}-{mm}.parquet"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logging.info("Raw file already exists, skipping download: %s", out_path)
        return

    logging.info("Downloading: %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    logging.info("Saved: %s", out_path)


def _clean_yellow(df: pd.DataFrame) -> pd.DataFrame:
    # Keep a small, stable set of columns across recent schemas.
    keep = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "payment_type",
    ]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()

    # Basic cleaning
    for c in ["passenger_count", "trip_distance", "fare_amount", "tip_amount", "total_amount"]:
        if c in df.columns:
            df = df[pd.to_numeric(df[c], errors="coerce").notna()]

    if "trip_distance" in df.columns:
        df = df[df["trip_distance"] > 0]

    if "total_amount" in df.columns:
        df = df[df["total_amount"] >= 0]

    # Feature: trip duration in minutes (handy for later modeling/QA)
    if {"tpep_pickup_datetime", "tpep_dropoff_datetime"} <= set(df.columns):
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
        df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
        df["trip_duration_min"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
        )
        df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] < 180)]

    return df.reset_index(drop=True)


def main(cfg: Config) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    url = _build_tlc_url(cfg.dataset, cfg.year, cfg.month)
    raw_path = RAW_DIR / f"{cfg.dataset}_tripdata_{cfg.year}-{cfg.month:02d}.parquet"
    _download_file(url, raw_path)

    logging.info("Reading raw parquet (sample n=%s): %s", cfg.n_rows, raw_path)
    # Read full file then sample deterministically; keeps code simple and repeatable.
    df = pd.read_parquet(raw_path)
    if cfg.n_rows > 0 and len(df) > cfg.n_rows:
        df = df.sample(n=cfg.n_rows, random_state=42)

    if cfg.dataset == "yellow":
        df_clean = _clean_yellow(df)
    else:
        raise ValueError(f"Unsupported dataset type: {cfg.dataset}")

    out_path = PROCESSED_DIR / "train.parquet"
    df_clean.to_parquet(out_path, index=False)
    logging.info("Wrote processed train set: %s (rows=%s, cols=%s)", out_path, len(df_clean), df_clean.shape[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    p = argparse.ArgumentParser(description="Download + clean NYC TLC taxi data into .data/")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--n-rows", type=int, default=50_000, help="Deterministic sample size (0 = keep all)")
    p.add_argument("--dataset", type=str, default="yellow", choices=["yellow"])
    args = p.parse_args()

    main(Config(year=args.year, month=args.month, n_rows=args.n_rows, dataset=args.dataset))
