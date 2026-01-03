from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

def main(test_size: float, val_size: float, seed: int) -> None:
    df = pd.read_parquet(IN_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_tmp, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(df_tmp, test_size=val_rel, random_state=seed)

    df_train.to_parquet(OUT_DIR / "train.parquet", index=False)
    df_val.to_parquet(OUT_DIR / "val.parquet", index=False)
    df_test.to_parquet(OUT_DIR / "test.parquet", index=False)

    print(
        f"Saved splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)} -> {OUT_DIR}"
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.test_size, args.val_size, args.seed)
