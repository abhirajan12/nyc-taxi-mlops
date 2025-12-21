#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_DIR="$REPO_ROOT/mlflow"

mkdir -p "$MLFLOW_DIR/artifacts"

exec mlflow server \
  --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" \
  --default-artifact-root "$MLFLOW_DIR/artifacts" \
  --host 127.0.0.1 \
  --port 5000
