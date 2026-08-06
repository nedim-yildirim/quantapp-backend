#!/usr/bin/env bash
# Assembles the minimal engine subset the backend needs into backend/engine/,
# so the Docker image is self-contained (no dependency on ~/QuantProjectV2 at
# runtime). Run before building the image or committing for deploy.
#
#   ./bundle_engine.sh [path-to-QuantProjectV2]
set -euo pipefail

SRC="${1:-$HOME/QuantProjectV2}"
DST="$(cd "$(dirname "$0")" && pwd)/engine"

echo "Bundling engine from: $SRC"
rm -rf "$DST"
mkdir -p "$DST/data" "$DST/results" "$DST/models"

# Code the weekly job imports (config, feature engineering, signal helpers)
cp "$SRC/config.py"   "$DST/"
cp "$SRC/main.py"     "$DST/"
cp "$SRC/signals.py"  "$DST/"

# Trained model + the data the long-only backtest and rankings need
cp "$SRC/models/lgbm_latest.pkl"        "$DST/models/"
cp "$SRC/data/prices.parquet"           "$DST/data/"
cp "$SRC/results/predictions.parquet"   "$DST/results/"
cp "$SRC/results/ic_log.csv"            "$DST/results/"
# fundamentals cache is optional (weekly job refetches if absent)
cp "$SRC/data/fundamentals.json"        "$DST/data/" 2>/dev/null || true

echo "Bundled into: $DST"
du -sh "$DST"
