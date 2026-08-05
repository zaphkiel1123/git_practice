#!/bin/bash
# run_all.sh — End-to-end pipeline: decode → features → train → report
#
# Usage:
#   ./run_all.sh                          # Use defaults (current dir)
#   ./run_all.sh /path/to/data 1min 5min  # Custom data dir, window, horizon

set -e

DATA_DIR="${1:-.}"
WINDOW="${2:-1min}"
HORIZON="${3:-5min}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Price Direction Model Pipeline"
echo "========================================"
echo "  Data directory: $DATA_DIR"
echo "  Window:         $WINDOW"
echo "  Horizon:        $HORIZON"
echo "========================================"
echo ""

# Step 1: Feature extraction
echo "[Step 1/2] Extracting features..."
python3 "$SCRIPT_DIR/feature_pipeline.py" "$DATA_DIR" \
    --window "$WINDOW" \
    --horizon "$HORIZON" \
    --output "$DATA_DIR/features.parquet"

echo ""

# Step 2: Model training
echo "[Step 2/2] Training models..."
python3 "$SCRIPT_DIR/train_model.py" "$DATA_DIR/features.parquet" \
    --folds 5 \
    --output-dir "$DATA_DIR"

echo ""
echo "Pipeline complete. Outputs in $DATA_DIR:"
echo "  - features.parquet      (feature matrix)"
echo "  - direction_model.txt   (trained classifier)"
echo "  - magnitude_model.txt   (trained regressor)"
echo "  - model_results.json    (metrics & config)"
