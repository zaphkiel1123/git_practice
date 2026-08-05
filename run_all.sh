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

# Use the Intel Python 3.9.6 with pandas/numpy/sklearn pre-installed
# Fall back to python3 if not available
if [ -x "/usr/intel/pkgs/python3/3.9.6/modules/r1/bin/python3" ]; then
    PYTHON="/usr/intel/pkgs/python3/3.9.6/modules/r1/bin/python3"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON="python3"
fi

echo "========================================"
echo "  Price Direction Model Pipeline"
echo "========================================"
echo "  Data directory: $DATA_DIR"
echo "  Window:         $WINDOW"
echo "  Horizon:        $HORIZON"
echo "  Python:         $PYTHON"
echo "========================================"
echo ""

# Step 1: Feature extraction
echo "[Step 1/2] Extracting features..."
$PYTHON "$SCRIPT_DIR/feature_pipeline.py" "$DATA_DIR" \
    --window "$WINDOW" \
    --horizon "$HORIZON" \
    --output "$DATA_DIR/features.csv"

echo ""

# Step 2: Model training
echo "[Step 2/2] Training models..."
$PYTHON "$SCRIPT_DIR/train_model.py" "$DATA_DIR/features.csv" \
    --folds 5 \
    --output-dir "$DATA_DIR"

echo ""
echo "Pipeline complete. Outputs in $DATA_DIR:"
echo "  - features.csv            (feature matrix)"
echo "  - direction_model.pkl     (trained classifier)"
echo "  - magnitude_model.pkl     (trained regressor)"
echo "  - model_results.json      (metrics & config)"
