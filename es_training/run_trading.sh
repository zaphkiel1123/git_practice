#!/bin/bash
# run_trading.sh — Full trading system pipeline:
#   train → backtest → explain → generate viewer output
#
# Usage:
#   ./run_trading.sh                         # Use data/ directory with defaults
#   ./run_trading.sh /path/to/data 1min 1.5  # Custom data dir, window, RR

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/data}"
WINDOW="${2:-1min}"
RR="${3:-1.5}"
MAX_HOLD="${4:-60}"
FOLDS="${5:-5}"
DATA_DIR="$(cd "$DATA_DIR" 2>/dev/null && pwd || echo "$DATA_DIR")"
OUTPUT_DIR="${DATA_DIR}/models"

# Python detection
if [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: python3 not found"
    exit 1
fi

echo "========================================"
echo "  ES Mini Trading System Pipeline"
echo "========================================"
echo "  Data directory: $DATA_DIR"
echo "  Window:         $WINDOW"
echo "  Risk/Reward:    $RR"
echo "  Max hold:       $MAX_HOLD bars"
echo "  WF folds:       $FOLDS"
echo "  Output:         $OUTPUT_DIR"
echo "  Python:         $PYTHON"
echo "========================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Train models
echo "[Step 1/4] Training models..."
$PYTHON "$SCRIPT_DIR/train_trading_model.py" "$DATA_DIR" \
    --window "$WINDOW" \
    --rr "$RR" \
    --max-hold "$MAX_HOLD" \
    --folds "$FOLDS" \
    --output-dir "$OUTPUT_DIR"
echo ""

# Step 2: Backtest
echo "[Step 2/4] Running backtest..."
$PYTHON "$SCRIPT_DIR/backtester.py" "$OUTPUT_DIR" \
    --rr "$RR" \
    --max-hold "$MAX_HOLD" \
    --output "$OUTPUT_DIR"
echo ""

# Step 3: Explain model factors
echo "[Step 3/4] Generating factor analysis..."
$PYTHON "$SCRIPT_DIR/explain_model.py" "$OUTPUT_DIR" \
    --trades "$OUTPUT_DIR/backtest_trades.csv" \
    --output "$OUTPUT_DIR"
echo ""

# Step 4: Generate viewer output
echo "[Step 4/4] Generating viewer output..."
$PYTHON "$SCRIPT_DIR/generate_trade_output.py" "$OUTPUT_DIR" \
    --trades "$OUTPUT_DIR/backtest_trades.csv" \
    --output "$OUTPUT_DIR/viewer_output"
echo ""

echo "========================================"
echo "  PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Models:           $OUTPUT_DIR/signal_model.joblib"
echo "                    $OUTPUT_DIR/volatility_model.joblib"
echo "                    $OUTPUT_DIR/quality_model.joblib"
echo "  Backtest:         $OUTPUT_DIR/backtest_trades.csv"
echo "                    $OUTPUT_DIR/backtest_metrics.json"
echo "  Factor report:    $OUTPUT_DIR/factor_report.txt"
echo "  Viewer files:     $OUTPUT_DIR/viewer_output/bars.data"
echo "                    $OUTPUT_DIR/viewer_output/trades.json"
echo ""
echo "To visualize:"
echo "  1. Open view_data/trade_viewer.html in a browser"
echo "  2. Load '$OUTPUT_DIR/viewer_output/bars.data' (minute data)"
echo "  3. Load '$OUTPUT_DIR/viewer_output/trades.json' (trade markers)"
echo ""
echo "For live signals:"
echo "  $PYTHON $SCRIPT_DIR/live_signal.py $OUTPUT_DIR --file <new_session.data>"
echo "  $PYTHON $SCRIPT_DIR/live_signal.py $OUTPUT_DIR --watch <live_data_dir>"
