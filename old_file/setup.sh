#!/bin/bash
# setup.sh — Create a virtual environment and install dependencies
#
# Usage:
#   source setup.sh        # Creates venv and activates it
#   ./setup.sh             # Creates venv (activate manually after)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at: $VENV_DIR"
else
    echo "  Venv already exists at: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "  Activated venv (python: $(which python3))"

echo "  Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

echo ""
echo "Setup complete. To activate in future sessions:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run the full pipeline:"
echo "  ./run_all.sh"
