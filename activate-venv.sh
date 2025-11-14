#!/bin/bash
# dev_venv.sh
# Create and activate a Python virtual environment in the script's directory,
# but skip creation/activation if already inside a venv.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Detect if we're already inside a venv
if [ -n "$VIRTUAL_ENV" ]; then
  echo "[WARNING] Already inside a virtual environment at: $VIRTUAL_ENV"
  echo "Skipping creation/activation of $VENV_DIR."
else
  # Ensure python3 is available
  if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Please install Python 3 first."
    return 1
  fi

  # Create venv if it doesn't exist
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
  else
    echo "Virtual environment already exists at $VENV_DIR"
  fi

  # Activate venv
  echo "Activating virtual environment..."
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  # Upgrade pip/setuptools/wheel inside venv
  pip install --upgrade pip setuptools wheel

  # Install dev tools
  pip install black flake8 isort mypy pytest ipython

  echo "✅ Development environment ready in $VENV_DIR."
  echo "To deactivate, run: deactivate"
fi
