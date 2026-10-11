#!/usr/bin/env bash
# One-command launcher: creates the virtualenv on first use, then runs the app.
#   ./run.sh            -> live webcam drowsiness detection
#   ./run.sh evaluate   -> metrics + charts into results/
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)
    if [ -z "$PY" ]; then
        echo "Python 3.10-3.12 is required (TensorFlow does not support newer versions yet)."
        echo "On macOS: brew install python@3.12"
        exit 1
    fi
    echo "Creating virtual environment with $PY ..."
    "$PY" -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

case "${1:-detect}" in
    detect)   exec .venv/bin/python src/detect.py ;;
    evaluate) exec .venv/bin/python src/evaluate.py ;;
    *)        echo "Usage: ./run.sh [detect|evaluate]"; exit 1 ;;
esac
