#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f "$HOME/Desktop/work/venv/bin/activate" ]; then source "$HOME/Desktop/work/venv/bin/activate"; fi
python -u GICpostprocess.py
