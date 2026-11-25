#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f "$HOME/Desktop/work/venv/bin/activate" ]; then source "$HOME/Desktop/work/venv/bin/activate"; fi
python -u GICpostprocess.py
"$HOME/Desktop/software/ParaView-6.2.1-MPI-Linux-Python3.12-x86_64/bin/pvbatch" --force-offscreen-rendering GICparaview_state.py
