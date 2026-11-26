#!/bin/bash
set -e -o pipefail
cd "$(dirname "$0")"

if [ -f "$HOME/Desktop/work/code/gprMax/.venv/bin/activate" ]; then source "$HOME/Desktop/work/code/gprMax/.venv/bin/activate"; fi
# nvcc on PATH (container), otherwise the one pip installed in the workspace venv; PyCUDA compiles the kernels at run time
command -v nvcc > /dev/null || export PATH="$PATH:$HOME/Desktop/work/code/venv/lib/python3.12/site-packages/nvidia/cu13/bin"
mkdir -p results
python -u GICsimulation.py 2>&1 | tee results/log.txt
