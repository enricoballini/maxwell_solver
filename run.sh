#!/bin/bash
# Build the CUDA library and run the GIC simulation with the Python venv (no MATLAB needed).
# Usage: ./run.sh
set -e
cd "$(dirname "$0")"

source venv/bin/activate
(cd src && python Compile.py)
python GICsimulation.py          # plot windows open via Qt
