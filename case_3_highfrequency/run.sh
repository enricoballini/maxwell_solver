#!/bin/bash
set -e -o pipefail
cd "$(dirname "$0")"

if [ -f "$HOME/Desktop/work/code/venv/bin/activate" ]; then source "$HOME/Desktop/work/code/venv/bin/activate"; fi
mkdir -p results
(cd ../src && python Compile.py)
python -u highfrequency.py 2>&1 | tee results/log.txt
