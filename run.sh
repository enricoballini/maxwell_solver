#!/bin/bash
set -e -o pipefail
cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
mkdir -p results
(cd src && python Compile.py)
python -u GICsimulation.py 2>&1 | tee results/log.txt
