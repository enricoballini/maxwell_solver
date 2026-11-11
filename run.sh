#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
(cd src && python Compile.py)
python -u GICsimulation.py          
