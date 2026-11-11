#!/bin/bash
set -e
cd "$(dirname "$0")"

source venv/bin/activate
(cd src && python Compile.py)
python -u GICsimulation.py          
