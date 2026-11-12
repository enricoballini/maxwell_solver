#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
(cd src && python Compile.py)
for test in tests/test_*.py; do
    echo "== $test"
    python -u "$test"
done
