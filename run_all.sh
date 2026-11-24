#!/bin/bash
set -e
cd "$(dirname "$0")"

for run_script in case_*/run.sh; do
    bash "$run_script"
done
