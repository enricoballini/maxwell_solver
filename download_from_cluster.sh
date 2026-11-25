#!/bin/bash
set -e
cd "$(dirname "$0")"

remote_dir=/home/enba/work_1/maxwell_solver
ssh_options=(-o ControlMaster=auto -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=60)

ssh "${ssh_options[@]}" HPsp "cd $remote_dir && rm -f results.zip && find . -maxdepth 2 -type d -path './case_*/results' | zip -qr results.zip -@"
scp "${ssh_options[@]}" HPsp:$remote_dir/results.zip .
ssh "${ssh_options[@]}" HPsp "rm $remote_dir/results.zip"
unzip -qo results.zip
rm results.zip
