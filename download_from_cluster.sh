#!/bin/bash
set -e
cd "$(dirname "$0")"

ssh -o ControlMaster=auto -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=60 HPsp 'cd /home/enba/work_1/maxwell_solver && rm -f results.zip && zip -qr results.zip results'
scp -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" HPsp:/home/enba/work_1/maxwell_solver/results.zip "$HOME/Desktop/work/maxwell_solver/"
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" HPsp 'rm /home/enba/work_1/maxwell_solver/results.zip'
unzip -qo results.zip
rm results.zip
