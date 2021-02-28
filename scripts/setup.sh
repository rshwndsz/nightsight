#!/bin/sh

python3 -m venv ./env
source ./env/bin/activate
pip3 install --upgrade pip

if [[ $1 -eq "training" ]]; then
    pip install -e .[training]
else
    pip install -e .
fi

source ./env/bin/activate
