#!/bin/sh

if [[ $1 -eq "training" ]]; then
    pip install -e .[training]
else
    pip install -e .
fi
