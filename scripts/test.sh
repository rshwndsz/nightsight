#!/bin/sh

if [[ $1 = 'all' ]]; then
    echo "Running all tests"
    pytest -v
elif [[ $1 = 'slow' ]]; then
    echo "Running only slow tests. Grab a cup of tea."
    pytest -v -m "slow"
else
    echo "Running fast tests. Grab your seat."
    pytest -v -m "not slow"
fi