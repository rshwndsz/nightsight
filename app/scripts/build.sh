#!/bin/sh

# Check if executing from the right place
if [ ! -f $PWD/CMakeLists.txt ]; then
    echo "Get into nightsight/app and then execute this script."
    exit
fi

# Configure build sytem if not done already
if [ ! -d $PWD/build ]; then
    ./scripts/configure.sh
fi

# Build
cd build
cmake --build . --config Release
cd ..
