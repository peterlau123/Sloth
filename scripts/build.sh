#!/bin/bash

set -e

echo "Building Sloth..."

mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "Build complete!"