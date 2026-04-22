#!/bin/bash

set -e

echo "Running tests..."

pip install pytest pytest-benchmark pytest-cov

pytest tests/ -v --tb=short

echo "Tests complete!"