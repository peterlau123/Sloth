# LeetCUDA Reference Implementation

This directory is a git submodule placeholder for [LeetCUDA](https://github.com/xlite-dev/LeetCUDA).

## Setup

LeetCUDA repository is large and may take time to clone. Initialize it with:

```bash
# Option 1: Initialize all submodules (may take a while)
git submodule update --init --recursive

# Option 2: Clone with shallow depth (faster)
git submodule update --init --depth 1 benchmarks/third_party/LeetCUDA

# Option 3: Manual clone
git clone https://github.com/xlite-dev/LeetCUDA.git benchmarks/third_party/LeetCUDA
```

## What is LeetCUDA?

[LeetCUDA](https://github.com/xlite-dev/LeetCUDA) is a comprehensive CUDA operators repository with high-quality implementations. It serves as our reference for benchmark comparisons.

## Usage in Benchmarks

Once initialized, the benchmark framework will automatically detect LeetCUDA:

```bash
python benchmarks/run_benchmark.py add --impls sloth-triton,torch,leetcuda
```