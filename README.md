# Sloth - CUDA Operators Learning Project

A comprehensive CUDA operators learning project implementing operators from simple to complex, with three implementations per operator: **cuTe DSL**, **Triton**, and **CUTLASS**.

## Project Structure

```
Sloth/
├── operators/                    # All operators (organized by difficulty)
│   ├── 01_elementwise/           # Elementwise operations (beginner)
│   │   └── add/
│   │       ├── cute/             # cuTe DSL implementation
│   │       ├── triton/           # Triton implementation
│   │       ├── cutlass/          # CUTLASS implementation
│   │       └── __init__.py       # Unified Python interface
│   ├── 02_reduce/                # Reduction operations (intermediate)
│   ├── 03_gemm/                  # Matrix multiplication (advanced)
│   ├── 04_conv/                  # Convolution operations (advanced)
│   └── 05_attention/             # Attention mechanisms (advanced)
│
├── tests/                        # pytest test suite
│   ├── correctness/              # Correctness tests
│   ├── performance/              # Performance tests (pytest-benchmark)
│   └── conftest.py
│
├── benchmarks/                   # Benchmark framework
│   ├── configs/
│   │   └── workload.yaml         # Workload configurations
│   ├── runners/                  # Implementation runners
│   ├── compare/                  # Comparison tools
│   ├── third_party/
│   │   └── LeetCUDA/             # Reference implementation (git submodule)
│   └── run_benchmark.py          # CLI entry point
│
├── common/
│   ├── cpp/                      # C++ utilities
│   │   ├── include/utils/
│   │   └── bindings.cpp          # pybind11 bindings
│   └── python/sloth/             # Python package
│
├── scripts/                      # Build and run scripts
│
├── pyproject.toml
├── setup.py
└── CMakeLists.txt
```

## Features

- **Three implementations per operator**: cuTe DSL, Triton, and CUTLASS
- **Unified Python interface**: Easy to switch between implementations
- **Comprehensive testing**: pytest for correctness, pytest-benchmark for performance
- **Benchmark comparison**: Compare against PyTorch and LeetCUDA with speedup reports
- **Workload configuration**: YAML-based workload definitions

## Installation

### Prerequisites

- CUDA Toolkit 11.x+ (with compatible GPU, SM 80+ recommended)
- Python 3.9+
- PyTorch 2.0+
- Triton 2.1+

### Clone with Submodules

```bash
# Clone repository with submodules
git clone --recursive https://github.com/your-username/Sloth.git

# Or initialize submodules after cloning
git submodule update --init --recursive
```

### Build

```bash
# Install Python dependencies
pip install -e ".[dev]"

# Build C++/CUDA extensions
python setup.py build_ext --inplace
```

## Usage

### Running Operators

```python
import torch
from operators.01_elementwise.add import add

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")

# Use different implementations
result_triton = add(a, b, impl="triton")
result_torch = add(a, b, impl="torch")
result_cute = add(a, b, impl="cute")  # requires C++ build
result_cutlass = add(a, b, impl="cutlass")  # requires C++ build
```

### Running Tests

```bash
# Correctness tests
pytest tests/correctness/

# Performance tests
pytest tests/performance/
```

### Running Benchmarks

```bash
# Benchmark single operator
python benchmarks/run_benchmark.py add

# Benchmark all operators
python benchmarks/run_benchmark.py --all

# Specify implementations
python benchmarks/run_benchmark.py add --impls sloth-triton,torch

# Save report
python benchmarks/run_benchmark.py add --output reports/add.md --format markdown
```

## Benchmark Output Example

```
================================================================================
                           BENCHMARK REPORT: add
================================================================================

Workload: medium (4096x4096, float32)
----------------------------------------------
| Implementation | Time (ms) | vs Torch    |
|----------------|-----------|-------------|
| Sloth-triton   | 0.312     | 1.45x       |
| Sloth-cute     | 0.350     | 1.28x       |
| Sloth-cutlass  | 0.380     | 1.18x       |
| Torch          | 0.450     | 1.00x       |
----------------------------------------------

SUMMARY (Geometric Mean Speedup vs Torch)
----------------------------------------------
| Implementation | Speedup   |
|----------------|-----------|
| Sloth-triton   | 1.42x     |
| Sloth-cute     | 1.34x     |
| Sloth-cutlass  | 1.28x     |
----------------------------------------------
```

## Adding a New Operator

1. Create operator directory: `operators/<level>/<operator_name>/`
2. Create subdirectories: `cute/`, `triton/`, `cutlass/`
3. Implement each version with unified interface in `__init__.py`
4. Add workload configuration in `benchmarks/configs/workload.yaml`
5. Write tests in `tests/correctness/test_<operator>.py`

## Operators Roadmap

| Level | Operator | Status |
|-------|----------|--------|
| 01_elementwise | add | Done (template) |
| 01_elementwise | mul | Planned |
| 01_elementwise | relu | Planned |
| 02_reduce | sum | Planned |
| 02_reduce | softmax | Planned |
| 02_reduce | layer_norm | Planned |
| 03_gemm | gemm | Planned |
| 03_gemm | gemv | Planned |
| 04_conv | conv2d | Planned |
| 04_conv | conv3d | Planned |
| 05_attention | flash_attention | Planned |

## References

- [cuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [Triton](https://github.com/openai/triton)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) (reference implementation)

## License

MIT License