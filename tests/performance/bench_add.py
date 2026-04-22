import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common", "python"))

from operators.01_elementwise.add import add


class TestAddPerformance:
    @pytest.fixture
    def large_tensors(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        a = torch.randn(16384, 16384, device="cuda")
        b = torch.randn(16384, 16384, device="cuda")
        return a, b

    @pytest.benchmark(group="add")
    def test_add_triton_performance(self, benchmark, large_tensors):
        a, b = large_tensors
        benchmark(add, a, b, impl="triton")

    @pytest.benchmark(group="add")
    def test_add_torch_performance(self, benchmark, large_tensors):
        a, b = large_tensors
        benchmark(lambda x, y: x + y, a, b)