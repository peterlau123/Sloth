import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common", "python"))

from operators.01_elementwise.add import add


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture
def small_tensors(device):
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    return a, b


@pytest.fixture
def medium_tensors(device):
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    return a, b


class TestAddCorrectness:
    def test_add_triton_correctness(self, small_tensors):
        a, b = small_tensors
        expected = a + b
        result = add(a, b, impl="triton")
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_add_torch_correctness(self, small_tensors):
        a, b = small_tensors
        expected = a + b
        result = add(a, b, impl="torch")
        assert torch.allclose(result, expected)

    def test_add_different_shapes(self, device):
        shapes = [(100,), (100, 100), (10, 100, 100)]
        for shape in shapes:
            a = torch.randn(shape, device=device)
            b = torch.randn(shape, device=device)
            result = add(a, b, impl="triton")
            expected = a + b
            assert torch.allclose(result, expected)

    def test_add_different_dtypes(self, device):
        dtypes = [torch.float32, torch.float16]
        for dtype in dtypes:
            a = torch.randn(100, 100, dtype=dtype, device=device)
            b = torch.randn(100, 100, dtype=dtype, device=device)
            result = add(a, b, impl="triton")
            expected = a + b
            assert torch.allclose(result, expected, rtol=1e-3 if dtype == torch.float16 else 1e-5)