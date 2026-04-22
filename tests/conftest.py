import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def torch_reference():
    return lambda a, b: a + b
