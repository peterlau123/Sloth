import torch


def create_tensor(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    requires_grad: bool = False,
    fill: Optional[Union[float, int, str]] = None,
) -> torch.Tensor:
    if fill is None:
        tensor = torch.randn(shape, dtype=dtype, device=device)
    elif fill == "ones":
        tensor = torch.ones(shape, dtype=dtype, device=device)
    elif fill == "zeros":
        tensor = torch.zeros(shape, dtype=dtype, device=device)
    elif isinstance(fill, (int, float)):
        tensor = torch.full(shape, fill, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown fill value: {fill}")

    tensor.requires_grad = requires_grad
    return tensor


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    return torch.allclose(actual, expected, rtol=rtol, atol=atol)


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def mean_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().mean().item()
