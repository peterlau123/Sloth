"""
Elementwise Add Operator

Provides three implementations:
- cute: cuTe DSL implementation
- triton: Triton implementation
- cutlass: CUTLASS implementation
"""

import torch
from typing import Optional

try:
    from sloth._sloth import add_cute, add_cutlass
except ImportError:
    add_cute = None
    add_cutlass = None

from .triton.add_kernel import add_triton


def add(
    a: torch.Tensor,
    b: torch.Tensor,
    impl: str = "torch",
) -> torch.Tensor:
    """
    Elementwise add operation.

    Args:
        a: First input tensor
        b: Second input tensor
        impl: Implementation to use ("cute", "triton", "cutlass", "torch")

    Returns:
        Output tensor c = a + b
    """
    if impl == "torch":
        return a + b
    elif impl == "triton":
        return add_triton(a, b)
    elif impl == "cute":
        if add_cute is None:
            raise ImportError("cuTe implementation not available. Please rebuild the library.")
        return add_cute(a, b)
    elif impl == "cutlass":
        if add_cutlass is None:
            raise ImportError("CUTLASS implementation not available. Please rebuild the library.")
        return add_cutlass(a, b)
    else:
        raise ValueError(f"Unknown implementation: {impl}. Choose from: cute, triton, cutlass, torch")
