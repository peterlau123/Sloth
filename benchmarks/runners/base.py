from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import torch
import time


class BaseRunner(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_operator(self) -> Callable:
        pass

    def benchmark(
        self,
        inputs: Dict[str, torch.Tensor],
        warmup: int = 10,
        repeat: int = 100,
    ) -> Dict[str, Any]:
        operator = self.get_operator()
        args = self.prepare_args(inputs)

        for _ in range(warmup):
            self.run_operator(operator, args)

        times = []
        torch.cuda.synchronize()

        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.run_operator(operator, args)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        import numpy as np

        times_array = np.array(times)

        return {
            "name": self.name,
            "times": times,
            "mean_ms": float(times_array.mean()),
            "std_ms": float(times_array.std()),
            "min_ms": float(times_array.min()),
            "max_ms": float(times_array.max()),
            "median_ms": float(np.median(times_array)),
        }

    @abstractmethod
    def prepare_args(self, inputs: Dict[str, torch.Tensor]) -> Any:
        pass

    @abstractmethod
    def run_operator(self, operator: Callable, args: Any) -> torch.Tensor:
        pass

    def verify(self, output: torch.Tensor, reference: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return torch.allclose(output, reference, rtol=rtol, atol=atol)
