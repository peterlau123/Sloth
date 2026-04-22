import torch
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class BenchmarkResult:
    name: str
    times: List[float]
    mean: float
    std: float
    min: float
    max: float

    @property
    def median(self) -> float:
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]


def benchmark(
    func: Callable,
    args: Tuple,
    kwargs: Optional[Dict] = None,
    warmup: int = 10,
    repeat: int = 100,
    cuda: bool = True,
) -> BenchmarkResult:
    if kwargs is None:
        kwargs = {}

    times = []

    if cuda:
        torch.cuda.synchronize()
        for _ in range(warmup):
            func(*args, **kwargs)
        torch.cuda.synchronize()

        for _ in range(repeat):
            if cuda:
                torch.cuda.synchronize()
                start = time.perf_counter()
                func(*args, **kwargs)
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
            times.append((end - start) * 1000)
    else:
        for _ in range(warmup):
            func(*args, **kwargs)

        for _ in range(repeat):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    import numpy as np

    times_array = np.array(times)

    return BenchmarkResult(
        name=func.__name__,
        times=times,
        mean=float(times_array.mean()),
        std=float(times_array.std()),
        min=float(times_array.min()),
        max=float(times_array.max()),
    )


@contextmanager
def cuda_timer():
    class TimerResult:
        def __init__(self):
            self.elapsed_ms = 0.0

    result = TimerResult()
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield result
    torch.cuda.synchronize()
    end = time.perf_counter()
    result.elapsed_ms = (end - start) * 1000
