from .base import BaseRunner
from .sloth_runner import SlothCuteRunner, SlothTritonRunner, SlothCutlassRunner
from .torch_runner import TorchRunner, TorchSoftmaxRunner, TorchGemmRunner
from .leetcuda_runner import LeetCUDARunner

__all__ = [
    "BaseRunner",
    "SlothCuteRunner",
    "SlothTritonRunner",
    "SlothCutlassRunner",
    "TorchRunner",
    "TorchSoftmaxRunner",
    "TorchGemmRunner",
    "LeetCUDARunner",
]


def get_runner(name: str, operator: str) -> BaseRunner:
    runner_map = {
        "sloth-cute": lambda op: SlothCuteRunner(),
        "sloth-triton": lambda op: SlothTritonRunner(),
        "sloth-cutlass": lambda op: SlothCutlassRunner(),
        "torch": lambda op: get_torch_runner(op),
        "leetcuda": lambda op: LeetCUDARunner(op),
    }

    if name not in runner_map:
        raise ValueError(f"Unknown runner: {name}")

    return runner_map[name](operator)


def get_torch_runner(operator: str) -> BaseRunner:
    runner_map = {
        "add": TorchRunner,
        "softmax": TorchSoftmaxRunner,
        "gemm": TorchGemmRunner,
    }

    if operator not in runner_map:
        raise ValueError(f"No torch runner for operator: {operator}")

    return runner_map[operator]()
