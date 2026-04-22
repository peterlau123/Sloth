from .base import BaseRunner
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common", "python"))

from operators.01_elementwise.add import add


class SlothRunner(BaseRunner):
    def __init__(self, impl: str):
        super().__init__(f"sloth-{impl}")
        self.impl = impl

    def get_operator(self):
        return lambda a, b: add(a, b, impl=self.impl)

    def prepare_args(self, inputs):
        return (inputs["a"], inputs["b"])

    def run_operator(self, operator, args):
        return operator(*args)


class SlothCuteRunner(SlothRunner):
    def __init__(self):
        super().__init__("cute")


class SlothTritonRunner(SlothRunner):
    def __init__(self):
        super().__init__("triton")


class SlothCutlassRunner(SlothRunner):
    def __init__(self):
        super().__init__("cutlass")