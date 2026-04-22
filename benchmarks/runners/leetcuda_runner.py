from .base import BaseRunner
import torch
import os
import importlib.util


class LeetCUDARunner(BaseRunner):
    def __init__(self, operator_name: str):
        super().__init__("leetcuda")
        self.operator_name = operator_name
        self._module = None
        self._load_module()

    def _load_module(self):
        leetcuda_path = os.path.join(os.path.dirname(__file__), "..", "third_party", "LeetCUDA")
        if os.path.exists(leetcuda_path):
            spec = importlib.util.spec_from_file_location("leetcuda", os.path.join(leetcuda_path, "__init__.py"))
            if spec and spec.loader:
                self._module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self._module)

    def get_operator(self):
        if self._module is None:
            raise ImportError("LeetCUDA not available. Add it as submodule in benchmarks/third_party/LeetCUDA")

        operator_map = {
            "add": self._module.add if hasattr(self._module, "add") else None,
            "softmax": self._module.softmax if hasattr(self._module, "softmax") else None,
            "gemm": self._module.gemm if hasattr(self._module, "gemm") else None,
        }

        op = operator_map.get(self.operator_name)
        if op is None:
            raise NotImplementedError(f"LeetCUDA does not have implementation for {self.operator_name}")
        return op

    def prepare_args(self, inputs):
        if self.operator_name == "add":
            return (inputs["a"], inputs["b"])
        elif self.operator_name == "softmax":
            return (inputs["x"],)
        elif self.operator_name == "gemm":
            return (inputs["a"], inputs["b"])
        return tuple(inputs.values())

    def run_operator(self, operator, args):
        return operator(*args)
