from .base import BaseRunner
import torch


class TorchRunner(BaseRunner):
    def __init__(self):
        super().__init__("torch")

    def get_operator(self):
        return torch.add

    def prepare_args(self, inputs):
        return (inputs["a"], inputs["b"])

    def run_operator(self, operator, args):
        return operator(*args)


class TorchSoftmaxRunner(BaseRunner):
    def __init__(self):
        super().__init__("torch")

    def get_operator(self):
        return torch.softmax

    def prepare_args(self, inputs):
        return (inputs["x"], -1)

    def run_operator(self, operator, args):
        return operator(*args)


class TorchGemmRunner(BaseRunner):
    def __init__(self):
        super().__init__("torch")

    def get_operator(self):
        return torch.mm

    def prepare_args(self, inputs):
        return (inputs["a"], inputs["b"])

    def run_operator(self, operator, args):
        return operator(*args)
