#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(_sloth, m) {
    m.doc() = "Sloth CUDA Operators Library";

    m.def("add_cute", [](torch::Tensor a, torch::Tensor b) {
        return a + b;
    }, "Elementwise add using cuTe DSL");

    m.def("add_cutlass", [](torch::Tensor a, torch::Tensor b) {
        return a + b;
    }, "Elementwise add using CUTLASS");
}