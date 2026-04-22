from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import subprocess


def get_cuda_arch():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True
        )
        arch = result.stdout.strip().replace(".", "")
        return f"sm_{arch}" if arch else "sm_80"
    except:
        return "sm_80"


cuda_arch = get_cuda_arch()

ext_modules = [
    Pybind11Extension(
        "sloth._sloth",
        sources=[
            "common/cpp/bindings.cpp",
            "common/cpp/utils/timer.cpp",
            "common/cpp/utils/checker.cpp",
            "operators/01_elementwise/add/cute/add_kernel.cu",
            "operators/01_elementwise/add/cutlass/add_kernel.cu",
        ],
        include_dirs=[
            "common/cpp/include",
            "operators",
            os.path.join(os.environ.get("CUTLASS_PATH", "third_party/cutlass"), "include"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", f"-arch={cuda_arch}", "--expt-relaxed-constexpr"],
        },
        libraries=["cudart"],
    ),
]

setup(
    name="sloth",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["sloth"],
    package_dir={"": "common/python"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    zip_safe=False,
)
