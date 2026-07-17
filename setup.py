"""
AMD ROCm HIP port of fp6_llm — setup.py

Build strategy:
  PyTorch's ROCm build system (CUDAExtension on ROCm) invokes hipcc to compile
  .hip files. The compiler flags key is still called "nvcc" in the PyTorch API
  (for historical reasons), but on ROCm the flags are passed to hipcc/clang++.

  Flags note (Issue #2 fix):
  - "nvcc" key → passed to hipcc (not nvcc) on ROCm systems
  - -Xcompiler removed: nvcc-specific flag, clang/hipcc doesn't need it
  - --offload-arch replaces -gencode (hipcc syntax)
  - -ffast-math removed: causes -Wnan-infinity-disabled errors with PyTorch headers

Usage:
  export PYTORCH_ROCM_ARCH="gfx942"    # MI300X / MI325X
  export PYTORCH_ROCM_ARCH="gfx950"    # MI350X (CDNA4)
  pip install -e . --no-build-isolation
"""
import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Target GPU architecture. Comma-separated for multi-arch builds.
# Examples: "gfx942" (MI300X/MI325X), "gfx950" (MI350X), "gfx942,gfx950"
ROCM_ARCH = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")

extra_compile_args = {
    # Host compiler (g++) flags
    "cxx": ["-O3", "-std=c++17"],
    # Device compiler (hipcc/clang++) flags.
    # NOTE: PyTorch's CUDAExtension uses the key "nvcc" for historical reasons,
    # but on ROCm systems these flags are forwarded to hipcc, not nvcc.
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--offload-arch=" + ROCM_ARCH,   # hipcc syntax (replaces nvcc -gencode)
        # -Xcompiler omitted: nvcc-specific, rejected by clang/hipcc
        # -ffast-math omitted: causes -Wnan-infinity-disabled in PyTorch headers
    ],
}

setup(
    name="fp6_llm",
    author="Haojun Xia et al. — AMD ROCm port by AMD-AIOSS",
    version="0.2.1",
    description=(
        "AMD ROCm HIP port of fp6_llm: "
        "efficient FPx-quantized LLM inference kernels (FP6, FP5, FP4)"
    ),
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=["torch"],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fp6_llm_cuda",
            sources=[
                "fp6_llm/csrc/pybind.cpp",       # PyTorch wrappers, g++ compiled
                "fp6_llm/csrc/fp6_linear.hip",   # GPU kernels, hipcc compiled
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
