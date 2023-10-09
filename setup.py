from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17"
    ],
    "nvcc": [
        "-O3", 
        "--use_fast_math",
        "-std=c++17",
        "-maxrregcount=255",
        "--ptxas-options=-v,-warn-lmem-usage,--warn-on-spills",
        "-gencode=arch=compute_80,code=sm_80"
    ],
}

setup(
    name="fp6_llm",
    author="Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song",
    version="0.1",
    author_email="xhjustc@gmail.com",
    description = "An efficient GPU support for LLM inference with 6-bit quantization (FP6).",
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers"
    ],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fp6_llm",
            sources=[
                "fp6_llm/csrc/pybind.cpp", 
                "fp6_llm/csrc/fp6_linear.cu"
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)