from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
csrc_path = os.path.join(this_dir, "csrc")

torch_dir = os.path.dirname(torch.__file__)
torch_lib = os.path.join(torch_dir, "lib")

extra_link_args = [
        f"-Wl,-rpath,{torch_lib}",
        "-Wl,-rpath,$ORIGIN"
        ]

ext = CUDAExtension(
        name="TruncGPTQ._C",
        sources=[
            os.path.join("csrc", "binding.cpp"),
            os.path.join("csrc", "gptq_kernel.cu"),
            ],
        include_dirs=[csrc_path],
        library_dirs=[torch_lib],
        extra_link_args=extra_link_args,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
            },
        )


setup(
        name="TruncGPTQ",
        version="0.1.0",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=[ext],
        cmdclass={
            "build_ext": BuildExtension
            },
        )
