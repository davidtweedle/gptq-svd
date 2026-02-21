from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, torch

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

extra_link_args = [f"-Wl,-rpath,{torch_lib}"]

csrc_path = "csrc"

setup(
        name="TruncGPTQ",
        version="0.1.0",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=[
            CUDAExtension(
                name="TruncGPTQ._C",
                sources=[
                    os.path.join(csrc_path, "binding.cpp"),
                    os.path.join(csrc_path, "gptq_kernel.cu"),
                    ],
                extra_link_args=extra_link_args,
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        ],
                    },
                )
            ],
        cmdclass={
            "build_ext": BuildExtension
            },
        )
