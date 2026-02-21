from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

csrc_path = os.path.join(os.path.dirname(__file__), "csrc")

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
                extra_compile_args={
                    "cxx": ["-03"],
                    "nvcc": [
                        "-03",
                        "--use_fast_math",
                        ],
                    },
                )
            ],
        cmdclass={
            "build_ext": BuildExtension
            },
        )
