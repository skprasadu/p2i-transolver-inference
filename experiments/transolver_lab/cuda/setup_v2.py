"""Build only on Modal's CUDA devel image."""

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="p2i_tiled_softmax",
    ext_modules=[
        CUDAExtension(
            "p2i_tiled_softmax",
            [str(Path(__file__).parent / "tiled_softmax.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
