"""Modal-only FP32 build: upstream 64x64 and NVIDIA 512x32 slice paths.

No fast-math or reduced-precision flags: numerical equivalence is a release gate.
"""

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="p2i_physics_attention_cuda",
    ext_modules=[
        CUDAExtension(
            "p2i_physics_attention_cuda",
            [str(Path(__file__).parent / "cuda/physics_attention_kernel.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
