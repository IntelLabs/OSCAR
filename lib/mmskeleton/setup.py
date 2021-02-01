import os
import sys
import platform
import subprocess
import time

from setuptools import find_packages, setup, Extension
from setuptools.command.install import install

import numpy as np
from Cython.Build import cythonize  # noqa: E402
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class TorchAndCythonBuildExtension(BuildExtension):
    def finalize_options(self):
        if self.distribution.ext_modules:
            nthreads = getattr(self, "parallel", None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            from Cython.Build.Dependencies import cythonize

            self.distribution.ext_modules[:] = cythonize(
                self.distribution.ext_modules, nthreads=nthreads, force=self.force
            )
        super().finalize_options()


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != "Windows":
        extra_compile_args = {"cxx": ["-Wno-unused-function", "-Wno-write-strings"]}
    extension = Extension(
        "{}.{}".format(module, name),
        [os.path.join("mmskeleton", *module.split("."), p) for p in sources],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    (extension,) = cythonize(extension)
    return extension


def make_cuda_ext(name, module, sources, include_dirs=[]):

    define_macros = []

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
    else:
        raise EnvironmentError("CUDA is required to compile MMSkeleton!")

    return CUDAExtension(
        name="{}.{}".format(module, name),
        sources=[os.path.join("mmskeleton", *module.split("."), p) for p in sources],
        define_macros=define_macros,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
mmskl_home = r'{}'
"""
    SHA = "b4c076"
    SHORT_VERSION = "0.7rc1"
    VERSION = SHORT_VERSION + "+" + SHA
    MMSKELETON_HOME = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mmskeleton"
    )

    with open("mmskeleton/mmskeleton/version.py", "w") as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION, MMSKELETON_HOME))


write_version_py()

setup(
    name="mmskeleton",
    version="0.7rc1+b4c076",
    description="Open MMLab Skeleton-based Human Understanding Toolbox",
    url="https://github.com/open-mmlab/mmskeleton",
    package_dir={"": "mmskeleton"},
    packages=["mmskeleton"],
    package_data={"mmskeleton.ops": ["*/*.so"]},
    license="Apache License 2.0",
    setup_requires=["setuptools", "wheel", "Cython", "torch", "numpy"],
    install_requires=["mmcv", "torch>=1.1", "torchvision", "lazy_import"],
    cmdclass={"build_ext": TorchAndCythonBuildExtension},
    ext_modules=[
        make_cython_ext(
            name="cpu_nms", module="mmskeleton.ops.nms", sources=["cpu_nms.pyx"]
        ),
        make_cuda_ext(
            name="gpu_nms",
            module="mmskeleton.ops.nms",
            sources=["nms_kernel.cu", "gpu_nms.pyx"],
            include_dirs=[np.get_include()],
        ),
    ],
    zip_safe=False,
)
