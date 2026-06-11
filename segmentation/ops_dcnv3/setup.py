# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

requirements = ['torch', 'torchvision']


class BuildExtensionSkipCudaVersionCheck(BuildExtension):
    """Allow building extensions when the system nvcc version differs from torch."""

    def _check_cuda_version(self):
        try:
            super()._check_cuda_version()
        except RuntimeError as err:
            if 'mismatches the version that was used to compile' in str(err):
                print(f'WARNING: {err}\nProceeding with build anyway.')
            else:
                raise


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'src')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = [
            # "-DCUDA_HAS_FP16=1",
            # "-D__CUDA_NO_HALF_OPERATORS__",
            # "-D__CUDA_NO_HALF_CONVERSIONS__",
            # "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError('Cuda is not availabel')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            'DCNv3',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name='DCNv3',
    version='1.0',
    author='InternImage',
    url='https://github.com/OpenGVLab/InternImage',
    description=
    'PyTorch Wrapper for CUDA Functions of DCNv3',
    packages=find_packages(exclude=(
        'configs',
        'tests',
    )),
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtensionSkipCudaVersionCheck},
)
