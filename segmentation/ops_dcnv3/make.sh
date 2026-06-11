#!/usr/bin/env bash
# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
# On recent NVIDIA toolchains, auto-detect can return unsupported arch values.
# Keep H100-compatible default while still allowing override from environment.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

python setup.py build install
