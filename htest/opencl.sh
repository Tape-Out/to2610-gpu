#!/usr/bin/env bash
# OpenCL 那一层的整片测试：编带 tgpu 设备的 PoCL 与 clrun，再在交付的 .v 上跑 test_opencl.py。
# 用法：opencl.sh <输出目录>；CHIP_ASIC 同 chip.sh
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"
bash sw/pocl/build.sh "$O/pocl" "$O/pocl/inst"
CLRUN=$O/pocl/inst/bin/clrun CHIP_MODULE=test_opencl bash htest/chip.sh "$O/chip"
