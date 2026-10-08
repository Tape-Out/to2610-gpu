#!/usr/bin/env bash
# OpenGL 那一层的整片测试：编 gldraw，再在交付的 .v 上跑 test_opengl.py。
# 用法：opengl.sh <输出目录>；CHIP_ASIC 同 chip.sh
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"
bash sw/gl/build.sh "$O/gl"
GLBUILD=$O/gl CHIP_MODULE=test_opengl bash htest/chip.sh "$O/chip"
