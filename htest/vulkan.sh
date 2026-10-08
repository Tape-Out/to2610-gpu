#!/usr/bin/env bash
# Vulkan 计算管线那一层的整片测试：编着色器与 vkrun，再在交付的 .v 上跑 test_vulkan.py。
# 用法：vulkan.sh <输出目录>；CHIP_ASIC 同 chip.sh
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"
bash sw/vk/build.sh "$O/vk"
VKBUILD=$O/vk CHIP_MODULE=test_vulkan bash htest/chip.sh "$O/chip"
