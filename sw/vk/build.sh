#!/usr/bin/env bash
# 着色器编成 SPIR-V，vkrun（计算管线）与 vkdraw（图形管线）编成可执行文件。要 glslang-tools 与 libvulkan-dev；
# 跑的时候有一个 Vulkan 设备即可（流水线上与本机都是 mesa-vulkan-drivers 里的 lavapipe）。用法：build.sh <输出目录>
set -euo pipefail
cd "$(dirname "$0")"
O=$(realpath -m "$1")
mkdir -p "$O"
for f in kernels/*.comp draw/*.vert draw/*.frag; do
  glslangValidator -V --target-env vulkan1.3 -o "$O/$(basename "$f" | tr . _).spv" "$f" > /dev/null
done
for t in vkrun vkdraw; do
  cc -std=c23 -O2 -Wall -Wextra -Werror -o "$O/$t" "$t.c" -lvulkan
done
echo "$O：vkrun、vkdraw 与 $(ls "$O"/*.spv | wc -l) 个着色器"
