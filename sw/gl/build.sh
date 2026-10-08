#!/usr/bin/env bash
# gldraw 编成可执行文件。要 libegl-dev、libopengl-dev、libgl-dev；跑在 Mesa 的 llvmpipe 上（EGL 的无表面平台，不要窗口）。
# 用法：build.sh <输出目录>
set -euo pipefail
cd "$(dirname "$0")"
O=$(realpath -m "$1")
mkdir -p "$O"
cc -std=c23 -O2 -Wall -Wextra -Werror -o "$O/gldraw" gldraw.c -lEGL -lOpenGL
echo "$O：gldraw"
