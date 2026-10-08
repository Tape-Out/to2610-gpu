#!/usr/bin/env bash
# 编一份带 tgpu 设备的 PoCL 与 clrun。PoCL 取 pocl.pin 钉住的那一版，打上登记设备的补丁、把 tgpu/ 拷进去；
# 不带 LLVM，所以没有 OpenCL C 编译器：这个设备只认内建内核与 sw/gpu.py build 出的 .tgk 二进制。
# 用法：build.sh <工作目录> <安装前缀>；clrun 装在 <安装前缀>/bin
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
W=$(realpath -m "$1")
P=$(realpath -m "$2")
read -r tag sha < "$here/pocl.pin"
mkdir -p "$W"
if [ ! -d "$W/src/.git" ]; then
  rm -rf "$W/src"
  git clone -q --depth 1 --branch "$tag" https://github.com/pocl/pocl.git "$W/src"
fi
cd "$W/src"
got=$(git rev-parse HEAD)
[ "$got" = "$sha" ] || { echo "PoCL $tag 是 $got，钉的是 $sha"; exit 1; }
git checkout -q -- .
rm -rf lib/CL/devices/tgpu
git apply "$here/register.patch"
cp -r "$here/tgpu" lib/CL/devices/tgpu
cmake -S . -B "$W/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
  -DENABLE_LLVM=OFF -DENABLE_HOST_CPU_DEVICES=OFF -DENABLE_TGPU_DEVICE=ON -DENABLE_ICD=OFF \
  -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_POCLCC=OFF -DENABLE_HWLOC=OFF > "$W/cmake.log" 2>&1
cmake --build "$W/build" > "$W/build.log"
cmake --install "$W/build" > /dev/null
mkdir -p "$P/bin"
cc -O2 -Wall -Werror -I "$W/src/include" "$here/clrun.c" -L "$P/lib" -Wl,-rpath,"$P/lib" -lOpenCL -o "$P/bin/clrun"
echo "PoCL $tag 带 tgpu 装在 $P"
