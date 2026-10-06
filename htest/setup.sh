#!/usr/bin/env bash
# 拼源码树：tiny-gpu 的源码照上游的做法过一遍 sv2v（它有 IEEE 1800 不收的写法，上游自己也是先转再用），
# 再补上上游例化时漏接的四个输入（见 tieoff.py）。
# 用法：setup.sh <tiny-gpu 仓>
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(realpath "$1")/third_party/tiny-gpu/src
[ -f "$T/gpu.sv" ] || { echo "找不到 $T/gpu.sv：tiny-gpu 仓的子模块取了吗"; exit 1; }
mkdir -p build/src
sv2v -w build/src/gpu.sv2v.v "$T"/*.sv
python3 htest/tieoff.py build/src/gpu.sv2v.v
{ echo '`timescale 1ns/1ns'; cat build/src/gpu.sv2v.v; } > build/src/gpu.v
rm build/src/gpu.sv2v.v
