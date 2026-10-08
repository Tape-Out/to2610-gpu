#!/usr/bin/env bash
# OpenCL 那一层的埋错：tgpu 设备里改掉一处，重编 PoCL、在同一份交付的 .v 上再跑 htest/opencl.sh，每一处都要红；原样要过。
# 用法：opencl-faults.sh <输出目录>。DRY=1 只核对每一处都改得到 tgpu.c，不跑测试
set -u
cd "$(dirname "$0")/.."
O=$(realpath -m "$1")
rm -rf "$O"
mkdir -p "$O"

# 名字~sed 的写法：运算的源寄存器、第二个输入的位置、分块大小不按每块线程数取整、只装前一半数据存储、不读回、线程号算错
MUTS='srcreg~s/p\[n++\] = ALU (op, R4, R4, R5);/p[n++] = ALU (op, R4, R4, R4);/
second~s/elementwise (p, op, 0, k4, two ? 2 \* k4 : k4)/elementwise (p, op, 0, k4 + 1, two ? 2 * k4 : k4)/
tile~s/size_t tile = MEM \/ (two ? 3 : 2) \/ d->tpb \* d->tpb;/size_t tile = MEM \/ (two ? 3 : 2);/
half~s/launch (d, prog, np, mem, MEM, n);/launch (d, prog, np, mem, MEM \/ 2, n);/
noread~s/return err ? err : fetch (d, 0, buf, m);/return err;/
tid~s/p\[n++\] = ALU (ADD, R0, R0, TID);/p[n++] = ALU (ADD, R0, R0, BID);/'

bad=0
count=0
while IFS='~' read -r name expr; do
  m=$O/src-$name
  cp -r sw/pocl/tgpu "$m"
  sed -i -e "$expr" "$m/tgpu.c"
  if cmp -s sw/pocl/tgpu/tgpu.c "$m/tgpu.c"; then
    echo "$name 没埋上"
    bad=1
  fi
done <<< "$MUTS"
[ $bad = 0 ] || exit 1
[ -n "${DRY:-}" ] && { echo "六处都改得到"; exit 0; }

A=$O/asic
$XIRANG asic to2610-gpu --no-run -o "$A" > "$O/asic.log" 2>&1 || { tail -n 5 "$O/asic.log"; exit 1; }
if ! CHIP_ASIC=$A TGPU_SRC=$PWD/sw/pocl/tgpu bash htest/opencl.sh "$O/clean" > "$O/clean.log" 2>&1; then
  tail -n 20 "$O/clean.log"
  echo "原样没过"
  exit 1
fi
while IFS='~' read -r name expr; do
  count=$((count + 1))
  if CHIP_ASIC=$A TGPU_SRC=$O/src-$name bash htest/opencl.sh "$O/$name" > "$O/$name.log" 2>&1; then
    echo "$name 埋了错还过了"
    bad=1
  else
    echo "$name 红了：$(grep -a 'FAIL\|错 [1-9]\|退出码' "$O/$name.log" | head -n 2 | tr '\n' ' ' | cut -c1-160)"
  fi
done <<< "$MUTS"
[ $bad = 0 ] || { echo "有埋错没被抓到"; exit 1; }
echo "OpenCL 那一层：原样过，$count 处埋错都红"
