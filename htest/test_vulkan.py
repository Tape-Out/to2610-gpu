"""Vulkan 那一层的整片测试，跑在交付的那份 .v 上，设备在流水线上与本机都是 lavapipe。

计算管线：同一份计算写成 GLSL 计算着色器（sw/vk/kernels），经 vkrun 跑，与片子跑汇编内核读回的整块数据存储逐字节比。
图形管线：同一组三角形经 vkdraw 画，帧缓冲与片子跑 raster.asm 的逐像素比。片子上的结果另与软件模型比过（test_chip）。
"""
import os
import pathlib
import random
import subprocess

import cocotb
from test_chip import G, kernel, run, up

VK = pathlib.Path(os.environ["VKBUILD"])


def vk(name: str, mem: list[int], threads: int) -> list[int]:
    r = subprocess.run([VK / "vkrun", VK / f"{name}_comp.spv", str(threads)], input=bytes(mem), capture_output=True, check=True)
    assert "llvmpipe" in r.stderr.decode(), f"跑在了 {r.stderr.decode().strip()} 上，不是 lavapipe"
    return list(r.stdout)


@cocotb.test()
async def compute(dut):
    """saxpy、matmul、shade、mlp 四个内核：片子与 lavapipe 的整块数据存储逐字节相同。
    初值以外的格子填 0x5A 再少起一个线程，结果就要不同：少算的那一格留着 0x5A，比较才分得出来。"""
    c, spi = await up(dut)
    for name in ("saxpy", "matmul", "shade", "mlp"):
        k = kernel(name)
        mem = (k.data + [0] * 256)[:256]
        chip, _ = await run(c, spi, k)
        got = vk(name, mem, k.threads)
        assert got == chip, f"{name}：lavapipe 与片子有 {sum(a != b for a, b in zip(got, chip))} 格不同"
        fill = (k.data + [0x5A] * 256)[:256]
        assert vk(name, fill, k.threads - 1) != vk(name, fill, k.threads), f"{name}：少一个线程也一样，比较没有意义"
        dut._log.info("%s：lavapipe 与片子逐字节相同", name)


@cocotb.test()
async def raster(dut):
    """光栅化：两个有重叠的三角形一个接一个画，片子与 lavapipe 的帧缓冲逐像素相同。"""
    c, spi = await up(dut)
    k = kernel("raster")
    ts = [G.triangle(*t) for t in (((1, 1), (6, 2), (3, 6), 200), ((0, 7), (7, 7), (4, 0), 90))]
    await spi.do(G.load(G.Kernel(k.program, [0] * 64, k.threads)))
    await spi.do(G.draw(ts))
    chip = await spi.do(G.memory())
    mem = [0] * 256
    for t in ts:
        mem = vk("raster", t + mem[10:], k.threads)
    assert mem[64:128] == chip[64:128], "片子\n" + G.show(chip[64:128], 8) + "\nlavapipe\n" + G.show(mem[64:128], 8)
    dut._log.info("raster：两个三角形，lavapipe 与片子逐像素相同\n%s", G.show(mem[64:128], 8))


def draw(tris) -> list[int]:
    """Vulkan 的图形管线画一组三角形（vkdraw，帧缓冲清成 0）。gpu.py 的顶点是像素中心，换到管线的坐标要加 0.5。"""
    text = "".join(" ".join(f"{x + 0.5} {y + 0.5}" for x, y in pts) + f" {col}\n" for *pts, col in tris)
    r = subprocess.run([VK / "vkdraw", VK / "tri_vert.spv", VK / "tri_frag.spv"], input=text.encode(), capture_output=True, check=True)
    assert "llvmpipe" in r.stderr.decode(), f"跑在了 {r.stderr.decode().strip()} 上，不是 lavapipe"
    return list(r.stdout)


@cocotb.test()
async def graphics(dut):
    """渲染管线：同两个三角形走 Vulkan 的图形管线（光栅化、片元着色、按图元次序写），帧缓冲与片子逐像素相同，
    公共边、像素中心正落在边上的都算在内。再随机画三百个三角形，与软件模型比（软件模型与片子在 raster 里比过）。"""
    c, spi = await up(dut)
    k = kernel("raster")
    tris = [((1, 1), (6, 2), (3, 6), 200), ((0, 7), (7, 7), (4, 0), 90)]
    await spi.do(G.load(G.Kernel(k.program, [0] * 64, k.threads)))
    await spi.do(G.draw([G.triangle(*t) for t in tris]))
    chip = (await spi.do(G.memory()))[64:128]
    got = draw(tris)
    assert got == chip, "片子\n" + G.show(chip, 8) + "\nlavapipe\n" + G.show(got, 8)
    rng, n = random.Random(2610), 0
    while n < 300:
        pts = [(rng.randrange(8), rng.randrange(8)) for _ in range(3)]
        try:
            t = G.triangle(*pts, 1)
        except ValueError:
            continue
        n += 1
        want = [int(G.inside(t, i % 8, i // 8)) for i in range(64)]
        assert draw([(*pts, 1)]) == want, f"{pts}：lavapipe 与软件模型不同"
    dut._log.info("graphics：两个三角形与片子逐像素相同，三百个随机三角形与软件模型相同\n%s", G.show(got, 8))
