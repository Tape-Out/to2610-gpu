"""OpenGL 那一层的整片测试，跑在交付的那份 .v 上。

同一组三角形经 gldraw 走 OpenGL 的图形管线（Mesa 的 llvmpipe，EGL 无表面平台），帧缓冲与片子跑 raster.asm 的逐像素比；
再随机画三百个，与软件模型比（软件模型与片子在 test_chip 的 raster 里比过）。像素中心正落在边上的都算在内。
"""
import os
import pathlib
import random
import subprocess

import cocotb
from test_chip import G, kernel, up

GL = pathlib.Path(os.environ["GLBUILD"])


def draw(tris) -> list[int]:
    """gpu.py 的顶点是像素中心，换到管线的坐标要加 0.5。"""
    text = "".join(" ".join(f"{x + 0.5} {y + 0.5}" for x, y in pts) + f" {col}\n" for *pts, col in tris)
    r = subprocess.run([GL / "gldraw"], input=text.encode(), capture_output=True, check=True,
                       env=os.environ | {"EGL_PLATFORM": "surfaceless"})
    assert "llvmpipe" in r.stderr.decode(), f"跑在了 {r.stderr.decode().strip()} 上，不是 llvmpipe"
    return list(r.stdout)


@cocotb.test()
async def graphics(dut):
    c, spi = await up(dut)
    k = kernel("raster")
    tris = [((1, 1), (6, 2), (3, 6), 200), ((0, 7), (7, 7), (4, 0), 90)]
    await spi.do(G.load(G.Kernel(k.program, [0] * 64, k.threads)))
    await spi.do(G.draw([G.triangle(*t) for t in tris]))
    chip = (await spi.do(G.memory()))[64:128]
    got = draw(tris)
    assert got == chip, "片子\n" + G.show(chip, 8) + "\nllvmpipe\n" + G.show(got, 8)
    rng, n = random.Random(2611), 0
    while n < 300:
        pts = [(rng.randrange(8), rng.randrange(8)) for _ in range(3)]
        try:
            t = G.triangle(*pts, 1)
        except ValueError:
            continue
        n += 1
        want = [int(G.inside(t, i % 8, i // 8)) for i in range(64)]
        assert draw([(*pts, 1)]) == want, f"{pts}：llvmpipe 与软件模型不同"
    dut._log.info("graphics：两个三角形与片子逐像素相同，三百个随机三角形与软件模型相同\n%s", G.show(got, 8))
