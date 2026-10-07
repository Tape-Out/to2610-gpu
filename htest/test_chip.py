"""to2610-gpu 的整片测试，跑在交付的那份 .v 上，经五口顶层进出。

每个用例先复位。装载、起跑、读回走 sw/ 里上板用的同一份代码；结果既与软件模型比，也与各自另写的算式比。
"""
import os
import pathlib
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

import bench as B

SW = pathlib.Path(__file__).resolve().parent.parent / "sw"
sys.path.insert(0, str(SW))
import gpu as G  # noqa: E402
import spis  # noqa: E402


async def up(dut):
    c = B.Chip(dut, os.environ["CHIP_REPORT"])
    cocotb.start_soon(Clock(dut.clock, 1000 / c.mhz, units="ns").start())
    dut.reset.value = 1
    spi = B.Spi(c)
    cocotb.start_soon(c.run())
    await c.cycles(20)
    dut.reset.value = 0
    await c.cycles(10)
    return c, spi


def kernel(name: str) -> G.Kernel:
    return G.assemble((SW / "kernels" / f"{name}.asm").read_text(encoding="utf-8"))


async def run(c, spi, k: G.Kernel, limit: int = 200_000) -> tuple[list[int], int]:
    """装载、起跑，在 done 脚上等结束，读回整块数据存储与拍数。"""
    done, busy = c.bit("done[0]", "out"), c.bit("busy[0]", "out")
    await spi.do(G.load(k))
    await spi.do(G.start())
    n = 0
    while not (c.read()[0] >> done) & 1:
        await c.cycles(16)
        n += 16
        assert n < limit, "内核没有结束"
    o, e = c.read()
    assert (e >> done) & 1 and (e >> busy) & 1, "done 与 busy 两根脚要一直驱动"
    assert not (o >> busy) & 1, "结束之后 busy 要落下"
    assert await spi.do(G.done()) == 1
    return await spi.do(G.memory()), await spi.do(G.cycles())


@cocotb.test()
async def management(dut):
    """标识、配置、状态；两块存储写满随机数再整块读回；没有东西的地址回错并记在状态字里。"""
    c, spi = await up(dut)
    assert await spi.do(spis.ident()) == b"SPIS"
    assert await spi.do(G.config()) == (4, 4)
    assert await spi.do(spis.rd1(G.BASE + G.STATUS)) == 0
    rng = random.Random(2610)
    prog = [rng.getrandbits(16) for _ in range(256)]
    data = [rng.getrandbits(8) for _ in range(256)]
    await spi.do(spis.wr(G.BASE + G.PROG, *prog))
    await spi.do(spis.wr(G.BASE + G.DATA, *data))
    assert await spi.do(spis.rd(G.BASE + G.PROG, 256)) == prog
    assert await spi.do(spis.rd(G.BASE + G.DATA, 256)) == data
    # 高位写不进去：程序字 16 位，数据 8 位
    await spi.do(spis.wr(G.BASE + G.PROG + 4 * 7, 0xFFFF_1234))
    await spi.do(spis.wr(G.BASE + G.DATA + 4 * 7, 0xFFFF_FF56))
    assert await spi.do(spis.rd1(G.BASE + G.PROG + 4 * 7)) == 0x1234
    assert await spi.do(spis.rd1(G.BASE + G.DATA + 4 * 7)) == 0x56
    assert await spi.do(spis.rd(G.BASE + G.PROG + 4 * 6, 3)) == [prog[6], 0x1234, prog[8]]
    assert await spi.do(spis.status()) == 0
    for bad in (0x2000_0000, G.BASE + 0x024, G.BASE + 0xC00):
        await spi.do(spis.rd1(bad))
        assert await spi.do(spis.status()) == 1, hex(bad)
        assert await spi.do(spis.status()) == 0


@cocotb.test()
async def saxpy(dut):
    """y = 3x + y，十六个线程。跑两遍：第二遍在第一遍的结果上再加一次，证明能重新起跑。"""
    c, spi = await up(dut)
    k = kernel("saxpy")
    x, y = k.data[:16], k.data[16:32]
    got, n = await run(c, spi, k)
    want = [(3 * a + b) & 0xFF for a, b in zip(x, y)]
    assert got[16:32] == want, got[16:32]
    assert got[:16] == x
    assert got == G.emulate(k)
    assert 50 < n < 20_000, n
    dut._log.info("saxpy：十六个线程 %d 拍", n)
    # 不重新装数据，只收手再起跑
    await spi.do(spis.wr(G.BASE + G.CTRL, 0))
    await spi.do(G.start())
    done = c.bit("done[0]", "out")
    await c.cycles(8)
    assert not (c.read()[0] >> done) & 1, "重新起跑时 done 要先落下"
    while not (c.read()[0] >> done) & 1:
        await c.cycles(16)
    again = await spi.do(G.memory(32))
    assert again[16:32] == [(3 * a + b) & 0xFF for a, b in zip(x, want)], again[16:32]


@cocotb.test()
async def matmul(dut):
    """4 × 4 矩阵乘，每个线程里有循环与条件跳转。"""
    c, spi = await up(dut)
    k = kernel("matmul")
    a, b = k.data[:16], k.data[16:32]
    got, n = await run(c, spi, k)
    want = [sum(a[4 * r + i] * b[4 * i + col] for i in range(4)) & 0xFF for r in range(4) for col in range(4)]
    assert got[32:48] == want, got[32:48]
    assert got == G.emulate(k)
    dut._log.info("matmul：十六个线程 %d 拍", n)


@cocotb.test()
async def shade(dut):
    """8 × 8 的图，六十四个线程一人一个像素：十六个块轮流占四个核。"""
    c, spi = await up(dut)
    k = kernel("shade")
    got, n = await run(c, spi, k)
    want = [(((x - 4) ** 2 + (y - 4) ** 2) * 7) & 0xFF for y in range(8) for x in range(8)]
    assert got[64:128] == want, got[64:128]
    assert got == G.emulate(k)
    dut._log.info("shade：六十四个线程 %d 拍\n%s", n, G.show(got[64:128], 8))


@cocotb.test()
async def mlp(dut):
    """批量推理：两层感知机判 2 × 2 图样里亮点个数的奇偶，十六种图样一个线程一种，权在数据存储里。"""
    c, spi = await up(dut)
    k = kernel("mlp")
    x, w1, b1, w2 = k.data[:64], k.data[64:80], k.data[80:84], k.data[84:88]

    def relu(v: int) -> int:
        v &= 0xFF
        return 0 if v >= 128 else v

    want = []
    for i in range(16):
        h = [relu(b1[j] + sum(w1[4 * j + q] * x[4 * i + q] for q in range(4))) for j in range(4)]
        want.append(sum(a * b for a, b in zip(w2, h)) & 0xFF)
    assert want == [bin(i).count("1") & 1 for i in range(16)], "这组权算的不是奇偶"
    got, n = await run(c, spi, k)
    assert got[96:112] == want, got[96:112]
    assert got == G.emulate(k)
    dut._log.info("mlp：十六个线程 %d 拍", n)


@cocotb.test()
async def one_block(dut):
    """线程数只够一个块：只有一个核在跑，别的核不该动数据存储。"""
    c, spi = await up(dut)
    k = kernel("saxpy")
    k.threads = 4
    got, _ = await run(c, spi, k)
    want = [(3 * a + b) & 0xFF for a, b in zip(k.data[:4], k.data[16:20])] + k.data[20:32]
    assert got[16:32] == want, got[16:32]
    assert got == G.emulate(k)


# 上游 test/test_matadd.py 与 test/test_matmul.py 里的程序与数据，机器码逐字照搬：
# 封装之后，上游自己的两个测试也要在这颗片上过
UP_MATADD = G.Kernel(
    [
        0b0101000011011110,  # MUL R0, %blockIdx, %blockDim
        0b0011000000001111,  # ADD R0, R0, %threadIdx
        0b1001000100000000,  # CONST R1, #0
        0b1001001000001000,  # CONST R2, #8
        0b1001001100010000,  # CONST R3, #16
        0b0011010000010000,  # ADD R4, R1, R0
        0b0111010001000000,  # LDR R4, R4
        0b0011010100100000,  # ADD R5, R2, R0
        0b0111010101010000,  # LDR R5, R5
        0b0011011001000101,  # ADD R6, R4, R5
        0b0011011100110000,  # ADD R7, R3, R0
        0b1000000001110110,  # STR R7, R6
        0b1111000000000000,  # RET
    ],
    [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
    8,
)
UP_MATMUL = G.Kernel(
    [
        0b0101000011011110,  # MUL R0, %blockIdx, %blockDim
        0b0011000000001111,  # ADD R0, R0, %threadIdx
        0b1001000100000001,  # CONST R1, #1
        0b1001001000000010,  # CONST R2, #2
        0b1001001100000000,  # CONST R3, #0
        0b1001010000000100,  # CONST R4, #4
        0b1001010100001000,  # CONST R5, #8
        0b0110011000000010,  # DIV R6, R0, R2
        0b0101011101100010,  # MUL R7, R6, R2
        0b0100011100000111,  # SUB R7, R0, R7
        0b1001100000000000,  # CONST R8, #0
        0b1001100100000000,  # CONST R9, #0
        0b0101101001100010,  # LOOP: MUL R10, R6, R2
        0b0011101010101001,  # ADD R10, R10, R9
        0b0011101010100011,  # ADD R10, R10, R3
        0b0111101010100000,  # LDR R10, R10
        0b0101101110010010,  # MUL R11, R9, R2
        0b0011101110110111,  # ADD R11, R11, R7
        0b0011101110110100,  # ADD R11, R11, R4
        0b0111101110110000,  # LDR R11, R11
        0b0101110010101011,  # MUL R12, R10, R11
        0b0011100010001100,  # ADD R8, R8, R12
        0b0011100110010001,  # ADD R9, R9, R1
        0b0010000010010010,  # CMP R9, R2
        0b0001100000001100,  # BRn LOOP
        0b0011100101010000,  # ADD R9, R5, R0
        0b1000000010011000,  # STR R9, R8
        0b1111000000000000,  # RET
    ],
    [1, 2, 3, 4, 1, 2, 3, 4],
    4,
)


@cocotb.test()
async def upstream_matadd(dut):
    """上游的矩阵加法测试：八个线程各加一对数，判据照上游的写法。"""
    c, spi = await up(dut)
    k = UP_MATADD
    got, n = await run(c, spi, k)
    assert got[16:24] == [a + b for a, b in zip(k.data[0:8], k.data[8:16])], got[16:24]
    assert got == G.emulate(k)
    dut._log.info("上游 matadd：八个线程 %d 拍", n)


@cocotb.test()
async def upstream_matmul(dut):
    """上游的矩阵乘法测试：2 × 2，四个线程各算一个元素，判据照上游的写法。"""
    c, spi = await up(dut)
    k = UP_MATMUL
    got, n = await run(c, spi, k)
    a, b = k.data[0:4], k.data[4:8]
    want = [sum(a[2 * (i // 2) + j] * b[2 * j + i % 2] for j in range(2)) for i in range(4)]
    assert want == [7, 10, 15, 22]
    assert got[8:12] == want, got[8:12]
    assert got == G.emulate(k)
    dut._log.info("上游 matmul：四个线程 %d 拍", n)


def gray(p: int) -> int:
    return (p >> 5) << 5 | (p >> 5) << 2 | p >> 6


class Screen:
    """视频那十二根脚：逐拍读，或按步长粗看。颜色按 RGB332 拼回一个字节。"""

    def __init__(self, c, dut):
        self.c, self.dut = c, dut
        self.hs, self.vs, self.de, self.pk = (c.bit(f"vga_{n}[0]", "out") for n in ("hs", "vs", "de", "pclk"))
        self.col = [c.bit(f"vga_{n}[{i}]", "out") for n, k in (("b", 2), ("g", 3), ("r", 3)) for i in range(k)]

    def pins(self) -> list[int]:
        return [self.hs, self.vs, self.de, self.pk, *self.col]

    def pix(self, o: int) -> int:
        return sum(((o >> b) & 1) << i for i, b in enumerate(self.col))

    def on(self, o: int, b: int) -> bool:
        return bool((o >> b) & 1)

    async def until(self, f, step: int = 16, limit: int = 2_000_000) -> int:
        n = 0
        while not f(self.c.read()[0]):
            await ClockCycles(self.dut.clock, step)
            n += step
            assert n < limit, "等不到"
        return n

    async def rows(self, n: int):
        """等场同步过去，跳到第 0 行之前，逐拍取像素钟上升沿那一拍的颜色，收 n 行；顺带记行同步每次落下与抬起的拍。"""
        await self.until(lambda o: not self.on(o, self.vs))
        await self.until(lambda o: self.on(o, self.vs))
        await ClockCycles(self.dut.clock, 33 * 1600 - 400)
        got, line, edges = [], [], []
        h, k, t = True, False, 0
        while len(got) < n:
            await ClockCycles(self.dut.clock, 1)
            t += 1
            o = self.c.read()[0]
            if self.on(o, self.hs) != h:
                h = not h
                edges.append((t, h))
            if self.on(o, self.pk) and not k:
                if self.on(o, self.de):
                    line.append(self.pix(o))
                elif line:
                    got.append(line)
                    line = []
            k = self.on(o, self.pk)
            assert t < (n + 2) * 1600, (len(got), len(line))
        return got, edges


@cocotb.test()
async def video(dut):
    """VGA 与 HDMI 口：关着时同步不动；行 1600 拍、行同步 192 拍、场 525 行、场同步两行、每场 480 行 DE；
    数据存储里的图按放大倍数出现在每行 DE 的开头，灰度与 RGB332 两种；扫着屏照样能算。"""
    c, spi = await up(dut)
    s = Screen(c, dut)
    _, e = c.read()
    assert all(s.on(e, b) for b in s.pins()), "视频脚要一直驱动"
    for _ in range(2000):
        await ClockCycles(dut.clock, 1)
        o = c.read()[0]
        assert s.on(o, s.hs) and s.on(o, s.vs) and not s.on(o, s.de) and s.pix(o) == 0, "关着时同步停在高、DE 与颜色是 0"

    await spi.do(G.video())
    k = kernel("shade")
    got, _ = await run(c, spi, k)
    assert got == G.emulate(k), "扫屏不能碰计算"
    img = got[64:128]
    await spi.do(G.video(64, 8, 8))
    assert await spi.do(spis.rd(G.BASE + G.VCTRL, 3)) == [1, 64 | 8 << 8 | 8 << 16, 80 | 60 << 16]
    await spi.do(spis.wr(G.BASE + G.VSCALE, 3 | 2 << 16))

    await s.until(lambda o: s.on(o, s.vs))
    await s.until(lambda o: not s.on(o, s.vs))
    n = await s.until(lambda o: s.on(o, s.vs))
    assert abs(n - 2 * 1600) <= 16, n
    lines, last = 0, False
    while True:
        await ClockCycles(dut.clock, 16)
        n += 16
        o = c.read()[0]
        lines += s.on(o, s.de) and not last
        last = s.on(o, s.de)
        if not s.on(o, s.vs):
            break
        assert n < 1_000_000
    assert abs(n - 525 * 1600) <= 16, n
    assert lines == 480, lines

    for rgb in (False, True):
        if rgb:
            await spi.do(spis.wr(G.BASE + G.VCTRL, 3))
        rows, edges = await s.rows(18)
        for y, row in enumerate(rows):
            want = [img[(y // 2) * 8 + x // 3] if y < 16 and x < 24 else 0 for x in range(640)]
            assert row == (want if rgb else [gray(p) for p in want]), (rgb, y, row[:26])
        falls = [t for t, h in edges if not h]
        assert [b - a for a, b in zip(falls, falls[1:])] == [1600] * (len(falls) - 1) and len(falls) > 10, falls
        widths = [b[0] - a[0] for a, b in zip(edges, edges[1:]) if not a[1]]
        assert widths and all(w == 192 for w in widths), widths
    dut._log.info("视频：每行 1600 拍、行同步 192 拍、每场 525 行 480 行 DE；8 × 8 的图按 3 × 2 放大，灰度与 RGB332 都对")


@cocotb.test()
async def raster(dut):
    """渲染管线：主机把三角形换成边函数（顶点与建立），GPU 一个线程一个像素做光栅化与着色。画两个有重叠的三角形，
    后画的盖住先画的；帧缓冲与另写的判定、与软件模型逐像素相同；内核里带的默认三角形就是 triangle 算出来的那一个。"""
    c, spi = await up(dut)
    k = kernel("raster")
    tris = [((1, 1), (6, 2), (3, 6), 200), ((0, 7), (7, 7), (4, 0), 90)]
    ts = [G.triangle(*t) for t in tris]
    assert k.data[:10] == ts[0], k.data[:10]
    await spi.do(G.load(G.Kernel(k.program, [0] * 64, k.threads)))
    await spi.do(G.draw(ts))
    fb = (await spi.do(G.memory()))[64:128]
    want = [0] * 64
    for (a, b, cc, col), t in zip(tris, ts):
        for i in range(64):
            if G.inside(t, i % 8, i // 8):
                want[i] = col
    assert fb == want, G.show(fb, 8)
    assert 0 < sum(v == 200 for v in fb) < sum(v == 90 for v in fb), "两个都要看得见，后画的那个更大"
    mem = [0] * 256
    for t in ts:
        mem = G.emulate(G.Kernel(k.program, t + mem[10:], k.threads))
    assert mem[64:128] == fb
    dut._log.info("raster：两个三角形\n%s", G.show(fb, 8))
