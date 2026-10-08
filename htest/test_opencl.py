"""OpenCL 那一层的整片测试，跑在交付的那份 .v 上。

clrun 只调标准的 OpenCL 接口，经 PoCL 的 tgpu 设备、TCP（帧格式见 sw/spis.py 的 frame）进到这里，
由测试台的 SPI 主机在管理口上收发。内建内核与主机算的逐个比，.tgk 与软件模型比。
"""
import os
import pathlib
import select
import socket
import subprocess
import tempfile

import cocotb
from test_chip import SW, G, up

CLRUN = os.environ["CLRUN"]


async def clrun(c, spi, *args: str, limit: int = 3_000_000) -> str:
    """起一个 clrun，把它在 TCP 上发来的每一帧经管理口收发，直到它退出；返回它的输出，退出码不是 0 就不过。"""
    srv = socket.create_server(("127.0.0.1", 0))
    srv.setblocking(False)
    env = os.environ | {"POCL_DEVICES": "tgpu",
                        "POCL_TGPU0_PARAMETERS": f"tcp:127.0.0.1:{srv.getsockname()[1]}"}
    p = subprocess.Popen([CLRUN, *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    conn, buf, n = None, b"", 0
    try:
        while True:
            if conn is None and select.select([srv], [], [], 0)[0]:
                conn, _ = srv.accept()
            if conn is not None and select.select([conn], [], [], 0)[0]:
                got = conn.recv(1 << 16)
                buf += got
                while len(buf) >= 4 and len(buf) >= 4 + (k := int.from_bytes(buf[:4], "big")):
                    tx, buf = buf[4:4 + k], buf[4 + k:]
                    conn.sendall(await spi.xfer(tx))
                if not got:
                    conn.close()
                    conn = None
            if conn is None and p.poll() is not None:
                break
            await c.cycles(64)
            n += 64
            assert n < limit, f"clrun {' '.join(args)} 在 {limit} 拍里没跑完"
    finally:
        if p.poll() is None:
            p.kill()
        srv.close()
    out = p.stdout.read()
    c.dut._log.info("clrun %s：\n%s", " ".join(args), out)
    assert p.returncode == 0, f"clrun 退出码 {p.returncode}"
    return out


@cocotb.test()
async def builtin(dut):
    """设备信息读得出；PoCL 的三个内建内核各 100 个元素（一块 84 个放不下，分两块，第二块凑成 4 的倍数）。"""
    c, spi = await up(dut)
    out = await clrun(c, spi, "info")
    assert "to2610-gpu" in out and "计算单元 4，工作组最大 4" in out, out
    out = await clrun(c, spi, "builtin", "100")
    for name in ("pocl.add.i8", "pocl.mul.i8", "pocl.copy.i8"):
        assert f"{name} 100 个，错 0 个" in out, out


async def tgk(dut, k: G.Kernel):
    """写成 .tgk 经 clCreateProgramWithBinary 装进去，数据存储的初值是它的 .data，跑完整块与软件模型比。"""
    c, spi = await up(dut)
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "k.tgk"
        f.write_bytes(G.tgk(k))
        out = await clrun(c, spi, "tgk", str(f))
    got = [int(x) for x in out.split()[-256:]]
    assert got == G.emulate(k), got


def kernel(name: str) -> G.Kernel:
    return G.assemble((SW / "kernels" / f"{name}.asm").read_text(encoding="utf-8"))


@cocotb.test()
async def shade(dut):
    """渲染：8 × 8 的图一个线程一个像素。"""
    await tgk(dut, kernel("shade"))


@cocotb.test()
async def mlp(dut):
    """推理：4-4-1 的两层感知机，十六种图样一个线程一种。"""
    await tgk(dut, kernel("mlp"))


@cocotb.test()
async def raster(dut):
    """渲染管线里芯片那一段：三角形的光栅化，三条边函数逐像素判里外再着色。"""
    await tgk(dut, kernel("raster"))


@cocotb.test()
async def whole(dut):
    """缓冲区与整块数据存储一一对应：输入放在后一半，64 个线程把第 128 至 191 字节搬到 192 至 255。"""
    await tgk(dut, G.assemble(f"""
        .threads 64
        .data {" ".join(str(i) for i in range(192))}
        MUL R0, %blockIdx, %blockDim
        ADD R0, R0, %threadIdx
        CONST R1, #128
        CONST R3, #192
        ADD R4, R1, R0
        LDR R4, R4
        ADD R7, R3, R0
        STR R7, R4
        RET
    """))
