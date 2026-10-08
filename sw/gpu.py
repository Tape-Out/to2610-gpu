"""to2610-gpu 的主机端：汇编 tiny-gpu 的内核、软件模型、经 SPI 装载与起跑、读回结果。

    python3 sw/gpu.py emu sw/kernels/shade.asm --show 8            只在本机跑软件模型
    python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 run sw/kernels/shade.asm --show 8
    python3 sw/gpu.py --spidev 0.0 ident
    python3 sw/gpu.py --model run sw/kernels/saxpy.asm                 不接板子，对着照寄存器写的软件模型
    python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 serve 2610            挂到 TCP 上给 PoCL 的 tgpu 设备
    python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 video --at 64 --size 8x8   数据存储的第 64 字节起当 8 × 8 的图扫到屏上

内核的写法照 tiny-gpu 的说明：`.threads N` 给线程数，`.data …` 依次往数据存储里放初值，其后是指令。
每个线程从头跑到 RET；`%blockIdx`、`%blockDim`、`%threadIdx` 三个只读寄存器告诉它自己是谁。
地址照本仓 hwsrc/tgpu_soc.v 的文件头。
"""
import argparse
import pathlib
import re
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import spis  # noqa: E402

BASE = 0x1000_0000
CTRL, THREADS, STATUS, CYCLES, ID, CONFIG = 0x000, 0x004, 0x008, 0x00C, 0x010, 0x014
VCTRL, VFMT, VSCALE = 0x018, 0x01C, 0x020
PROG, DATA = 0x400, 0x800
TGPU = 0x5447_5055
WRITABLE = {CTRL: 0x1, THREADS: 0xFF, VCTRL: 0x3, VFMT: 0xFF_FFFF, VSCALE: 0x3FF_03FF}

OPS = {"NOP": 0x0, "CMP": 0x2, "ADD": 0x3, "SUB": 0x4, "MUL": 0x5, "DIV": 0x6,
       "LDR": 0x7, "STR": 0x8, "CONST": 0x9, "RET": 0xF}
REGS = {f"R{i}": i for i in range(13)} | {"%BLOCKIDX": 13, "%BLOCKDIM": 14, "%THREADIDX": 15}


class Kernel:
    def __init__(self, program: list[int], data: list[int], threads: int):
        self.program, self.data, self.threads = program, data, threads


def assemble(text: str) -> Kernel:
    """两遍：先收标号，再出码。指令 16 位：高 4 位操作码，其后三个 4 位寄存器号，或一个寄存器号加 8 位立即数。"""
    lines: list[tuple[int, list[str]]] = []
    labels: dict[str, int] = {}
    data: list[int] = []
    threads = 0
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.split(";")[0].strip()
        if not s:
            continue
        if m := re.fullmatch(r"(\w+):", s):
            labels[m.group(1).upper()] = len(lines)
            continue
        tok = s.replace(",", " ").split()
        if tok[0] == ".threads":
            threads = int(tok[1], 0)
        elif tok[0] == ".data":
            data += [int(x, 0) & 0xFF for x in tok[1:]]
        else:
            lines.append((n, tok))

    def reg(n: int, t: str) -> int:
        if t.upper() not in REGS:
            raise ValueError(f"第 {n} 行：不认识的寄存器 {t}")
        return REGS[t.upper()]

    program = []
    for n, tok in lines:
        op, a = tok[0].upper(), tok[1:]
        if m := re.fullmatch(r"BR([NZP]+)", op):
            if a[0].upper() not in labels:
                raise ValueError(f"第 {n} 行：没有标号 {a[0]}")
            cond = sum({"N": 4, "Z": 2, "P": 1}[c] for c in set(m.group(1)))
            program.append(0x1 << 12 | cond << 9 | labels[a[0].upper()])
        elif op in ("NOP", "RET"):
            program.append(OPS[op] << 12)
        elif op == "CONST":
            program.append(OPS[op] << 12 | reg(n, a[0]) << 8 | int(a[1].lstrip("#"), 0) & 0xFF)
        elif op == "CMP":
            program.append(OPS[op] << 12 | reg(n, a[0]) << 4 | reg(n, a[1]))
        elif op == "LDR":
            program.append(OPS[op] << 12 | reg(n, a[0]) << 8 | reg(n, a[1]) << 4)
        elif op == "STR":
            program.append(OPS[op] << 12 | reg(n, a[0]) << 4 | reg(n, a[1]))
        elif op in OPS:
            program.append(OPS[op] << 12 | reg(n, a[0]) << 8 | reg(n, a[1]) << 4 | reg(n, a[2]))
        else:
            raise ValueError(f"第 {n} 行：不认识的指令 {tok[0]}")
    if len(program) > 256 or len(data) > 256:
        raise ValueError("程序与数据各只有 256 格")
    return Kernel(program, data, threads)


def emulate(k: Kernel, tpb: int = 4, steps: int = 100_000) -> list[int]:
    """软件模型，照硬件实际的行为写：寄存器与存储都是 8 位，算术回绕；
    CMP 只分得出相等与不等（上游的减法是无符号的，「小于」那一位永远是 0，「大于」那一位在不等时为 1），
    所以 BRn 的实际含义是「不等则跳」。线程一个接一个跑，内核里各线程写的地址不能重叠。"""
    mem = (k.data + [0] * 256)[:256]
    for i in range(k.threads):
        r = [0] * 13 + [i // tpb, tpb, i % tpb]
        pc = nzp = 0
        for _ in range(steps):
            w = k.program[pc]
            op, d, s, t, imm = w >> 12, (w >> 8) & 15, (w >> 4) & 15, w & 15, w & 0xFF
            pc += 1
            v = None
            if op == 0xF:
                break
            if op == 0x1:
                if nzp & (w >> 9) & 7:
                    pc = imm
            elif op == 0x2:
                nzp = 2 if r[s] == r[t] else 4
            elif op == 0x3:
                v = r[s] + r[t]
            elif op == 0x4:
                v = r[s] - r[t]
            elif op == 0x5:
                v = r[s] * r[t]
            elif op == 0x6:
                if r[t] == 0:
                    raise ZeroDivisionError(f"线程 {i} 在第 {pc - 1} 条指令上除以 0，硬件上结果不定")
                v = r[s] // r[t]
            elif op == 0x7:
                v = mem[r[s]]
            elif op == 0x8:
                mem[r[s]] = r[t]
            elif op == 0x9:
                v = imm
            if v is not None and d < 13:
                r[d] = v & 0xFF
        else:
            raise RuntimeError(f"线程 {i} 跑了 {steps} 步还没到 RET")
    return mem


def load(k: Kernel):
    """数据存储整块写一遍，`.data` 没给到的补 0：片上的存储上电后内容不定，上一个内核的结果也还留着。"""
    yield from spis.wr(BASE + CTRL, 0)
    yield from spis.wr(BASE + PROG, *k.program)
    yield from spis.wr(BASE + DATA, *(k.data + [0] * (256 - len(k.data))))
    yield from spis.wr(BASE + THREADS, k.threads)


def start():
    yield from spis.wr(BASE + CTRL, 1)


def done():
    return (yield from spis.rd1(BASE + STATUS)) & 1


def cycles():
    return (yield from spis.rd1(BASE + CYCLES))


def config():
    """(核数, 每块线程数)。先核对标识，不是这颗芯片就报错。"""
    ident, cfg = yield from spis.rd(BASE + ID, 2)
    if ident != TGPU:
        raise RuntimeError(f"标识读出 0x{ident:08x}，不是 to2610-gpu")
    return cfg >> 8 & 0xFF, cfg & 0xFF


def memory(n: int = 256):
    w = yield from spis.rd(BASE + DATA, n)
    return [x & 0xFF for x in w]


def triangle(a, b, c, color: int) -> list[int]:
    """把一个三角形换成 raster.asm 要的十个字节：三条边的 A、B、C（8 位回绕）与颜色。这是渲染管线的顶点与建立
    两步，留在主机上做。边函数在三角形里不为负；8 × 8 的每个像素上它都要落在 −128 至 127，否则 8 位取不对符号。"""
    pts = (a, b, c)
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if not area:
        raise ValueError(f"{pts} 三个点在一条线上")
    s = 1 if area > 0 else -1
    out = []
    for p, q in zip(pts, pts[1:] + pts[:1]):
        ea, eb = -(q[1] - p[1]) * s, (q[0] - p[0]) * s
        ec = -(ea * p[0] + eb * p[1])
        if any(not -128 <= ea * x + eb * y + ec <= 127 for y in range(8) for x in range(8)):
            raise ValueError(f"{pts} 的边函数超出 8 位：顶点放在 0 至 7 之内再试")
        out += [ea & 0xFF, eb & 0xFF, ec & 0xFF]
    return out + [color & 0xFF]


def inside(t, x: int, y: int) -> bool:
    """triangle 那十个字节在像素 (x, y) 上的判定，照硬件的写法：三条边都不为负。"""
    def sx(v):
        return v - 256 if v >= 128 else v
    return all(sx((t[3 * i] * x + t[3 * i + 1] * y + t[3 * i + 2]) & 0xFF) >= 0 for i in range(3))


def draw(tris: list[list[int]]):
    """在已经装好的 raster.asm 上一个接一个画：每个三角形写一次那十个字节、起跑、等算完。帧缓冲不清。"""
    for t in tris:
        yield from spis.wr(BASE + DATA, *t)
        yield from spis.wr(BASE + CTRL, 0)
        yield from start()
        while not (yield from done()):
            pass


def tgk(k: Kernel) -> bytes:
    """Linux 那一侧的 tgpu 读的文件：小端的 'TGK1'、线程数、程序条数、数据字节数、两个字节的空，再是程序与数据。"""
    head = b"TGK1" + struct.pack("<HHHH", k.threads, len(k.program), len(k.data), 0)
    return head + struct.pack(f"<{len(k.program)}H", *k.program) + bytes(k.data)


def video(at: int = 0, w: int = 16, h: int = 16, rgb: bool = False, on: bool = True):
    """把数据存储里从 at 起的 w × h 个字节扫到 VGA 与 HDMI 上，横竖各按整数倍放大到铺满 640 × 480。"""
    if not (0 < w <= 255 and 0 < h <= 255 and 0 <= at and at + w * h <= 256):
        raise ValueError(f"{w} × {h} 从 {at} 起放不进 256 字节的数据存储")
    yield from spis.wr(BASE + VFMT, at | w << 8 | h << 16, 640 // w | (480 // h) << 16)
    yield from spis.wr(BASE + VCTRL, int(on) | int(rgb) << 1)


class Model:
    """照 hwsrc/tgpu_soc.v 写的整颗芯片，说 spis 的协议。没有板子时本仓的工具与 PoCL 的 tgpu 设备都能对着它跑。
    起跑那一下就把内核跑完；线程数不是每块线程数的整数倍时与硬件一样永远不结束。拍数不模拟，读出来是 0。"""

    def __init__(self, cores: int = 4, tpb: int = 4):
        self.cores, self.tpb = cores, tpb
        self.prog, self.data = [0] * 256, [0] * 256
        self.reg = {CTRL: 0, THREADS: 0, VCTRL: 0, VFMT: 16 << 8 | 16 << 16, VSCALE: 40 | 30 << 16}
        self.done = self.err = 0

    def get(self, a: int) -> int:
        o = a - BASE
        if PROG <= o < PROG + 0x400 and o % 4 == 0:
            return self.prog[(o - PROG) // 4]
        if DATA <= o < DATA + 0x400 and o % 4 == 0:
            return self.data[(o - DATA) // 4]
        if o in self.reg:
            return self.reg[o]
        match o:
            case 0x008:
                return self.done
            case 0x00C:
                return 0
            case 0x010:
                return TGPU
            case 0x014:
                return self.cores << 8 | self.tpb
        self.err = 1
        return 0

    def put(self, a: int, w: int):
        o = a - BASE
        if PROG <= o < PROG + 0x400 and o % 4 == 0:
            self.prog[(o - PROG) // 4] = w & 0xFFFF
        elif DATA <= o < DATA + 0x400 and o % 4 == 0:
            self.data[(o - DATA) // 4] = w & 0xFF
        elif o in WRITABLE:
            self.reg[o] = w & WRITABLE[o]
            if o == CTRL:
                self.done = 0
                t = self.reg[THREADS]
                if w & 1 and t % self.tpb == 0:
                    self.data = emulate(Kernel(self.prog, self.data, t), self.tpb)
                    self.done = 1
        else:
            self.err = 1

    def xfer(self, tx: bytes) -> bytes:
        rx = bytearray(len(tx))
        a = int.from_bytes(tx[1:5], "big")
        match tx[0]:
            case 0x02:
                for i in range(5, len(tx) - 3, 4):
                    self.put(a + i - 5, int.from_bytes(tx[i:i + 4], "big"))
            case 0x03:
                rx[6:10] = self.get(a).to_bytes(4, "big")
            case 0x0B:
                for i in range(tx[5] + 1):
                    rx[7 + 4 * i:11 + 4 * i] = self.get(a + 4 * i).to_bytes(4, "big")
            case 0x05:
                rx[1], self.err = self.err, 0
            case 0x9F:
                rx[1:5] = b"SPIS"
        return bytes(rx)


def show(mem: list[int], width: int) -> str:
    return "\n".join(" ".join(f"{v:3}" for v in mem[i:i + width]) for i in range(0, len(mem), width))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="to2610-gpu 主机端")
    link = ap.add_mutually_exclusive_group()
    link.add_argument("--spidev", help="总线.片选，如 0.0")
    link.add_argument("--ftdi", help="pyftdi 的地址，如 ftdi://ftdi:232h/1")
    link.add_argument("--model", action="store_true", help="不接板子，对着照寄存器写的软件模型")
    ap.add_argument("--hz", type=int, default=2_000_000, help="SCK，不超过芯片主频的八分之一")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ident")
    p = sub.add_parser("serve", help="把这条 SPI 挂到 TCP 上，给 PoCL 的 tgpu 设备用（POCL_TGPU0_PARAMETERS=tcp:127.0.0.1:端口）")
    p.add_argument("port", type=int)
    p = sub.add_parser("draw", help="渲染：装上 raster.asm，在 8 × 8 的帧缓冲上一个接一个画三角形")
    p.add_argument("tris", nargs="+", metavar="x0,y0:x1,y1:x2,y2:颜色")
    p.add_argument("--video", action="store_true", help="画完扫到屏上")
    p = sub.add_parser("build", help="汇编成 Linux 上 tgpu 读的 .tgk")
    p.add_argument("kernel")
    p.add_argument("-o", "--out", required=True)
    p = sub.add_parser("video", help="把数据存储里的一块扫到屏上")
    p.add_argument("--at", type=int, default=0, help="起始字节")
    p.add_argument("--size", default="16x16", help="宽x高")
    p.add_argument("--rgb", action="store_true", help="每字节按 RGB332 解，默认是灰度")
    p.add_argument("--off", action="store_true", help="关掉，脚上只剩不动的同步")
    for name in ("emu", "run"):
        p = sub.add_parser(name)
        p.add_argument("kernel")
        p.add_argument("--show", type=int, default=16, help="每行打几个字节")
        p.add_argument("--rows", type=int, default=16, help="打几行")
    a = ap.parse_args(argv)
    if a.cmd == "emu":
        k = assemble(pathlib.Path(a.kernel).read_text(encoding="utf-8"))
        print(show(emulate(k)[:a.show * a.rows], a.show))
        return 0
    if a.cmd == "build":
        pathlib.Path(a.out).write_bytes(tgk(assemble(pathlib.Path(a.kernel).read_text(encoding="utf-8"))))
        return 0
    if not (a.spidev or a.ftdi or a.model):
        ap.error("要 --spidev、--ftdi 或 --model")
    if a.model:
        x = Model().xfer
    elif a.spidev:
        bus, _, dev = a.spidev.partition(".")
        x = spis.spidev(int(bus), int(dev or 0), a.hz)
    else:
        x = spis.ftdi(a.ftdi, a.hz)
    if a.cmd == "serve":
        spis.serve(x, a.port)
    run = lambda op: spis.run(op, x)  # noqa: E731
    if a.cmd == "draw":
        run(config())
        tris = []
        for s in a.tris:
            *v, col = s.split(":")
            tris.append(triangle(*(tuple(int(n) for n in p.split(",")) for p in v), int(col)))
        k = assemble((pathlib.Path(__file__).parent / "kernels" / "raster.asm").read_text(encoding="utf-8"))
        run(load(Kernel(k.program, [0] * 64, k.threads)))
        run(draw(tris))
        print(show(run(memory())[64:128], 8))
        if a.video:
            run(video(64, 8, 8))
        return 0
    if a.cmd == "ident":
        cores, tpb = run(config())
        print(f"to2610-gpu：{cores} 核，每块 {tpb} 个线程")
        return 0
    if a.cmd == "video":
        run(config())
        w, _, h = a.size.lower().partition("x")
        run(video(a.at, int(w), int(h), a.rgb, not a.off))
        return 0
    k = assemble(pathlib.Path(a.kernel).read_text(encoding="utf-8"))
    cores, tpb = run(config())
    if k.threads % tpb:
        sys.exit(f"线程数 {k.threads} 不是每块线程数 {tpb} 的整数倍，硬件上这个内核不会结束")
    run(load(k))
    run(start())
    while not run(done()):
        pass
    got = run(memory())
    print(f"{run(cycles())} 拍")
    print(show(got[:a.show * a.rows], a.show))
    if got != emulate(k, tpb):
        print("与软件模型不一致", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
