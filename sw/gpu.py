"""to2610-gpu 的主机端：汇编 tiny-gpu 的内核、软件模型、经 SPI 装载与起跑、读回结果。

    python3 sw/gpu.py emu sw/kernels/shade.asm --show 8            只在本机跑软件模型
    python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 run sw/kernels/shade.asm --show 8
    python3 sw/gpu.py --spidev 0.0 ident

内核的写法照 tiny-gpu 的说明：`.threads N` 给线程数，`.data …` 依次往数据存储里放初值，其后是指令。
每个线程从头跑到 RET；`%blockIdx`、`%blockDim`、`%threadIdx` 三个只读寄存器告诉它自己是谁。
地址照本仓 hwsrc/tgpu_soc.v 的文件头。
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import spis  # noqa: E402

BASE = 0x1000_0000
CTRL, THREADS, STATUS, CYCLES, ID, CONFIG = 0x000, 0x004, 0x008, 0x00C, 0x010, 0x014
PROG, DATA = 0x400, 0x800
TGPU = 0x5447_5055

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


def show(mem: list[int], width: int) -> str:
    return "\n".join(" ".join(f"{v:3}" for v in mem[i:i + width]) for i in range(0, len(mem), width))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="to2610-gpu 主机端")
    link = ap.add_mutually_exclusive_group()
    link.add_argument("--spidev", help="总线.片选，如 0.0")
    link.add_argument("--ftdi", help="pyftdi 的地址，如 ftdi://ftdi:232h/1")
    ap.add_argument("--hz", type=int, default=2_000_000, help="SCK，不超过芯片主频的八分之一")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ident")
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
    if not (a.spidev or a.ftdi):
        ap.error("要 --spidev 或 --ftdi")
    if a.spidev:
        bus, _, dev = a.spidev.partition(".")
        x = spis.spidev(int(bus), int(dev or 0), a.hz)
    else:
        x = spis.ftdi(a.ftdi, a.hz)
    run = lambda op: spis.run(op, x)  # noqa: E731
    if a.cmd == "ident":
        cores, tpb = run(config())
        print(f"to2610-gpu：{cores} 核，每块 {tpb} 个线程")
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
