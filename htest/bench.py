"""整片测试台：一切都经五口顶层的 io_in / io_out / io_oe 进出，与板子上看到的一样。

每根信号在 payload 的哪一位，从 `ran asic` 写的 report.json 里查，不在这里另写一份：位表错了，测试就先错。
"""
import json
import pathlib

from cocotb.triggers import FallingEdge, RisingEdge


def _bits(v) -> int:
    return int("".join(c if c in "01" else "0" for c in v.binstr), 2)


class Chip:
    def __init__(self, dut, report: str):
        rep = json.loads(pathlib.Path(report).read_text(encoding="utf-8"))
        self.dut, self.mhz = dut, rep["mhz"]
        self.pin: dict[str, dict[str, int]] = {}
        for r in rep["pads"]["bits"]:
            for role in ("in", "out", "oe"):
                if r[role]:
                    self.pin.setdefault(r[role], {})[role] = r["bit"]
        self.inv = 0
        dut.io_in.value = 0

    def bit(self, sig: str, role: str) -> int:
        return self.pin[sig][role]

    def set(self, b: int, v: int) -> None:
        self.inv = (self.inv | (1 << b)) if v else (self.inv & ~(1 << b))

    def read(self) -> tuple[int, int]:
        """逐位读：复位前后有几位是 X，不能因此把整根向量当 0。"""
        return _bits(self.dut.io_out.value), _bits(self.dut.io_oe.value)

    async def run(self):
        """下降沿把攒下的输入送上去，各处只管改 inv。"""
        while True:
            await FallingEdge(self.dut.clock)
            self.dut.io_in.value = self.inv

    async def cycles(self, n: int):
        for _ in range(n):
            await RisingEdge(self.dut.clock)


class Spi:
    """SPI 主机：SCK 取主频的八分之一，是管理口的上限。跑的是 sw/ 里的同一份操作。"""

    HALF = 4

    def __init__(self, chip: Chip):
        self.c = chip
        self.sck = chip.bit("sck[0]", "in")
        self.cs = chip.bit("cs_n[0]", "in")
        self.mosi = chip.bit("mosi[0]", "in")
        self.miso = chip.bit("miso[0]", "out")
        assert self.miso == chip.bit("miso_oe[0]", "oe")
        chip.set(self.cs, 1)

    async def xfer(self, tx: bytes) -> bytes:
        c = self.c
        c.set(self.cs, 0)
        await c.cycles(6)
        rx = bytearray()
        for byte in tx:
            got = 0
            for i in range(7, -1, -1):
                c.set(self.mosi, (byte >> i) & 1)
                await c.cycles(self.HALF)
                c.set(self.sck, 1)
                await RisingEdge(c.dut.clock)
                o, e = c.read()
                assert (e >> self.miso) & 1, "cs_n 低着时 miso 要驱动"
                got = (got << 1) | ((o >> self.miso) & 1)
                await c.cycles(self.HALF - 1)
                c.set(self.sck, 0)
            rx.append(got)
        await c.cycles(8)
        c.set(self.cs, 1)
        await c.cycles(24)
        o, e = c.read()
        assert not (e >> self.miso) & 1, "cs_n 拉高之后 miso 要放开"
        return bytes(rx)

    async def do(self, op):
        try:
            tx = next(op)
            while True:
                tx = op.send(await self.xfer(tx))
        except StopIteration as done:
            return done.value
