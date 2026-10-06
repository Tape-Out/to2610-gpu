# to2610-gpu

A small parallel-compute chip for the ECOS 2610 shuttle: [tiny-gpu](https://github.com/Tape-Out/tiny-gpu) with four cores of four threads, its program and data memories on chip, loaded and started by a host over SPI.

![maturity](https://img.shields.io/badge/maturity-simulated-yellow) ![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0%20OR%20MulanPSL--2.0-blue)

Sixteen threads run the same kernel at once, each finding its own data from its thread number: the computing model of a GPU at its smallest. There is no processor on the chip. A host writes the kernel and the data through an SPI port that speaks the protocol of [`spis`](https://github.com/Tape-Out/spis), starts it, waits for the `done` pin and reads the data memory back.

tiny-gpu is taken unmodified through sv2v, as upstream builds it, and four inputs that upstream leaves unconnected on the program memory controller are tied to zero in the converted file (`htest/tieoff.py`). This repository adds the two memories and the management port in `hwsrc/`.

The board wiring, the register map, the chip tests and the limits are in [`docs/流片说明.md`](docs/流片说明.md); the tape-out report is generated from that file.

## Running a kernel

```console
$ python3 sw/gpu.py emu sw/kernels/shade.asm --show 8                       # the software model alone
$ python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 ident                         # 4 cores, 4 threads a block
$ python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 run sw/kernels/shade.asm --show 8
```

`sw/gpu.py` holds an assembler for tiny-gpu kernels, a software model that follows what the hardware actually does, and the SPI operations. `run` compares what the chip returns with the model.

| Kernel | Threads | Computes |
|:--:|:--:|:--:|
| `saxpy.asm` | 16 | y = 3x + y |
| `matmul.asm` | 16 | a 4 × 4 matrix product, one element a thread, with a loop |
| `shade.asm` | 64 | an 8 × 8 greyscale image, one pixel a thread |

One thing the model has to copy: the comparison upstream implements only tells equal from unequal, so `BRn` branches when the operands differ.

## Testing and tape-out

```console
$ ran test to2610-gpu                  # the chip tests, on the Verilog file that goes to the shuttle
$ ran asic to2610-gpu                  # to2610_gpu.v, ecc at 50 MHz, report.json
$ ran asic to2610-gpu --no-run         # only the Verilog file and ecc.toml
```

## License

This repository: 任选其一 [MIT](LICENSE-MIT) · [Apache 2.0](LICENSE-APACHE) · [木兰宽松许可证 第2版](LICENSE-MULAN).

`SPDX-License-Identifier: MIT OR Apache-2.0 OR MulanPSL-2.0`

tiny-gpu itself has no licence file upstream. It is fetched through the `tiny-gpu` repository as a submodule and neither changed nor copied here.

除非另行说明，你提交的贡献按上述三者同时授权，不附加其他条件。
