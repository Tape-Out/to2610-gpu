# to2610-gpu

A small parallel-compute chip for the ECOS 2610 shuttle: [tiny-gpu](https://github.com/Tape-Out/tiny-gpu) with four cores of four threads, its program and data memories on chip, loaded and started by a host over SPI.

![maturity](https://img.shields.io/badge/maturity-simulated-yellow) ![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0%20OR%20MulanPSL--2.0-blue)

Sixteen threads run the same kernel at once, each finding its own data from its thread number: the computing model of a GPU at its smallest. The same hardware shades an image, one pixel a thread, and runs a small neural network over a batch, one sample a thread. There is no processor on the chip. A host writes the kernel and the data through an SPI port that speaks the protocol of [`spis`](https://github.com/Tape-Out/spis), starts it, waits for the `done` pin and reads the data memory back.

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
| `mlp.asm` | 16 | batch inference: a 4-4-1 perceptron with ReLU tells the parity of a 2 × 2 pattern, one pattern a thread, weights in the data memory |

One thing the model has to copy: the comparison upstream implements only tells equal from unequal, so `BRn` branches when the operands differ.

## OpenCL

`sw/pocl/` adds a `tgpu` device to PoCL. Host programs call the plain OpenCL 1.2 API.

```console
$ bash sw/pocl/build.sh build/pocl ~/.local/pocl-tgpu            # PoCL v7.2 without LLVM, plus the clrun example
$ python3 sw/gpu.py --ftdi ftdi://ftdi:232h/1 serve 2610 &        # or --model, with no board at all
$ export POCL_DEVICES=tgpu POCL_TGPU0_PARAMETERS=tcp:127.0.0.1:2610
$ clrun builtin 100                                               # pocl.add.i8, pocl.mul.i8, pocl.copy.i8
$ python3 sw/gpu.py build sw/kernels/shade.asm -o shade.tgk && clrun tgk shade.tgk 8
```

Built-in kernels keep PoCL's names and semantics, and the device splits the data into tiles that fit the 256-byte data memory. A `.tgk` file goes through `clCreateProgramWithBinary`: one kernel, `main`, whose only argument is the whole data memory. There is no OpenCL C compiler, as that would need an LLVM backend for tiny-gpu.

## Vulkan and OpenGL

The same computations and the same triangles, written against the standard APIs and run on Mesa's software drivers, give byte-identical results to the chip. `sw/vk/` holds five GLSL compute shaders, one per assembly kernel, with `vkrun` to dispatch one; `vkdraw` sends triangles through the Vulkan graphics pipeline into an 8 × 8 framebuffer. `sw/gl/gldraw` does the same through OpenGL 4.5 on a surfaceless EGL display. Vertices from `gpu.py` are pixel centres, shifted by 0.5 on the way in, and edges follow the top-left rule.

```console
$ bash sw/vk/build.sh build/vk && bash sw/gl/build.sh build/gl
$ echo "1.5 1.5 6.5 2.5 3.5 6.5 200" | build/vk/vkdraw build/vk/tri_vert.spv build/vk/tri_frag.spv | xxd
$ echo "1.5 1.5 6.5 2.5 3.5 6.5 200" | EGL_PLATFORM=surfaceless build/gl/gldraw | xxd
```

## Testing and tape-out

```console
$ ran test to2610-gpu                  # the chip, OpenCL, Vulkan and OpenGL tests, on the Verilog file that goes to the shuttle
$ ran asic to2610-gpu                  # to2610_gpu.v, ecc at 50 MHz, report.json
$ ran asic to2610-gpu --no-run         # only the Verilog file and ecc.toml
```

## License

This repository: 任选其一 [MIT](LICENSE-MIT) · [Apache 2.0](LICENSE-APACHE) · [木兰宽松许可证 第2版](LICENSE-MULAN).

`SPDX-License-Identifier: MIT OR Apache-2.0 OR MulanPSL-2.0`

tiny-gpu itself has no licence file upstream. It is fetched through the `tiny-gpu` repository as a submodule and neither changed nor copied here.

除非另行说明，你提交的贡献按上述三者同时授权，不附加其他条件。
