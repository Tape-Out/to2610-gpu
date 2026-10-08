/* 到 to2610-gpu 管理口的一条线，与线上说的 spis 协议（Tape-Out/spis 的 README；主机端的 Python 版是 sw/spis.py）。 */
#ifndef TGPU_WIRE_H
#define TGPU_WIRE_H

#include <stddef.h>
#include <stdint.h>

typedef struct wire wire;

/* spidev:/dev/spidev0.0[@赫兹]，SCK 默认 2 MHz，不能超过芯片主频的八分之一；
   tcp:主机:端口，另一头是 sw/gpu.py serve，或整片测试的测试台 */
wire *wire_open (const char *spec);
void wire_close (wire *l);
int wire_xfer (wire *l, const uint8_t *tx, uint8_t *rx, size_t n);

int spis_wr (wire *l, uint32_t addr, const uint32_t *w, size_t n);
int spis_rd (wire *l, uint32_t addr, uint32_t *w, size_t n);

#endif
