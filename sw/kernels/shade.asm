; 一张 8 × 8 的灰度图，每个线程算一个像素：到中心距离的平方，越远越亮。写到地址 64 起，按行存。
; 这是最小的「片元着色」：六十四个像素互不相干，十六个块轮流占四个核。
.threads 64

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx      ; i：第几个像素
CONST R1, #8
DIV R2, R0, R1              ; y
MUL R3, R2, R1
SUB R3, R0, R3              ; x
CONST R4, #4
SUB R3, R3, R4              ; x − 4，负数按 8 位回绕
SUB R2, R2, R4
MUL R3, R3, R3              ; 平方时回绕掉的正好是 256 的倍数，结果不受影响
MUL R2, R2, R2
ADD R3, R3, R2              ; 0 至 32
CONST R5, #7
MUL R3, R3, R5              ; 拉到 0 至 224
CONST R6, #64
ADD R6, R6, R0
STR R6, R3
RET
