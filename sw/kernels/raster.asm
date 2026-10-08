; 光栅化：把一个三角形画进 8 × 8 的帧缓冲，一个线程管一个像素。
; 三角形由主机换算成三条边函数 E = A·x + B·y + C（gpu.py 的 triangle，左上规则折进了 C），三条都不为负就在三角形里，
; 那个像素写成三角形的颜色，不在就保留原来的颜色。帧缓冲不清，一个接一个画就是画家算法：后画的盖住先画的。
; 地址 0 至 8 是三条边的 A、B、C，9 是颜色，帧缓冲在 64 起。数都是 8 位回绕的：负数存成 256 减它的绝对值，
; 所以每个像素上的 E 要落在 −128 至 127 之间，主机建立三角形时核过。没有比较大小的指令，E 的符号用 E ÷ 128 取。
.threads 64
; 默认画的那一个：(1,1)、(6,2)、(3,6)，颜色 200
.data 255 5 251  252 253 29  5 254 253  200

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx      ; i：第几个像素
CONST R1, #8
DIV R2, R0, R1              ; y
MUL R3, R2, R1
SUB R3, R0, R3              ; x：没有与运算，用 i − 8y
CONST R12, #128
CONST R11, #1
CONST R10, #1               ; 在不在三角形里，三条边各乘一次

CONST R4, #0
LDR R5, R4
MUL R5, R5, R3
CONST R4, #1
LDR R6, R4
MUL R6, R6, R2
ADD R5, R5, R6
CONST R4, #2
LDR R6, R4
ADD R5, R5, R6              ; 第一条边的 E
DIV R5, R5, R12             ; 负的是 1
SUB R5, R11, R5
MUL R10, R10, R5

CONST R4, #3
LDR R5, R4
MUL R5, R5, R3
CONST R4, #4
LDR R6, R4
MUL R6, R6, R2
ADD R5, R5, R6
CONST R4, #5
LDR R6, R4
ADD R5, R5, R6
DIV R5, R5, R12
SUB R5, R11, R5
MUL R10, R10, R5

CONST R4, #6
LDR R5, R4
MUL R5, R5, R3
CONST R4, #7
LDR R6, R4
MUL R6, R6, R2
ADD R5, R5, R6
CONST R4, #8
LDR R6, R4
ADD R5, R5, R6
DIV R5, R5, R12
SUB R5, R11, R5
MUL R10, R10, R5

CONST R4, #64
ADD R4, R4, R0
LDR R5, R4                  ; 原来的颜色
CONST R6, #9
LDR R6, R6
SUB R6, R6, R5
MUL R6, R6, R10             ; 在里面才加上「新减旧」
ADD R5, R5, R6
STR R4, R5
RET
