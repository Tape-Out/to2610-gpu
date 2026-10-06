; 批量推理：一个 4-4-1 的两层感知机，判 2 × 2 的黑白图样里亮点个数的奇偶。
; 十六种图样一个线程认一种，权与偏置都在数据存储里，换一组权就是换一个模型。
; 奇偶是单层感知机做不了的函数：隐层第 j 个神经元算 ReLU(亮点数 − j)，输出层按 1、−2、2、−2 加权，正好剩下奇偶。
; 没有比较大小的指令，ReLU 用除法取符号位：h ÷ 128 在负数（8 位回绕后不小于 128）时为 1。
.threads 16
; 输入，地址 0 起，每个样本 4 个像素
.data 0 0 0 0  1 0 0 0  0 1 0 0  1 1 0 0
.data 0 0 1 0  1 0 1 0  0 1 1 0  1 1 1 0
.data 0 0 0 1  1 0 0 1  0 1 0 1  1 1 0 1
.data 0 0 1 1  1 0 1 1  0 1 1 1  1 1 1 1
; 第一层的权，地址 64 起，一行一个神经元
.data 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
; 第一层的偏置，地址 80 起：0、−1、−2、−3
.data 0 255 254 253
; 第二层的权，地址 84 起：1、−2、2、−2
.data 1 254 2 254

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx      ; i：第几个样本
CONST R1, #1
CONST R2, #4
MUL R3, R0, R2              ; 这个样本的输入在 4i 起
CONST R5, #64               ; 第一层的权，一路往后走
CONST R6, #0                ; 输出的累加
CONST R4, #0                ; j：第几个隐层神经元
hidden:
CONST R9, #80
ADD R9, R9, R4
LDR R7, R9                  ; 从偏置加起
CONST R11, #0
ADD R11, R11, R3            ; 回到这个样本的第一个像素
CONST R8, #0
dot:
LDR R9, R5
LDR R10, R11
MUL R9, R9, R10
ADD R7, R7, R9
ADD R5, R5, R1
ADD R11, R11, R1
ADD R8, R8, R1
CMP R8, R2
BRn dot
CONST R9, #128
DIV R9, R7, R9              ; 符号位
SUB R9, R1, R9
MUL R7, R7, R9              ; ReLU
CONST R9, #84
ADD R9, R9, R4
LDR R9, R9
MUL R9, R9, R7
ADD R6, R6, R9
ADD R4, R4, R1
CMP R4, R2
BRn hidden
CONST R9, #96
ADD R9, R9, R0
STR R9, R6                  ; 结果写到 96 起：奇数个亮点为 1
RET
