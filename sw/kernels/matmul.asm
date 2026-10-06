; C = A × B，4 × 4，十六个线程一人算一个元素。A 在地址 0 起，B 在 16 起，C 写到 32 起；都按行存。
.threads 16
.data 1 2 3 4  5 6 7 8  9 10 11 12  13 14 15 16
.data 1 0 2 0  0 1 0 2  3 0 1 0  0 3 0 1

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx      ; i：要算 C 的第几个元素
CONST R1, #4                ; N
CONST R2, #1
DIV R3, R0, R1              ; 行
MUL R4, R3, R1
SUB R4, R0, R4              ; 列
MUL R5, R3, R1              ; A 这一行的头
CONST R6, #16
ADD R6, R6, R4              ; B 这一列的头
CONST R7, #0                ; 累加
CONST R8, #0                ; k
next:
LDR R9, R5
LDR R10, R6
MUL R9, R9, R10
ADD R7, R7, R9
ADD R5, R5, R2              ; A 往右一格
ADD R6, R6, R1              ; B 往下一行
ADD R8, R8, R2
CMP R8, R1
BRn next                    ; k 还不等于 N 就再来：这颗 GPU 的比较只分得出相等与不等
CONST R9, #32
ADD R9, R9, R0
STR R9, R7
RET
