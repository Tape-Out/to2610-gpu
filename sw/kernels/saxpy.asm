; y[i] = a * x[i] + y[i]，十六个线程一人一项。x 在地址 0 起，y 在 16 起，a 是 3。
.threads 16
.data 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
.data 100 90 80 70 60 50 40 30 20 10 0 5 15 25 35 45

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx      ; i：全局的线程号
CONST R1, #3                ; a
CONST R2, #16               ; y 的起始地址
LDR R3, R0                  ; x[i]
ADD R4, R2, R0
LDR R5, R4                  ; y[i]
MUL R3, R3, R1
ADD R5, R5, R3
STR R4, R5
RET
