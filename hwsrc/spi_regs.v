// SPI 从口，说 Tape-Out/spis 的那套协议（模式 0，高位先出，地址与数据各 4 字节大端），片外主机经它读写片上的
// 寄存器与存储。主机端因此沿用同一份 spis.py。
//
//   0x02 写：4 字节地址，之后每满 4 字节写一个字，地址加 4
//   0x03 读一个字：4 字节地址，1 个空字节，4 字节数据
//   0x0B 连读 n + 1 个字：4 字节地址，1 字节 n，1 个空字节，之后每字 4 字节
//   0x05 状态：第 0 位是访问过没有东西的地址，读一次即清
//   0x9F 标识：53 50 49 53
//
// 三根输入脚用系统时钟过采样，SCK 不超过主频的八分之一。读只取命令要的那几个字，不预取：
// 连读时下一个字在当前字移出第三个字节之后才去读。片上这一侧一拍读完，不会让主机等。
module spi_regs (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        sck,
  input  wire        cs_n,
  input  wire        mosi,
  output reg         miso,
  output wire        miso_oe,
  output reg  [31:0] addr,
  output reg  [31:0] wdata,
  output reg         we,       // 一拍：把 wdata 写到 addr
  output reg         re,       // 一拍：读 addr，rdata 与 err 在这一拍给出
  input  wire [31:0] rdata,
  input  wire        err       // 与 we、re 同拍：这个地址上没有东西
);
  localparam [2:0] S_CMD = 3'd0, S_ADDR = 3'd1, S_N = 3'd2, S_DUMMY = 3'd3,
                   S_RDATA = 3'd4, S_WDATA = 3'd5, S_OUT = 3'd6;

  reg [2:0] sck_s, cs_s;
  reg [1:0] mosi_s;
  wire      rise = sck_s[2:1] == 2'b01;
  wire      fall = sck_s[2:1] == 2'b10;
  wire      sel = !cs_s[1];
  wire      din = mosi_s[1];
  assign miso_oe = sel;

  reg [2:0]  st;
  reg [7:0]  cmd;
  reg [2:0]  bitn;
  reg [1:0]  cnt;
  reg [6:0]  shin;
  reg [23:0] wsh;
  reg [31:0] osh;
  reg [31:0] hold;
  reg [7:0]  left;      // 连读还剩几个字没取
  reg        bad;

  wire [7:0] b = {shin, din};

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sck_s <= 3'b000;
      cs_s <= 3'b111;
      mosi_s <= 2'b00;
      st <= S_CMD;
      cmd <= 8'd0;
      bitn <= 3'd0;
      cnt <= 2'd0;
      shin <= 7'd0;
      wsh <= 24'd0;
      osh <= 32'd0;
      hold <= 32'd0;
      left <= 8'd0;
      bad <= 1'b0;
      miso <= 1'b0;
      addr <= 32'd0;
      wdata <= 32'd0;
      we <= 1'b0;
      re <= 1'b0;
    end else begin
      sck_s <= {sck_s[1:0], sck};
      cs_s <= {cs_s[1:0], cs_n};
      mosi_s <= {mosi_s[0], mosi};
      we <= 1'b0;
      re <= 1'b0;
      if (re) hold <= rdata;
      if ((re || we) && err) bad <= 1'b1;
      if (we) addr <= addr + 32'd4;

      if (!sel) begin
        st <= S_CMD;
        bitn <= 3'd0;
        cnt <= 2'd0;
        miso <= 1'b0;
      end else begin
        if (fall) begin
          miso <= osh[31];
          osh <= {osh[30:0], 1'b0};
        end
        if (rise) begin
          shin <= b[6:0];
          bitn <= bitn + 3'd1;
          if (bitn == 3'd7) begin
            case (st)
              S_CMD: begin
                cmd <= b;
                cnt <= 2'd0;
                osh <= 32'd0;
                st <= S_OUT;
                case (b)
                  8'h02, 8'h03, 8'h0b: st <= S_ADDR;
                  8'h9f: osh <= 32'h5350_4953;
                  8'h05: begin
                    osh <= {7'd0, bad, 24'd0};
                    bad <= 1'b0;
                  end
                  default: ;
                endcase
              end
              S_ADDR: begin
                addr <= {addr[23:0], b};
                cnt <= cnt + 2'd1;
                if (cnt == 2'd3) begin
                  if (cmd == 8'h02) st <= S_WDATA;
                  else if (cmd == 8'h03) begin
                    re <= 1'b1;
                    left <= 8'd0;
                    st <= S_DUMMY;
                  end else st <= S_N;
                end
              end
              S_N: begin
                left <= b;
                re <= 1'b1;
                st <= S_DUMMY;
              end
              S_DUMMY: begin
                osh <= hold;
                cnt <= 2'd0;
                st <= S_RDATA;
              end
              S_RDATA: begin
                cnt <= cnt + 2'd1;
                if (cnt == 2'd2 && left != 8'd0) begin
                  addr <= addr + 32'd4;
                  re <= 1'b1;
                end
                if (cnt == 2'd3) begin
                  // 要的字读完之后再给时钟，移出来的是 0，不碰片上任何东西
                  osh <= left != 8'd0 ? hold : 32'd0;
                  if (left != 8'd0) left <= left - 8'd1;
                end
              end
              S_WDATA: begin
                wsh <= {wsh[15:0], b};
                cnt <= cnt + 2'd1;
                if (cnt == 2'd3) begin
                  wdata <= {wsh, b};
                  we <= 1'b1;
                end
              end
              default: ;
            endcase
          end
        end
      end
    end
  end
endmodule
