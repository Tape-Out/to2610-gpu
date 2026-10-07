// 把数据存储里的一块当帧缓冲扫成 640×480@60 的 VGA。像素钟取主频的一半（50 MHz 出 25 MHz，比标准的
// 25.175 MHz 慢 0.7%，显示器都收）。一个字节一个像素：RGB332，或 8 位灰度展到三路上。
// 源图比屏小，每个源像素横着重复 sx 个屏上像素、竖着重复 sy 行；放大之后超出源图的部分是黑的。
// 另出 DE 与像素钟，板上的 TFP410 收这同一组并行信号转成 HDMI 口上的 DVI；像素钟的上升沿落在数据的正中。
module vga (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       en,
  input  wire       rgb,
  input  wire [7:0] base,
  input  wire [7:0] w,
  input  wire [7:0] h,
  input  wire [9:0] sx,
  input  wire [9:0] sy,
  output wire [7:0] addr,
  input  wire [7:0] pix,
  output reg  [2:0] r,
  output reg  [2:0] g,
  output reg  [1:0] b,
  output reg        hs,
  output reg        vs,
  output reg        de,
  output wire       pclk
);
  localparam HV = 640, HF = 16, HS = 96, HB = 48, HT = HV + HF + HS + HB;
  localparam VV = 480, VF = 10, VS = 2, VB = 33, VT = VV + VF + VS + VB;

  reg       ce;
  reg [9:0] hc, vc;
  // 源图里的位置：列与行各带一个放大用的子计数，row 是这一行第一个像素的地址
  reg [9:0] xs, ys;
  reg [7:0] col, line, row;

  wire h_end = hc == HT - 1;
  wire v_end = vc == VT - 1;
  wire in_x = hc < HV && col < w;
  wire in_y = vc < VV && line < h;

  assign addr = row + col;
  assign pclk = ce;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ce <= 1'b0;
      hc <= 10'd0;
      vc <= 10'd0;
      xs <= 10'd0;
      ys <= 10'd0;
      col <= 8'd0;
      line <= 8'd0;
      row <= 8'd0;
    end else begin
      ce <= !ce;
      if (ce) begin
        hc <= h_end ? 10'd0 : hc + 10'd1;
        if (h_end) begin
          col <= 8'd0;
          xs <= 10'd0;
        end else if (hc < HV) begin
          if (xs + 10'd1 >= sx) begin
            xs <= 10'd0;
            if (col != 8'hff) col <= col + 8'd1;
          end else xs <= xs + 10'd1;
        end
        if (h_end) begin
          vc <= v_end ? 10'd0 : vc + 10'd1;
          if (v_end) begin
            line <= 8'd0;
            ys <= 10'd0;
            row <= base;
          end else if (vc < VV) begin
            if (ys + 10'd1 >= sy) begin
              ys <= 10'd0;
              if (line != 8'hff) line <= line + 8'd1;
              row <= row + w;
            end else ys <= ys + 10'd1;
          end
        end
      end
    end
  end

  // 存储是组合读的，颜色与同步一起打一拍，三者在脚上对齐
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      {r, g, b} <= 8'd0;
      hs <= 1'b1;
      vs <= 1'b1;
      de <= 1'b0;
    end else if (ce) begin
      if (en && in_x && in_y) {r, g, b} <= rgb ? pix : {pix[7:5], pix[7:5], pix[7:6]};
      else {r, g, b} <= 8'd0;
      hs <= !(en && hc >= HV + HF && hc < HV + HF + HS);
      vs <= !(en && vc >= VV + VF && vc < VV + VF + VS);
      de <= en && hc < HV && vc < VV;
    end
  end
endmodule
