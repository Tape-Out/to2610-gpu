// to2610-gpu 的核心：tiny-gpu、它的程序存储与数据存储、一个 SPI 管理口。
// 片外主机经管理口写进内核程序与数据、设线程数、起跑，跑完读回数据存储。
//
// 管理口上的地址（字对齐，都在 0x1000_0000 起的一页里）：
//   0x000 CTRL     写 1 起跑：先给 tiny-gpu 一个复位，再拉高 start；写 0 收手。读回 start
//   0x004 THREADS  线程数，要是每块线程数的整数倍（上游在最后一块装不满时不结束）
//   0x008 STATUS   第 0 位 done，第 1 位 busy
//   0x00C CYCLES   从起跑到 done 的时钟拍数
//   0x010 ID       0x54475055（「TGPU」）
//   0x014 CONFIG   高字节核数，低字节每块线程数
//   0x018 VCTRL    VGA：第 0 位开，第 1 位 RGB332（0 是灰度）
//   0x01C VFMT     帧缓冲在数据存储里的起始地址、宽、高，各一个字节（低到高）
//   0x020 VSCALE   每个源像素占几个屏上像素：低 10 位横、16 起 10 位竖
//   0x400 起       程序存储，256 个字，每字 16 位
//   0x800 起       数据存储，256 个字节，每个占一个字
module tgpu_soc #(
  parameter NUM_CORES = 4,
  parameter THREADS_PER_BLOCK = 4
) (
  input  wire clk,
  input  wire rst_n,
  input  wire sck,
  input  wire cs_n,
  input  wire mosi,
  output wire miso,
  output wire miso_oe,
  output wire done,
  output wire busy,
  output wire [2:0] vga_r,
  output wire [2:0] vga_g,
  output wire [1:0] vga_b,
  output wire vga_hs,
  output wire vga_vs,
  output wire vga_de,
  output wire vga_pclk
);
  localparam CH = 4;    // 数据存储的通道数，上游的测试写死了 4
  localparam [7:0] CORES = NUM_CORES;
  localparam [7:0] TPB = THREADS_PER_BLOCK;

  wire [31:0] addr, wdata;
  wire        we, re;
  reg  [31:0] rdata;
  reg         err;
  spi_regs mgmt (
    .clk(clk), .rst_n(rst_n), .sck(sck), .cs_n(cs_n), .mosi(mosi), .miso(miso), .miso_oe(miso_oe),
    .addr(addr), .wdata(wdata), .we(we), .re(re), .rdata(rdata), .err(err)
  );

  wire       page = addr[31:12] == 20'h10000;
  wire [1:0] area = addr[11:10];
  wire [7:0] idx = addr[9:2];
  wire       sel_reg = page && area == 2'd0;
  wire       sel_prog = page && area == 2'd1;
  wire       sel_data = page && area == 2'd2;

  reg        start;
  // 起跑的三步，照上游测试的次序：3、2 两拍复位，1 这一拍写线程数（复位会把它清掉），到 0 才拉高 start
  reg [1:0]  rst_cnt;
  reg [7:0]  threads;
  reg [31:0] cycles;
  reg [1:0]  vctrl;
  reg [23:0] vfmt;
  reg [25:0] vscale;
  wire       gdone;
  wire       grst = !rst_n || rst_cnt[1];

  assign done = gdone;
  assign busy = start && !gdone;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start <= 1'b0;
      rst_cnt <= 2'd3;
      threads <= 8'd0;
      cycles <= 32'd0;
      // 复位后的格式就是整块数据存储当 16 × 16 的图，正好铺满一屏
      vctrl <= 2'd0;
      vfmt <= {8'd16, 8'd16, 8'd0};
      vscale <= {10'd30, 6'd0, 10'd40};
    end else begin
      if (rst_cnt != 2'd0) rst_cnt <= rst_cnt - 2'd1;
      if (start && rst_cnt == 2'd0 && !gdone) cycles <= cycles + 32'd1;
      if (we && sel_reg && idx == 8'd0) begin
        start <= wdata[0];
        if (wdata[0] && !start) begin
          rst_cnt <= 2'd3;
          cycles <= 32'd0;
        end
      end
      if (we && sel_reg && idx == 8'd1) threads <= wdata[7:0];
      if (we && sel_reg && idx == 8'd6) vctrl <= wdata[1:0];
      if (we && sel_reg && idx == 8'd7) vfmt <= wdata[23:0];
      if (we && sel_reg && idx == 8'd8) vscale <= {wdata[25:16], 6'd0, wdata[9:0]};
    end
  end

  wire                p_valid;
  wire [7:0]          p_addr;
  reg                 p_ready;
  reg  [15:0]         p_data;
  wire [CH - 1:0]     r_valid, w_valid;
  wire [CH * 8 - 1:0] r_addr, w_addr, w_data;
  reg  [CH - 1:0]     r_ready, w_ready;
  reg  [CH * 8 - 1:0] r_data;

  gpu #(.NUM_CORES(NUM_CORES), .THREADS_PER_BLOCK(THREADS_PER_BLOCK)) u_gpu (
    .clk(clk), .reset(grst), .start(start && rst_cnt == 2'd0), .done(gdone),
    .device_control_write_enable(rst_cnt == 2'd1), .device_control_data(threads),
    .program_mem_read_valid(p_valid), .program_mem_read_address(p_addr),
    .program_mem_read_ready(p_ready), .program_mem_read_data(p_data),
    .data_mem_read_valid(r_valid), .data_mem_read_address(r_addr),
    .data_mem_read_ready(r_ready), .data_mem_read_data(r_data),
    .data_mem_write_valid(w_valid), .data_mem_write_address(w_addr),
    .data_mem_write_data(w_data), .data_mem_write_ready(w_ready)
  );

  // 两块存储都是触发器搭的。对 tiny-gpu 的应答晚一拍，与上游测试里的存储模型同一个节拍
  reg [15:0] prog [0:255];
  reg [7:0]  data [0:255];
  integer    c;

  always @(posedge clk) begin
    if (we && sel_prog) prog[idx] <= wdata[15:0];
    p_data <= prog[p_addr];
    for (c = 0; c < CH; c = c + 1) begin
      r_data[8 * c +: 8] <= data[r_addr[8 * c +: 8]];
      if (w_valid[c]) data[w_addr[8 * c +: 8]] <= w_data[8 * c +: 8];
    end
    if (we && sel_data) data[idx] <= wdata[7:0];
  end

  wire [7:0] vaddr;
  vga video (
    .clk(clk), .rst_n(rst_n), .en(vctrl[0]), .rgb(vctrl[1]),
    .base(vfmt[7:0]), .w(vfmt[15:8]), .h(vfmt[23:16]), .sx(vscale[9:0]), .sy(vscale[25:16]),
    .addr(vaddr), .pix(data[vaddr]),
    .r(vga_r), .g(vga_g), .b(vga_b), .hs(vga_hs), .vs(vga_vs),
    .de(vga_de), .pclk(vga_pclk)
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p_ready <= 1'b0;
      r_ready <= {CH{1'b0}};
      w_ready <= {CH{1'b0}};
    end else begin
      p_ready <= p_valid;
      r_ready <= r_valid;
      w_ready <= w_valid;
    end
  end

  always @* begin
    rdata = 32'd0;
    err = 1'b0;
    if (sel_prog) rdata = {16'd0, prog[idx]};
    else if (sel_data) rdata = {24'd0, data[idx]};
    else if (sel_reg) begin
      case (idx)
        8'd0: rdata = {31'd0, start};
        8'd1: rdata = {24'd0, threads};
        8'd2: rdata = {30'd0, busy, gdone};
        8'd3: rdata = cycles;
        8'd4: rdata = 32'h5447_5055;
        8'd5: rdata = {16'd0, CORES, TPB};
        8'd6: rdata = {30'd0, vctrl};
        8'd7: rdata = {8'd0, vfmt};
        8'd8: rdata = {6'd0, vscale};
        default: err = 1'b1;
      endcase
    end else err = 1'b1;
  end
endmodule
