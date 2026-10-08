#version 460
// 像素坐标进来：8 × 8 的帧缓冲上 x、y 都是 0 至 8，换成 −1 至 1。Vulkan 的 y 朝下，与帧缓冲按行存的次序一致
layout(location = 0) in vec2 pos;
layout(location = 1) in uint color;
layout(location = 0) flat out uint col;

void main() {
  gl_Position = vec4(pos / 4.0 - 1.0, 0.0, 1.0);
  col = color;
}
