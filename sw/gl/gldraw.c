// 用 OpenGL 把一组三角形画进 8 × 8 的帧缓冲（GL_R8UI，清成 0），读回 64 字节写到标准输出，同 vkdraw 与 raster.asm。
// 不要窗口：EGL 走 Mesa 的无表面平台，画进帧缓冲对象。用法与 vkdraw 相同：gldraw < 三角形，一行 x0 y0 x1 y1 x2 y2 颜色。
// 读回的第 0 行是窗口坐标 y = 0 那一行，标准化坐标 −1 落在那里，与 Vulkan 帧缓冲的第 0 行是同一行三角形
#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define W 8
#define MAXTRI 64

static const char *VS = "#version 450 core\n"
                        "layout(location = 0) in vec2 pos;\n"
                        "layout(location = 1) in uint color;\n"
                        "layout(location = 0) flat out uint col;\n"
                        "void main() { gl_Position = vec4(pos / 4.0 - 1.0, 0.0, 1.0); col = color; }\n";
static const char *FS = "#version 450 core\n"
                        "layout(location = 0) flat in uint col;\n"
                        "layout(location = 0) out uint frag;\n"
                        "void main() { frag = col; }\n";

typedef struct {
  float x, y;
  uint32_t color;
} Vertex;

static void die(const char *what) {
  fprintf(stderr, "gldraw：%s\n", what);
  exit(1);
}

static GLuint shader(GLenum type, const char *src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, sizeof log, nullptr, log);
    die(log);
  }
  return s;
}

int main(void) {
  Vertex tri[MAXTRI * 3];
  int n = 0;
  float v[6];
  unsigned c;
  while (n < MAXTRI && scanf("%f %f %f %f %f %f %u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &c) == 7) {
    for (int k = 0; k < 3; k++) tri[n * 3 + k] = (Vertex){v[2 * k], v[2 * k + 1], c};
    n++;
  }

  EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
  if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) die("EGL 起不来");
  if (!eglBindAPI(EGL_OPENGL_API)) die("EGL 不给 OpenGL");
  EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT,
                                    (const EGLint[]){EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 5,
                                                     EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE});
  if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) die("建不了 OpenGL 4.5 的上下文");
  fprintf(stderr, "%s\n", (const char *)glGetString(GL_RENDERER));

  GLuint rb, fb;
  glGenRenderbuffers(1, &rb);
  glBindRenderbuffer(GL_RENDERBUFFER, rb);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_R8UI, W, W);
  glGenFramebuffers(1, &fb);
  glBindFramebuffer(GL_FRAMEBUFFER, fb);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("帧缓冲对象不完整");
  glClearBufferuiv(GL_COLOR, 0, (const GLuint[]){0, 0, 0, 0});

  GLuint prog = glCreateProgram();
  glAttachShader(prog, shader(GL_VERTEX_SHADER, VS));
  glAttachShader(prog, shader(GL_FRAGMENT_SHADER, FS));
  glLinkProgram(prog);
  GLint ok;
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) die("着色器连不上");
  glUseProgram(prog);

  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(Vertex) * 3 * (size_t)n), tri, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
  glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, sizeof(Vertex), (void *)8);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glViewport(0, 0, W, W);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glDrawArrays(GL_TRIANGLES, 0, 3 * n);

  uint8_t out[W * W];
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, W, W, GL_RED_INTEGER, GL_UNSIGNED_BYTE, out);
  if (glGetError() != GL_NO_ERROR) die("读回出错");
  fwrite(out, 1, sizeof out, stdout);

  eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(dpy, ctx);
  eglTerminate(dpy);
  return 0;
}
