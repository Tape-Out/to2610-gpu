/* 经 OpenCL 用 to2610-gpu：只调标准的 OpenCL 1.2 接口，换成别家的设备照样跑内建内核那一段。
     clrun info                 平台与设备
     clrun builtin [个数]        pocl.add.i8、pocl.mul.i8、pocl.copy.i8 各跑一遍，与主机算的逐个比
     clrun tgk 文件 [每行字节]   跑 sw/gpu.py build 出的内核，数据存储的初值取文件里的，跑完打出整块
   设备由 PoCL 的环境变量选：POCL_DEVICES=tgpu POCL_TGPU0_PARAMETERS=tcp:127.0.0.1:2610 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                              \
  do                                                                          \
    {                                                                         \
      cl_int e_ = (x);                                                        \
      if (e_ != CL_SUCCESS)                                                   \
        {                                                                     \
          fprintf (stderr, "%s：%d\n", #x, e_);                              \
          exit (1);                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

static cl_device_id dev;
static cl_context ctx;
static cl_command_queue q;

static void
open_device (void)
{
  cl_platform_id plat;
  CHECK (clGetPlatformIDs (1, &plat, NULL));
  CHECK (clGetDeviceIDs (plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL));
  cl_int e;
  ctx = clCreateContext (NULL, 1, &dev, NULL, NULL, &e);
  CHECK (e);
  q = clCreateCommandQueue (ctx, dev, 0, &e);
  CHECK (e);
}

static cl_mem
buffer (size_t n, void *init)
{
  cl_int e;
  cl_mem m = clCreateBuffer (
      ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), n, init, &e);
  CHECK (e);
  return m;
}

static int
info (void)
{
  char name[256], vendor[256], version[256], bik[1024];
  cl_uint cu;
  size_t wg;
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_NAME, sizeof name, name, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_VENDOR, sizeof vendor, vendor, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_VERSION, sizeof version, version, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wg, &wg, NULL));
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_BUILT_IN_KERNELS, sizeof bik, bik, NULL));
  printf ("%s（%s）%s\n计算单元 %u，工作组最大 %zu\n内建内核 %s\n", name, vendor,
          version, cu, wg, bik);
  return 0;
}

static int
builtin (size_t n)
{
  static const char *names[] = { "pocl.add.i8", "pocl.mul.i8", "pocl.copy.i8" };
  signed char *a = malloc (n), *b = malloc (n), *c = malloc (n);
  srand (2610);
  for (size_t i = 0; i < n; ++i)
    a[i] = rand (), b[i] = rand ();
  cl_int e;
  cl_program p = clCreateProgramWithBuiltInKernels (
      ctx, 1, &dev, "pocl.add.i8;pocl.mul.i8;pocl.copy.i8", &e);
  CHECK (e);
  CHECK (clBuildProgram (p, 1, &dev, NULL, NULL, NULL));
  int bad = 0;
  for (int k = 0; k < 3; ++k)
    {
      cl_kernel kr = clCreateKernel (p, names[k], &e);
      CHECK (e);
      cl_mem ma = buffer (n, a), mb = buffer (n, b), mc = buffer (n, NULL);
      cl_uint arg = 0;
      CHECK (clSetKernelArg (kr, arg++, sizeof ma, &ma));
      if (k < 2)
        CHECK (clSetKernelArg (kr, arg++, sizeof mb, &mb));
      CHECK (clSetKernelArg (kr, arg++, sizeof mc, &mc));
      CHECK (clEnqueueNDRangeKernel (q, kr, 1, NULL, &n, NULL, 0, NULL, NULL));
      CHECK (clEnqueueReadBuffer (q, mc, CL_TRUE, 0, n, c, 0, NULL, NULL));
      size_t wrong = 0;
      for (size_t i = 0; i < n; ++i)
        {
          signed char want = k == 0 ? a[i] + b[i] : k == 1 ? a[i] * b[i] : a[i];
          wrong += c[i] != want;
        }
      printf ("%s %zu 个，错 %zu 个\n", names[k], n, wrong);
      bad |= wrong != 0;
      clReleaseMemObject (ma);
      clReleaseMemObject (mb);
      clReleaseMemObject (mc);
      clReleaseKernel (kr);
    }
  clReleaseProgram (p);
  return bad;
}

static int
tgk (const char *path, int width)
{
  FILE *f = fopen (path, "rb");
  if (!f)
    {
      perror (path);
      return 1;
    }
  unsigned char bin[12 + 512 + 256];
  size_t len = fread (bin, 1, sizeof bin, f);
  fclose (f);
  if (len < 12)
    {
      fprintf (stderr, "%s 不是 .tgk\n", path);
      return 1;
    }
  size_t threads = bin[4] | bin[5] << 8, np = bin[6] | bin[7] << 8,
         nd = bin[8] | bin[9] << 8;
  unsigned char mem[256] = { 0 };
  if (12 + 2 * np + nd == len)
    memcpy (mem, bin + 12 + 2 * np, nd);

  const unsigned char *bins = bin;
  cl_int e, st;
  cl_program p
      = clCreateProgramWithBinary (ctx, 1, &dev, &len, &bins, &st, &e);
  CHECK (e);
  CHECK (clBuildProgram (p, 1, &dev, NULL, NULL, NULL));
  cl_kernel kr = clCreateKernel (p, "main", &e);
  CHECK (e);
  cl_mem m = buffer (sizeof mem, mem);
  CHECK (clSetKernelArg (kr, 0, sizeof m, &m));
  size_t local;
  CHECK (clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof local,
                          &local, NULL));
  CHECK (clEnqueueNDRangeKernel (q, kr, 1, NULL, &threads, &local, 0, NULL,
                                 NULL));
  CHECK (clEnqueueReadBuffer (q, m, CL_TRUE, 0, sizeof mem, mem, 0, NULL,
                              NULL));
  for (int i = 0; i < 256; ++i)
    printf ("%3d%c", mem[i], (i + 1) % width ? ' ' : '\n');
  clReleaseMemObject (m);
  clReleaseKernel (kr);
  clReleaseProgram (p);
  return 0;
}

int
main (int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf (stderr, "用法：clrun info | builtin [个数] | tgk 文件 [每行字节]\n");
      return 2;
    }
  open_device ();
  if (!strcmp (argv[1], "info"))
    return info ();
  if (!strcmp (argv[1], "builtin"))
    return builtin (argc > 2 ? strtoul (argv[2], NULL, 0) : 100);
  if (!strcmp (argv[1], "tgk") && argc > 2)
    return tgk (argv[2], argc > 3 ? atoi (argv[3]) : 16);
  fprintf (stderr, "不认识的命令 %s\n", argv[1]);
  return 2;
}
