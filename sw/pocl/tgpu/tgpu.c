/* to2610-gpu 当作 PoCL 的一个设备：缓冲区留在主机内存里，起跑时经管理口装进片上的两块存储，跑完读回。
   内建内核用 PoCL 自己的名字，与别的设备上的同名内核一个算法，换设备不改主机程序；
   自家写的内核是 sw/gpu.py build 出的 .tgk，经 clCreateProgramWithBinary 装进来。
   寄存器照 hwsrc/tgpu_soc.v 的文件头。 */

#include "tgpu.h"

#include <string.h>

#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "wire.h"
#include "pocl_builtin_kernels.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "utlist.h"

#define BASE 0x10000000u
#define TGPU_ID 0x54475055u
enum
{
  CTRL = 0x000,
  THREADS = 0x004,
  STATUS = 0x008,
  ID = 0x010,
  PROG = 0x400,
  DATA = 0x800,
  MEM = 256,
  HEAD = 12
};

/* 硬件不报超时：线程数配错或内核死循环时 done 永远不来，主机得自己收手 */
#define LIMIT_NS (30ull * 1000000000ull)

typedef struct
{
  wire *l;
  unsigned cores, tpb;
  cl_bool available;
  pocl_lock_t cq_lock;
  _cl_command_node *ready_list;
  _cl_command_node *command_list;
} tgpu_data;

/* tiny-gpu 的指令：高 4 位操作码，其后三个 4 位寄存器号，或一个寄存器号加 8 位立即数 */
enum
{
  R0,
  R1,
  R2,
  R3,
  R4,
  R5,
  R7 = 7,
  BID = 13,
  BDIM,
  TID
};
enum
{
  ADD = 0x3,
  MUL = 0x5
};
#define ALU(op, d, s, t) (uint16_t)((op) << 12 | (d) << 8 | (s) << 4 | (t))
#define CONST(d, v) (uint16_t)(0x9 << 12 | (d) << 8 | ((v) & 0xFF))
#define LDR(d, s) (uint16_t)(0x7 << 12 | (d) << 8 | (s) << 4)
#define STR(a, v) (uint16_t)(0x8 << 12 | (a) << 4 | (v))
#define RET 0xF000

/* i = blockIdx × blockDim + threadIdx，c[i] = a[i] op b[i]；op 为 0 时是 c[i] = a[i] */
static size_t
elementwise (uint16_t *p, unsigned op, unsigned a, unsigned b, unsigned c)
{
  size_t n = 0;
  p[n++] = ALU (MUL, R0, BID, BDIM);
  p[n++] = ALU (ADD, R0, R0, TID);
  p[n++] = CONST (R1, a);
  p[n++] = CONST (R3, c);
  p[n++] = ALU (ADD, R4, R1, R0);
  p[n++] = LDR (R4, R4);
  if (op)
    {
      p[n++] = CONST (R2, b);
      p[n++] = ALU (ADD, R5, R2, R0);
      p[n++] = LDR (R5, R5);
      p[n++] = ALU (op, R4, R4, R5);
    }
  p[n++] = ALU (ADD, R7, R3, R0);
  p[n++] = STR (R7, R4);
  p[n++] = RET;
  return n;
}

static const char *
launch (tgpu_data *d, const uint16_t *prog, size_t np, const uint8_t *data,
        size_t nd, unsigned threads)
{
  uint32_t w[MEM], zero = 0, one = 1, t = threads;
  for (size_t i = 0; i < np; ++i)
    w[i] = prog[i];
  if (spis_wr (d->l, BASE + CTRL, &zero, 1)
      || spis_wr (d->l, BASE + PROG, w, np))
    return "tgpu：管理口写不进去";
  for (size_t i = 0; i < nd; ++i)
    w[i] = data[i];
  if (spis_wr (d->l, BASE + DATA, w, nd)
      || spis_wr (d->l, BASE + THREADS, &t, 1)
      || spis_wr (d->l, BASE + CTRL, &one, 1))
    return "tgpu：管理口写不进去";
  uint64_t end = pocl_gettimemono_ns () + LIMIT_NS;
  for (uint32_t s = 0; !(s & 1);)
    {
      if (spis_rd (d->l, BASE + STATUS, &s, 1))
        return "tgpu：管理口读不回来";
      if (pocl_gettimemono_ns () > end)
        return "tgpu：三十秒没等到 done";
    }
  return NULL;
}

static const char *
fetch (tgpu_data *d, unsigned at, uint8_t *out, size_t n)
{
  uint32_t w[MEM];
  if (spis_rd (d->l, BASE + DATA + 4 * at, w, n))
    return "tgpu：管理口读不回来";
  for (size_t i = 0; i < n; ++i)
    out[i] = w[i];
  return NULL;
}

/* 数据存储只有 256 字节：输入输出各占一段，放不下就分块跑；每块凑成每块线程数的整数倍，多出来的线程算的是 0 */
static const char *
run_builtin (tgpu_data *d, unsigned id, uint8_t **arg, const size_t *size,
             size_t off, size_t n)
{
  int two = id != POCL_CDBI_COPY_I8;
  unsigned op = id == POCL_CDBI_ADD_I8 ? ADD : id == POCL_CDBI_MUL_I8 ? MUL : 0;
  uint8_t *out = arg[two ? 2 : 1];
  for (int i = 0; i <= two + 1; ++i)
    if (size[i] < off + n)
      return "tgpu：缓冲区比全局大小小";
  size_t tile = MEM / (two ? 3 : 2) / d->tpb * d->tpb;
  uint16_t p[16];
  uint8_t mem[MEM];
  for (size_t at = off; at < off + n; at += tile)
    {
      size_t k = off + n - at < tile ? off + n - at : tile;
      size_t k4 = (k + d->tpb - 1) / d->tpb * d->tpb;
      memset (mem, 0, sizeof mem);
      memcpy (mem, arg[0] + at, k);
      if (two)
        memcpy (mem + k4, arg[1] + at, k);
      size_t np = elementwise (p, op, 0, k4, two ? 2 * k4 : k4);
      const char *err = launch (d, p, np, mem, two ? 2 * k4 : k4, k4);
      if (!err)
        err = fetch (d, two ? 2 * k4 : k4, out + at, k);
      if (err)
        return err;
    }
  return NULL;
}

/* .tgk：小端的 'TGK1'、线程数、程序条数、数据字节数、两个字节的空，再是程序与数据（sw/gpu.py 的 tgk()） */
static unsigned
u16 (const uint8_t *p)
{
  return p[0] | p[1] << 8;
}

static int
tgk_ok (const uint8_t *b, size_t len)
{
  if (len < HEAD || memcmp (b, "TGK1", 4))
    return 0;
  size_t np = u16 (b + 6), nd = u16 (b + 8);
  return np >= 1 && np <= MEM && nd <= MEM && len == HEAD + 2 * np + nd;
}

/* 唯一的参数就是整块数据存储：缓冲区里的内容装进去（不足 256 字节的补 0），跑完照缓冲区的大小读回 */
static const char *
run_binary (tgpu_data *d, const uint8_t *bin, uint8_t *buf, size_t size,
            size_t off, size_t n, size_t local)
{
  if (off)
    return "tgpu：不支持全局偏移";
  if (local != d->tpb)
    return "tgpu：每个工作组的大小要等于硬件每块的线程数";
  if (n > 255)
    return "tgpu：线程数最多 255";
  size_t np = u16 (bin + 6);
  uint16_t prog[MEM];
  for (size_t i = 0; i < np; ++i)
    prog[i] = u16 (bin + HEAD + 2 * i);
  uint8_t mem[MEM] = { 0 };
  size_t m = size < MEM ? size : MEM;
  memcpy (mem, buf, m);
  const char *err = launch (d, prog, np, mem, MEM, n);
  return err ? err : fetch (d, 0, buf, m);
}

static const char *
run (_cl_command_node *node)
{
  tgpu_data *d = node->device->data;
  cl_kernel kernel = node->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &node->command.run.pc;
  if (pc->num_groups[1] * pc->local_size[1] != 1
      || pc->num_groups[2] * pc->local_size[2] != 1)
    return "tgpu：只有一维";
  size_t n = pc->num_groups[0] * pc->local_size[0];
  if (n == 0)
    return NULL;

  uint8_t *arg[3];
  size_t size[3];
  if (meta->num_args > 3)
    return "tgpu：参数太多";
  for (unsigned i = 0; i < meta->num_args; ++i)
    {
      struct pocl_argument *al = &node->command.run.arguments[i];
      if (meta->arg_info[i].type != POCL_ARG_TYPE_POINTER || !al->value
          || al->is_raw_ptr)
        return "tgpu：参数要是缓冲区";
      cl_mem m = *(cl_mem *)al->value;
      arg[i] = m->device_ptrs[node->device->global_mem_id].mem_ptr;
      size[i] = m->size;
    }

  if (program->builtin_kernel_names)
    return run_builtin (d, meta->builtin_kernel_id, arg, size,
                        pc->global_offset[0], n);
  unsigned dev_i = node->program_device_i;
  return run_binary (d, program->binaries[dev_i], arg[0], size[0],
                     pc->global_offset[0], n, pc->local_size[0]);
}

static void
exec (_cl_command_node *node)
{
  if (node->type != CL_COMMAND_NDRANGE_KERNEL)
    {
      pocl_exec_command (node);
      return;
    }
  cl_event ev = node->sync.event.event;
  pocl_update_event_running (ev);
  const char *err = run (node);
  if (err)
    {
      POCL_MSG_ERR ("%s\n", err);
      POCL_UPDATE_EVENT_FAILED_MSG (CL_FAILED, ev, "tgpu NDRange          ");
    }
  else
    POCL_UPDATE_EVENT_COMPLETE_MSG (ev, "tgpu NDRange          ");
}

/* 命令的收发照 cpu-minimal：芯片一次只跑一个内核，就绪一条执行一条 */
static void
schedule (tgpu_data *d)
{
  _cl_command_node *node;
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready (node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      exec (node);
      POCL_LOCK (d->cq_lock);
    }
}

static void
tgpu_submit (_cl_command_node *node, cl_command_queue cq)
{
  tgpu_data *d = node->device->data;
  node->state = POCL_COMMAND_READY;
  POCL_LOCK (d->cq_lock);
  pocl_command_push (node, &d->ready_list, &d->command_list);
  POCL_UNLOCK_OBJ (node->sync.event.event);
  schedule (d);
  POCL_UNLOCK (d->cq_lock);
}

static void
tgpu_join (cl_device_id device, cl_command_queue cq)
{
  tgpu_data *d = device->data;
  POCL_LOCK (d->cq_lock);
  schedule (d);
  POCL_UNLOCK (d->cq_lock);
}

static void
tgpu_notify (cl_device_id device, cl_event event, cl_event finished)
{
  tgpu_data *d = device->data;
  _cl_command_node *volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_unlock_events_inorder (event, finished);
      pocl_update_event_failed (CL_FAILED, NULL, 0, event, NULL);
      pocl_lock_events_inorder (finished, event);
      return;
    }
  if (node->state != POCL_COMMAND_READY || !pocl_command_is_ready (event)
      || event->status != CL_QUEUED)
    return;
  pocl_update_event_submitted (event);
  POCL_LOCK (d->cq_lock);
  CDL_DELETE (d->command_list, node);
  CDL_PREPEND (d->ready_list, node);
  POCL_UNLOCK_OBJ (event);
  schedule (d);
  POCL_LOCK_OBJ (event);
  POCL_UNLOCK (d->cq_lock);
}

static int
tgpu_supports_binary (cl_device_id device, size_t length, const char *binary)
{
  return tgk_ok ((const uint8_t *)binary, length);
}

static int
tgpu_build_binary (cl_program program, cl_uint device_i, int link_program,
                   int spir_build)
{
  return tgk_ok (program->binaries[device_i], program->binary_sizes[device_i])
             ? CL_SUCCESS
             : CL_INVALID_BINARY;
}

static int
tgpu_build_builtin (cl_program program, cl_uint device_i)
{
  return CL_SUCCESS;
}

/* .tgk 里只有一个内核，叫 main，参数是 __global uchar *mem */
static int
tgpu_setup_metadata (cl_device_id device, cl_program program,
                     unsigned program_device_i)
{
  if (program->builtin_kernel_names)
    return pocl_setup_builtin_metadata (device, program, program_device_i);
  program->num_kernels = 1;
  program->kernel_meta = calloc (1, sizeof (pocl_kernel_metadata_t));
  pocl_kernel_metadata_t *m = program->kernel_meta;
  m->name = strdup ("main");
  m->num_args = 1;
  m->arg_info = calloc (1, sizeof (struct pocl_argument_info));
  m->arg_info[0] = (struct pocl_argument_info){
    .type_name = strdup ("uchar*"),
    .name = strdup ("mem"),
    .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL,
    .access_qualifier = CL_KERNEL_ARG_ACCESS_NONE,
    .type_qualifier = CL_KERNEL_ARG_TYPE_NONE,
    .type = POCL_ARG_TYPE_POINTER,
    .type_size = sizeof (cl_mem),
  };
  m->has_arg_metadata = POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER
                        | POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER
                        | POCL_HAS_KERNEL_ARG_TYPE_NAME
                        | POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER
                        | POCL_HAS_KERNEL_ARG_NAME;
  m->data = calloc (program->num_devices, sizeof (void *));
  return 1;
}

static void
tgpu_local_size (cl_device_id dev, cl_kernel kernel, unsigned device_i,
                 size_t max_group_size, size_t gx, size_t gy, size_t gz,
                 size_t *lx, size_t *ly, size_t *lz)
{
  tgpu_data *d = dev->data;
  *lx = gx % d->tpb ? 1 : d->tpb;
  *ly = *lz = 1;
}

static char *
tgpu_build_hash (cl_device_id device)
{
  return strdup ("tgpu");
}

static unsigned
tgpu_probe (struct pocl_device_ops *ops)
{
  int n = pocl_device_get_env_count (ops->device_name);
  return n < 0 ? 0 : n;
}

static cl_int
tgpu_init (unsigned j, cl_device_id dev, const char *parameters)
{
  if (!parameters)
    {
      POCL_MSG_ERR ("tgpu：POCL_TGPU%u_PARAMETERS 要给管理口，"
                    "spidev:/dev/spidev0.0 或 tcp:主机:端口\n",
                    j);
      return CL_INVALID_DEVICE;
    }
  tgpu_data *d = calloc (1, sizeof *d);
  if (!d)
    return CL_OUT_OF_HOST_MEMORY;
  uint32_t w[2] = { 0 };
  if (!(d->l = wire_open (parameters)) || spis_rd (d->l, BASE + ID, w, 2)
      || w[0] != TGPU_ID)
    {
      POCL_MSG_ERR ("tgpu：%s 上的标识读出 0x%08x，不是 to2610-gpu\n",
                    parameters, w[0]);
      wire_close (d->l);
      free (d);
      return CL_INVALID_DEVICE;
    }
  d->cores = w[1] >> 8 & 0xFF;
  d->tpb = w[1] & 0xFF;

  pocl_init_default_device_infos (dev, "");
  SETUP_DEVICE_CL_VERSION (dev, 1, 2);
  dev->type = CL_DEVICE_TYPE_ACCELERATOR;
  dev->long_name = "to2610-gpu";
  dev->short_name = "tgpu";
  dev->vendor = "Tape-Out";
  dev->profile = "EMBEDDED_PROFILE";
  dev->max_compute_units = d->cores;
  dev->max_work_group_size = dev->max_work_item_sizes[0] = d->tpb;
  dev->max_work_item_sizes[1] = dev->max_work_item_sizes[2] = 1;
  dev->preferred_wg_size_multiple = d->tpb;
  dev->global_mem_size = dev->max_mem_alloc_size = 1 << 20;
  dev->image_support = CL_FALSE;
  dev->compiler_available = dev->linker_available = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->builtin_kernel_list = strdup ("pocl.add.i8;pocl.mul.i8;pocl.copy.i8");
  dev->num_builtin_kernels = 3;
  pocl_setup_builtin_kernels_with_version (dev);

  d->available = CL_TRUE;
  dev->available = &d->available;
  dev->data = d;
  POCL_INIT_LOCK (d->cq_lock);
  return CL_SUCCESS;
}

static cl_int
tgpu_uninit (unsigned j, cl_device_id dev)
{
  tgpu_data *d = dev->data;
  POCL_DESTROY_LOCK (d->cq_lock);
  wire_close (d->l);
  free (d);
  dev->data = NULL;
  return CL_SUCCESS;
}

void
pocl_tgpu_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "tgpu";
  ops->probe = tgpu_probe;
  ops->init = tgpu_init;
  ops->uninit = tgpu_uninit;
  ops->build_hash = tgpu_build_hash;

  ops->alloc_mem_obj = pocl_driver_alloc_mem_obj;
  ops->free = pocl_driver_free;
  ops->read = pocl_driver_read;
  ops->read_rect = pocl_driver_read_rect;
  ops->write = pocl_driver_write;
  ops->write_rect = pocl_driver_write_rect;
  ops->copy = pocl_driver_copy;
  ops->copy_with_size = pocl_driver_copy_with_size;
  ops->copy_rect = pocl_driver_copy_rect;
  ops->memfill = pocl_driver_memfill;
  ops->map_mem = pocl_driver_map_mem;
  ops->unmap_mem = pocl_driver_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->supports_binary = tgpu_supports_binary;
  ops->build_binary = tgpu_build_binary;
  ops->build_builtin = tgpu_build_builtin;
  ops->setup_metadata = tgpu_setup_metadata;
  ops->free_program = pocl_driver_free_program;
  ops->compute_local_size = tgpu_local_size;

  ops->submit = tgpu_submit;
  ops->join = tgpu_join;
  ops->flush = tgpu_join;
  ops->notify = tgpu_notify;
  ops->broadcast = pocl_broadcast;
}
