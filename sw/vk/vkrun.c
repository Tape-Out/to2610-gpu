// 在 Vulkan 设备上跑一个计算着色器：256 字节的数据存储从标准输入进、跑完从标准输出出，同 tiny-gpu 的数据存储。
// 用法：vkrun <着色器.spv> <线程数> < 数据 > 结果。设备名打到标准错误；有 CPU 类的设备（lavapipe）就用它
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define MEM 256

#define CHECK(x)                                                                 \
  do {                                                                           \
    VkResult r_ = (x);                                                           \
    if (r_ != VK_SUCCESS) {                                                      \
      fprintf(stderr, "vkrun：%s 返回 %d\n", #x, r_);                            \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

static uint32_t *slurp(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  *len = (size_t)ftell(f);
  rewind(f);
  uint32_t *p = malloc(*len);
  if (fread(p, 1, *len, f) != *len) {
    perror(path);
    exit(1);
  }
  fclose(f);
  return p;
}

static VkPhysicalDevice pick(VkInstance inst) {
  uint32_t n = 0;
  CHECK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
  if (!n) {
    fputs("vkrun：没有 Vulkan 设备\n", stderr);
    exit(1);
  }
  VkPhysicalDevice devs[16];
  n = n > 16 ? 16 : n;
  CHECK(vkEnumeratePhysicalDevices(inst, &n, devs));
  VkPhysicalDevice best = devs[0];
  for (uint32_t i = 0; i < n; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) best = devs[i];
  }
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(best, &p);
  fprintf(stderr, "%s\n", p.deviceName);
  return best;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fputs("用法：vkrun <着色器.spv> <线程数> < 数据 > 结果\n", stderr);
    return 2;
  }
  uint32_t threads = (uint32_t)strtoul(argv[2], nullptr, 0);
  uint8_t in[MEM] = {0};
  if (fread(in, 1, MEM, stdin) != MEM) {
    fputs("vkrun：标准输入要正好 256 字节\n", stderr);
    return 2;
  }

  VkInstance inst;
  CHECK(vkCreateInstance(&(VkInstanceCreateInfo){
                             .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                             .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                                                      .pApplicationName = "vkrun",
                                                                      .apiVersion = VK_API_VERSION_1_3},
                         },
                         nullptr, &inst));
  VkPhysicalDevice phys = pick(inst);

  uint32_t nq = 0, qf = UINT32_MAX;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
  VkQueueFamilyProperties qs[16];
  nq = nq > 16 ? 16 : nq;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qs);
  for (uint32_t i = 0; i < nq && qf == UINT32_MAX; i++)
    if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qf = i;
  if (qf == UINT32_MAX) {
    fputs("vkrun：设备没有计算队列\n", stderr);
    return 1;
  }

  VkDevice dev;
  CHECK(vkCreateDevice(phys,
                       &(VkDeviceCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .queueCreateInfoCount = 1,
                           .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                                           .queueFamilyIndex = qf,
                                                                           .queueCount = 1,
                                                                           .pQueuePriorities = &(float){1.0f}},
                       },
                       nullptr, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, qf, 0, &queue);

  // 一个字节放一个 uint：着色器里就是 uint mem[256]，与 tiny-gpu 一格一个 8 位数对得上
  const VkDeviceSize size = MEM * sizeof(uint32_t);
  VkBuffer buf;
  CHECK(vkCreateBuffer(dev,
                       &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                             .size = size,
                                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE},
                       nullptr, &buf));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, buf, &req);
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
    if ((req.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) type = i;
  if (type == UINT32_MAX) {
    fputs("vkrun：没有主机可见的存储\n", stderr);
    return 1;
  }
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(dev,
                         &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                                 .allocationSize = req.size,
                                                 .memoryTypeIndex = type},
                         nullptr, &mem));
  CHECK(vkBindBufferMemory(dev, buf, mem, 0));
  uint32_t *map;
  CHECK(vkMapMemory(dev, mem, 0, size, 0, (void **)&map));
  for (int i = 0; i < MEM; i++) map[i] = in[i];

  size_t len;
  uint32_t *code = slurp(argv[1], &len);
  VkShaderModule mod;
  CHECK(vkCreateShaderModule(dev,
                             &(VkShaderModuleCreateInfo){.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                                         .codeSize = len,
                                                         .pCode = code},
                             nullptr, &mod));

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev,
                                    &(VkDescriptorSetLayoutCreateInfo){
                                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                        .bindingCount = 1,
                                        .pBindings = &(VkDescriptorSetLayoutBinding){
                                            .binding = 0,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
                                    },
                                    nullptr, &dsl));
  const VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof threads};
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(dev,
                               &(VkPipelineLayoutCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                             .setLayoutCount = 1,
                                                             .pSetLayouts = &dsl,
                                                             .pushConstantRangeCount = 1,
                                                             .pPushConstantRanges = &pcr},
                               nullptr, &pl));
  VkPipeline pipe;
  CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1,
                                 &(VkComputePipelineCreateInfo){
                                     .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                     .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                               .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                               .module = mod,
                                               .pName = "main"},
                                     .layout = pl,
                                 },
                                 nullptr, &pipe));

  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(dev,
                               &(VkDescriptorPoolCreateInfo){
                                   .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                   .maxSets = 1,
                                   .poolSizeCount = 1,
                                   .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                               },
                               nullptr, &pool));
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(dev,
                                 &(VkDescriptorSetAllocateInfo){.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                                .descriptorPool = pool,
                                                                .descriptorSetCount = 1,
                                                                .pSetLayouts = &dsl},
                                 &set));
  vkUpdateDescriptorSets(dev, 1,
                         &(VkWriteDescriptorSet){
                             .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = set,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &(VkDescriptorBufferInfo){buf, 0, size},
                         },
                         0, nullptr);

  VkCommandPool cp;
  CHECK(vkCreateCommandPool(dev,
                            &(VkCommandPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                                       .queueFamilyIndex = qf},
                            nullptr, &cp));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev,
                                 &(VkCommandBufferAllocateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                                .commandPool = cp,
                                                                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                                .commandBufferCount = 1},
                                 &cb));
  CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                             .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof threads, &threads);
  vkCmdDispatch(cb, (threads + 63) / 64, 1, 1);
  // 着色器写完，主机才能读
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                       &(VkMemoryBarrier){.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT},
                       0, nullptr, 0, nullptr);
  CHECK(vkEndCommandBuffer(cb));
  CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb},
                      VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(queue));

  uint8_t out[MEM];
  for (int i = 0; i < MEM; i++) out[i] = (uint8_t)map[i];
  fwrite(out, 1, MEM, stdout);

  vkDestroyCommandPool(dev, cp, nullptr);
  vkDestroyDescriptorPool(dev, pool, nullptr);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyShaderModule(dev, mod, nullptr);
  vkUnmapMemory(dev, mem);
  vkFreeMemory(dev, mem, nullptr);
  vkDestroyBuffer(dev, buf, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  free(code);
  return 0;
}
