// 用 Vulkan 的图形管线把一组三角形画进 8 × 8 的帧缓冲（R8_UINT，清成 0），读回 64 字节写到标准输出，同 raster.asm 的帧缓冲。
// 用法：vkdraw <顶点着色器.spv> <片元着色器.spv> < 三角形，一行一个：x0 y0 x1 y1 x2 y2 颜色（像素坐标，像素中心在 .5）。
// 不剔除、不测深度，一次画完：管线保证按图元的次序写，后画的盖住先画的
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define W 8
#define MAXTRI 64

#define CHECK(x)                                                                 \
  do {                                                                           \
    VkResult r_ = (x);                                                           \
    if (r_ != VK_SUCCESS) {                                                      \
      fprintf(stderr, "vkdraw：%s 返回 %d\n", #x, r_);                           \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

typedef struct {
  float x, y;
  uint32_t color;
} Vertex;

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

static uint32_t memtype(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  fputs("vkdraw：没有合用的存储\n", stderr);
  exit(1);
}

static VkShaderModule module(VkDevice dev, const char *path) {
  size_t len;
  uint32_t *code = slurp(path, &len);
  VkShaderModule m;
  CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = len, .pCode = code},
                             nullptr, &m));
  free(code);
  return m;
}

static VkDeviceMemory backing(VkDevice dev, VkPhysicalDevice phys, VkMemoryRequirements req, VkMemoryPropertyFlags want) {
  VkDeviceMemory m;
  CHECK(vkAllocateMemory(dev,
                         &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                                 .allocationSize = req.size,
                                                 .memoryTypeIndex = memtype(phys, req.memoryTypeBits, want)},
                         nullptr, &m));
  return m;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fputs("用法：vkdraw <顶点着色器.spv> <片元着色器.spv> < 三角形\n", stderr);
    return 2;
  }
  Vertex tri[MAXTRI * 3];
  uint32_t n = 0;
  float v[6];
  unsigned c;
  while (n < MAXTRI && scanf("%f %f %f %f %f %f %u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &c) == 7) {
    for (int k = 0; k < 3; k++) tri[n * 3 + k] = (Vertex){v[2 * k], v[2 * k + 1], c};
    n++;
  }

  VkInstance inst;
  CHECK(vkCreateInstance(&(VkInstanceCreateInfo){.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                                 .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                                                                          .pApplicationName = "vkdraw",
                                                                                          .apiVersion = VK_API_VERSION_1_3}},
                         nullptr, &inst));
  uint32_t nd = 16;
  VkPhysicalDevice devs[16];
  CHECK(vkEnumeratePhysicalDevices(inst, &nd, devs));
  if (!nd) {
    fputs("vkdraw：没有 Vulkan 设备\n", stderr);
    return 1;
  }
  VkPhysicalDevice phys = devs[0];
  for (uint32_t i = 0; i < nd; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) phys = devs[i];
  }
  VkPhysicalDeviceProperties pp;
  vkGetPhysicalDeviceProperties(phys, &pp);
  fprintf(stderr, "%s\n", pp.deviceName);

  uint32_t nq = 16, qf = UINT32_MAX;
  VkQueueFamilyProperties qs[16];
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qs);
  for (uint32_t i = 0; i < nq && qf == UINT32_MAX; i++)
    if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) qf = i;
  if (qf == UINT32_MAX) {
    fputs("vkdraw：设备没有图形队列\n", stderr);
    return 1;
  }
  VkDevice dev;
  CHECK(vkCreateDevice(phys,
                       &(VkDeviceCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext = &(VkPhysicalDeviceVulkan13Features){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                                                        .dynamicRendering = VK_TRUE,
                                                                        .synchronization2 = VK_TRUE},
                           .queueCreateInfoCount = 1,
                           .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                                           .queueFamilyIndex = qf,
                                                                           .queueCount = 1,
                                                                           .pQueuePriorities = &(float){1.0f}}},
                       nullptr, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, qf, 0, &queue);

  const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkBuffer vb, rb;
  CHECK(vkCreateBuffer(dev,
                       &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                             .size = sizeof tri,
                                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT},
                       nullptr, &vb));
  CHECK(vkCreateBuffer(dev,
                       &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                             .size = W * W,
                                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT},
                       nullptr, &rb));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, vb, &req);
  VkDeviceMemory vm = backing(dev, phys, req, host);
  CHECK(vkBindBufferMemory(dev, vb, vm, 0));
  vkGetBufferMemoryRequirements(dev, rb, &req);
  VkDeviceMemory rm = backing(dev, phys, req, host);
  CHECK(vkBindBufferMemory(dev, rb, rm, 0));
  void *map;
  CHECK(vkMapMemory(dev, vm, 0, sizeof tri, 0, &map));
  for (uint32_t i = 0; i < n * 3; i++) ((Vertex *)map)[i] = tri[i];
  vkUnmapMemory(dev, vm);

  VkImage img;
  CHECK(vkCreateImage(dev,
                      &(VkImageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                           .imageType = VK_IMAGE_TYPE_2D,
                                           .format = VK_FORMAT_R8_UINT,
                                           .extent = {W, W, 1},
                                           .mipLevels = 1,
                                           .arrayLayers = 1,
                                           .samples = VK_SAMPLE_COUNT_1_BIT,
                                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                                           .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
                      nullptr, &img));
  vkGetImageMemoryRequirements(dev, img, &req);
  VkDeviceMemory im = backing(dev, phys, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  CHECK(vkBindImageMemory(dev, img, im, 0));
  const VkImageSubresourceRange whole = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view;
  CHECK(vkCreateImageView(dev,
                          &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                   .image = img,
                                                   .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                                   .format = VK_FORMAT_R8_UINT,
                                                   .subresourceRange = whole},
                          nullptr, &view));

  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}, nullptr, &pl));
  VkShaderModule vs = module(dev, argv[1]), fs = module(dev, argv[2]);
  const VkFormat fmt = VK_FORMAT_R8_UINT;
  VkPipeline pipe;
  CHECK(vkCreateGraphicsPipelines(
      dev, VK_NULL_HANDLE, 1,
      &(VkGraphicsPipelineCreateInfo){
          .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
          .pNext = &(VkPipelineRenderingCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                    .colorAttachmentCount = 1,
                                                    .pColorAttachmentFormats = &fmt},
          .stageCount = 2,
          .pStages = (VkPipelineShaderStageCreateInfo[]){
              {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
              {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"}},
          .pVertexInputState =
              &(VkPipelineVertexInputStateCreateInfo){
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                  .vertexBindingDescriptionCount = 1,
                  .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX},
                  .vertexAttributeDescriptionCount = 2,
                  .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                                                                                        {1, 0, VK_FORMAT_R32_UINT, 8}}},
          .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                                           .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
          .pViewportState =
              &(VkPipelineViewportStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                   .viewportCount = 1,
                                                   .pViewports = &(VkViewport){0, 0, W, W, 0, 1},
                                                   .scissorCount = 1,
                                                   .pScissors = &(VkRect2D){{0, 0}, {W, W}}},
          .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                                           .polygonMode = VK_POLYGON_MODE_FILL,
                                                                           .cullMode = VK_CULL_MODE_NONE,
                                                                           .lineWidth = 1.0f},
          .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                                       .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
          .pColorBlendState =
              &(VkPipelineColorBlendStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                     .attachmentCount = 1,
                                                     .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = VK_COLOR_COMPONENT_R_BIT}},
          .layout = pl},
      nullptr, &pipe));

  VkCommandPool cp;
  CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf}, nullptr, &cp));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev,
                                 &(VkCommandBufferAllocateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                                .commandPool = cp,
                                                                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                                .commandBufferCount = 1},
                                 &cb));
  CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                             .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .imageMemoryBarrierCount = 1,
                                .pImageMemoryBarriers = &(VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                                                 .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                                                 .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                                                 .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                                                 .image = img,
                                                                                 .subresourceRange = whole}});
  vkCmdBeginRendering(cb, &(VkRenderingInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                             .renderArea = {{0, 0}, {W, W}},
                                             .layerCount = 1,
                                             .colorAttachmentCount = 1,
                                             .pColorAttachments = &(VkRenderingAttachmentInfo){
                                                 .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                                 .imageView = view,
                                                 .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                 .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                 .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                                 .clearValue = {.color = {.uint32 = {0}}}}});
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  vkCmdBindVertexBuffers(cb, 0, 1, &vb, &(VkDeviceSize){0});
  vkCmdDraw(cb, n * 3, 1, 0, 0);
  vkCmdEndRendering(cb);
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
                                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .imageMemoryBarrierCount = 1,
                                .pImageMemoryBarriers = &(VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                                                                 .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                                                 .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                                                 .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                                                                 .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                                                                 .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                                                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                                                 .image = img,
                                                                                 .subresourceRange = whole}});
  vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1,
                         &(VkBufferImageCopy){.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, W, 1}});
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                .memoryBarrierCount = 1,
                                                .pMemoryBarriers = &(VkMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                                                                       .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                                                                       .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                                                       .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                                                                                       .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT}});
  CHECK(vkEndCommandBuffer(cb));
  CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb},
                      VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(queue));

  CHECK(vkMapMemory(dev, rm, 0, W * W, 0, &map));
  fwrite(map, 1, W * W, stdout);
  vkUnmapMemory(dev, rm);

  vkDestroyCommandPool(dev, cp, nullptr);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyShaderModule(dev, vs, nullptr);
  vkDestroyShaderModule(dev, fs, nullptr);
  vkDestroyPipelineLayout(dev, pl, nullptr);
  vkDestroyImageView(dev, view, nullptr);
  vkDestroyImage(dev, img, nullptr);
  vkFreeMemory(dev, im, nullptr);
  vkDestroyBuffer(dev, vb, nullptr);
  vkDestroyBuffer(dev, rb, nullptr);
  vkFreeMemory(dev, vm, nullptr);
  vkFreeMemory(dev, rm, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  return 0;
}
