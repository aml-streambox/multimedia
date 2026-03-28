/*
 * vfmcap_vulkan.c - Vulkan compute converter for AMLY -> P010/NV12
 *
 * Ported from amlvenc yuv422_converter_vulkan.c.
 * Key changes from amlvenc version:
 *   - Three shader pipelines: P010, NV12 SDR (no HDR), NV12 HDR (PQ→SDR tone map)
 *   - No wave_swap() — standard P010/NV12 output
 *   - Runtime pipeline selection via format parameter
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <stdbool.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>

#include "vfmcap_vulkan.h"

/* ---------- Configuration ---------- */

#ifndef VFMCAP_VK_DEBUG
#define VFMCAP_VK_DEBUG 0
#endif

#define DMABUF_CACHE_SIZE 8

#define VK_CHECK(result, msg) do { \
    if (result != VK_SUCCESS) { \
        snprintf(ctx.last_error, sizeof(ctx.last_error), "%s: %d", msg, result); \
        fprintf(stderr, "[vfmcap-vk] ERROR: %s\n", ctx.last_error); \
        return -1; \
    } \
} while(0)

/* ---------- Embedded SPIR-V shaders ---------- */

#include "../shaders/amly_to_p010_spv.h"
#include "../shaders/amly_to_nv12_spv.h"
#include "../shaders/amly_to_nv12_hdr_spv.h"

/* ---------- DMA-buf import cache ---------- */

typedef struct {
    int             fd;
    int             fd_dup;
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceSize    size;
    int             valid;
    uint64_t        last_used;
} DmabufCacheEntry;

/* ---------- Vulkan context ---------- */

typedef struct {
    VkInstance              instance;
    VkPhysicalDevice        physical_device;
    VkDevice                device;
    VkQueue                 compute_queue;
    uint32_t                compute_queue_family;
    VkCommandPool           command_pool;
    VkDescriptorPool        descriptor_pool;
    VkDescriptorSetLayout   descriptor_set_layout;
    VkDescriptorSet         descriptor_set;
    VkCommandBuffer         command_buffer;
    VkFence                 fence;
    VkPipelineLayout        pipeline_layout;

    /* Three pipelines: P010, NV12 SDR, NV12 HDR */
    VkPipeline              pipeline_p010;
    VkPipeline              pipeline_nv12;
    VkPipeline              pipeline_nv12_hdr;
    VkShaderModule          shader_p010;
    VkShaderModule          shader_nv12;
    VkShaderModule          shader_nv12_hdr;

    VkPhysicalDeviceMemoryProperties memory_props;

    uint32_t                width;
    uint32_t                height;
    int                     initialized;
    uint64_t                frame_count;
    char                    last_error[256];

    /* Cached output import */
    DmabufCacheEntry        cached_output;

    /* Cached input imports */
    DmabufCacheEntry        input_cache[DMABUF_CACHE_SIZE];
    int                     input_cache_count;

    /* Pending state for async submit/wait */
    int                     pending_in_fd;
    int                     has_pending;
} VulkanCtx;

static VulkanCtx ctx = {0};

/* ---------- Helpers ---------- */

static int find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props)
{
    for (uint32_t i = 0; i < ctx.memory_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (ctx.memory_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return -1;
}

/* ---------- DMA-buf import ---------- */

static int import_dmabuf(int fd, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory)
{
    VkExternalMemoryBufferCreateInfo ext_mem_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext_mem_info,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkResult result = vkCreateBuffer(ctx.device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        close(fd);
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "vkCreateBuffer failed: %d", result);
        return -1;
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx.device, *buffer, &mem_reqs);

    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd,
    };

    VkDeviceSize alloc_size = mem_reqs.size;
    struct stat fd_stat;
    if (fstat(fd, &fd_stat) == 0 && (VkDeviceSize)fd_stat.st_size > alloc_size) {
        alloc_size = fd_stat.st_size;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = alloc_size,
        .memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, 0),
    };

    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No suitable memory type");
        vkDestroyBuffer(ctx.device, *buffer, NULL);
        return -1;
    }

    result = vkAllocateMemory(ctx.device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "vkAllocateMemory failed: %d", result);
        vkDestroyBuffer(ctx.device, *buffer, NULL);
        return -1;
    }

    result = vkBindBufferMemory(ctx.device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "vkBindBufferMemory failed: %d", result);
        vkFreeMemory(ctx.device, *memory, NULL);
        vkDestroyBuffer(ctx.device, *buffer, NULL);
        return -1;
    }

    return 0;
}

/* ---------- Cache management ---------- */

static void cache_entry_destroy(DmabufCacheEntry *entry)
{
    if (!entry->valid) return;
    vkDestroyBuffer(ctx.device, entry->buffer, NULL);
    vkFreeMemory(ctx.device, entry->memory, NULL);
    entry->valid = 0;
    entry->fd = -1;
    entry->fd_dup = -1;
}

static int input_cache_get(int fd, VkDeviceSize size)
{
    for (int i = 0; i < ctx.input_cache_count; i++) {
        if (ctx.input_cache[i].valid && ctx.input_cache[i].fd == fd &&
            ctx.input_cache[i].size == size) {
            ctx.input_cache[i].last_used = ctx.frame_count;
            return i;
        }
    }

    int slot = -1;
    if (ctx.input_cache_count < DMABUF_CACHE_SIZE) {
        slot = ctx.input_cache_count++;
    } else {
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
            if (ctx.input_cache[i].last_used < oldest) {
                oldest = ctx.input_cache[i].last_used;
                slot = i;
            }
        }
        cache_entry_destroy(&ctx.input_cache[slot]);
    }

    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "dup(input fd %d) failed: %s", fd, strerror(errno));
        return -1;
    }

    VkBuffer buffer;
    VkDeviceMemory memory;
    if (import_dmabuf(fd_dup, size, &buffer, &memory) != 0) {
        return -1;
    }

    ctx.input_cache[slot].fd = fd;
    ctx.input_cache[slot].fd_dup = fd_dup;
    ctx.input_cache[slot].buffer = buffer;
    ctx.input_cache[slot].memory = memory;
    ctx.input_cache[slot].size = size;
    ctx.input_cache[slot].valid = 1;
    ctx.input_cache[slot].last_used = ctx.frame_count;

    return slot;
}

static int output_cache_get(int fd, VkDeviceSize size)
{
    if (ctx.cached_output.valid && ctx.cached_output.fd == fd &&
        ctx.cached_output.size == size) {
        return 0;
    }

    if (ctx.cached_output.valid) {
        cache_entry_destroy(&ctx.cached_output);
    }

    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "dup(output fd %d) failed: %s", fd, strerror(errno));
        return -1;
    }

    VkBuffer buffer;
    VkDeviceMemory memory;
    if (import_dmabuf(fd_dup, size, &buffer, &memory) != 0) {
        return -1;
    }

    ctx.cached_output.fd = fd;
    ctx.cached_output.fd_dup = fd_dup;
    ctx.cached_output.buffer = buffer;
    ctx.cached_output.memory = memory;
    ctx.cached_output.size = size;
    ctx.cached_output.valid = 1;

    return 0;
}

/* ---------- Shader loading ---------- */

static int load_shader(const unsigned char *spv_data, size_t spv_size,
                       VkShaderModule *shader_module)
{
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode = (const uint32_t *)spv_data,
    };

    VkResult result = vkCreateShaderModule(ctx.device, &create_info,
                                           NULL, shader_module);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "vkCreateShaderModule failed: %d", result);
        return -1;
    }
    return 0;
}

/* ---------- Initialization ---------- */

int vfmcap_vk_init(uint32_t width, uint32_t height)
{
    if (ctx.initialized) return 0;

    ctx.width = width;
    ctx.height = height;
    ctx.input_cache_count = 0;
    ctx.cached_output.valid = 0;
    ctx.cached_output.fd = -1;
    ctx.pending_in_fd = -1;
    ctx.has_pending = 0;

    for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
        ctx.input_cache[i].valid = 0;
        ctx.input_cache[i].fd = -1;
    }

    VkResult result;

    /* Instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VfmCapConverter",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "VulkanCompute",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    const char *inst_exts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
    };

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = inst_exts,
    };

    result = vkCreateInstance(&instance_info, NULL, &ctx.instance);
    VK_CHECK(result, "vkCreateInstance");

    /* Physical device selection */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &dev_count, NULL);
    if (dev_count == 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No Vulkan devices");
        return -1;
    }

    VkPhysicalDevice *devices = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx.instance, &dev_count, devices);

    for (uint32_t i = 0; i < dev_count; i++) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, NULL);
        VkQueueFamilyProperties *qf_props = malloc(qf_count * sizeof(*qf_props));
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, qf_props);

        for (uint32_t j = 0; j < qf_count; j++) {
            if (qf_props[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx.physical_device = devices[i];
                ctx.compute_queue_family = j;
                break;
            }
        }
        free(qf_props);
        if (ctx.physical_device != VK_NULL_HANDLE) break;
    }
    free(devices);

    if (ctx.physical_device == VK_NULL_HANDLE) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "No device with compute support");
        return -1;
    }

    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &ctx.memory_props);

    /* Logical device */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    const char *dev_exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = 5,
        .ppEnabledExtensionNames = dev_exts,
    };

    result = vkCreateDevice(ctx.physical_device, &device_info, NULL, &ctx.device);
    VK_CHECK(result, "vkCreateDevice");

    vkGetDeviceQueue(ctx.device, ctx.compute_queue_family, 0, &ctx.compute_queue);

    /* Command pool */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    };

    result = vkCreateCommandPool(ctx.device, &pool_info, NULL, &ctx.command_pool);
    VK_CHECK(result, "vkCreateCommandPool");

    /* Command buffer */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    result = vkAllocateCommandBuffers(ctx.device, &cmd_alloc, &ctx.command_buffer);
    VK_CHECK(result, "vkAllocateCommandBuffers");

    /* Fence */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    result = vkCreateFence(ctx.device, &fence_info, NULL, &ctx.fence);
    VK_CHECK(result, "vkCreateFence");

    /* Descriptor pool */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes,
    };

    result = vkCreateDescriptorPool(ctx.device, &desc_pool_info, NULL,
                                    &ctx.descriptor_pool);
    VK_CHECK(result, "vkCreateDescriptorPool");

    /* Descriptor set layout: 3 storage buffers (input, Y output, UV output) */
    VkDescriptorSetLayoutBinding bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };

    result = vkCreateDescriptorSetLayout(ctx.device, &layout_info, NULL,
                                         &ctx.descriptor_set_layout);
    VK_CHECK(result, "vkCreateDescriptorSetLayout");

    /* Descriptor set */
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx.descriptor_set_layout,
    };

    result = vkAllocateDescriptorSets(ctx.device, &desc_alloc, &ctx.descriptor_set);
    VK_CHECK(result, "vkAllocateDescriptorSets");

    /* Pipeline layout: push constants = { width, height, pairs_per_row, hdr_mode } */
    VkPushConstantRange push_constant = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 4,
    };

    VkPipelineLayoutCreateInfo pl_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx.descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant,
    };

    result = vkCreatePipelineLayout(ctx.device, &pl_layout_info, NULL,
                                    &ctx.pipeline_layout);
    VK_CHECK(result, "vkCreatePipelineLayout");

    /* Load all shaders */
    if (load_shader(amly_to_p010_spv, sizeof(amly_to_p010_spv),
                    &ctx.shader_p010) != 0) {
        return -1;
    }
    if (load_shader(amly_to_nv12_spv, sizeof(amly_to_nv12_spv),
                    &ctx.shader_nv12) != 0) {
        return -1;
    }
    if (load_shader(amly_to_nv12_hdr_spv, sizeof(amly_to_nv12_hdr_spv),
                    &ctx.shader_nv12_hdr) != 0) {
        return -1;
    }

    /* Create P010 pipeline */
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = ctx.shader_p010,
            .pName = "main",
        },
        .layout = ctx.pipeline_layout,
    };

    result = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &ctx.pipeline_p010);
    VK_CHECK(result, "vkCreateComputePipelines(P010)");

    /* Create NV12 SDR pipeline */
    pipeline_info.stage.module = ctx.shader_nv12;
    result = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &ctx.pipeline_nv12);
    VK_CHECK(result, "vkCreateComputePipelines(NV12)");

    /* Create NV12 HDR pipeline */
    pipeline_info.stage.module = ctx.shader_nv12_hdr;
    result = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &ctx.pipeline_nv12_hdr);
    VK_CHECK(result, "vkCreateComputePipelines(NV12_HDR)");

    ctx.initialized = 1;
    ctx.frame_count = 0;
    fprintf(stderr, "[vfmcap-vk] Initialized: %ux%u, P010+NV12_SDR+NV12_HDR pipelines ready\n",
            width, height);

    return 0;
}

/* ---------- Async submit ---------- */

int vfmcap_vk_convert_submit(int in_fd, int out_fd, uint32_t width,
                             uint32_t height, vfmcap_vk_fmt_t fmt,
                             uint32_t hdr_mode)
{
    if (!ctx.initialized) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Not initialized");
        return -1;
    }

    VkResult result;

    /* Fence management */
    if (ctx.frame_count == 0) {
        vkResetFences(ctx.device, 1, &ctx.fence);
    }

    /* Calculate buffer sizes */
    VkDeviceSize input_size = (VkDeviceSize)width * height * 5 / 2;
    VkDeviceSize y_plane_size, uv_plane_size, output_size;

    if (fmt == VFMCAP_VK_FMT_P010) {
        y_plane_size = (VkDeviceSize)width * height * 2;  /* 16-bit per Y */
        uv_plane_size = (VkDeviceSize)width * height;     /* 16-bit U + 16-bit V, half height */
    } else {
        y_plane_size = (VkDeviceSize)width * height;      /* 8-bit per Y */
        uv_plane_size = (VkDeviceSize)width * height / 2; /* 8-bit U + 8-bit V, half height */
    }
    output_size = y_plane_size + uv_plane_size;

    /* Cached input import + DMA_BUF_SYNC */
    int in_idx = input_cache_get(in_fd, input_size);
    if (in_idx < 0) return -1;

    struct dma_buf_sync sync_start = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
    };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_start);

    VkBuffer in_buffer = ctx.input_cache[in_idx].buffer;

    /* Cached output import */
    if (output_cache_get(out_fd, output_size) != 0) return -1;
    VkBuffer out_buffer = ctx.cached_output.buffer;

    /* Record command buffer */
    VkCommandBuffer cmd = ctx.command_buffer;

    result = vkResetCommandPool(ctx.device, ctx.command_pool, 0);
    VK_CHECK(result, "vkResetCommandPool");

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    result = vkBeginCommandBuffer(cmd, &begin_info);
    VK_CHECK(result, "vkBeginCommandBuffer");

    /* Update descriptors */
    VkDescriptorBufferInfo buffer_infos[] = {
        { in_buffer, 0, input_size },
        { out_buffer, 0, y_plane_size },
        { out_buffer, y_plane_size, uv_plane_size },
    };

    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ctx.descriptor_set, .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ctx.descriptor_set, .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ctx.descriptor_set, .dstBinding = 2,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[2] },
    };

    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    /* Bind pipeline: P010, NV12 SDR, or NV12 HDR */
    VkPipeline pipeline;
    if (fmt == VFMCAP_VK_FMT_P010)
        pipeline = ctx.pipeline_p010;
    else if (hdr_mode != 0)
        pipeline = ctx.pipeline_nv12_hdr;
    else
        pipeline = ctx.pipeline_nv12;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx.pipeline_layout, 0, 1,
                            &ctx.descriptor_set, 0, NULL);

    /* Push constants */
    uint32_t pairs_per_row = width / 2;
    /* hdr_mode only applies to NV12 shader; P010 shader ignores 4th constant */
    uint32_t push_data[] = { width, height, pairs_per_row,
                             (fmt == VFMCAP_VK_FMT_NV12) ? hdr_mode : 0u };
    vkCmdPushConstants(cmd, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_data), push_data);

    /* 2D dispatch
     * P010:     local_size=(128,1,1), 1 pair/thread, 1 row/workgroup
     *           tile = 128 pairs, groups_x = ceil(pairs_per_row/128), groups_y = height
     * NV12 SDR: local_size=(64,1,1), 2 pairs/thread, 1 row/workgroup
     *           tile = 128 pairs, groups_x = ceil(pairs_per_row/128), groups_y = height
     * NV12 HDR: local_size=(128,1,1), 1 pair/thread, 1 row/workgroup
     *           tile = 128 pairs, groups_x = ceil(pairs_per_row/128), groups_y = height
     */
    uint32_t groups_x, groups_y;
    groups_x = (pairs_per_row + 127) / 128;
    groups_y = height;
    vkCmdDispatch(cmd, groups_x, groups_y, 1);

    /* Memory barrier */
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT,
    };

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT |
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);

    result = vkEndCommandBuffer(cmd);
    VK_CHECK(result, "vkEndCommandBuffer");

    /* Submit */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    result = vkQueueSubmit(ctx.compute_queue, 1, &submit_info, ctx.fence);
    VK_CHECK(result, "vkQueueSubmit");

    ctx.pending_in_fd = in_fd;
    ctx.has_pending = 1;

    return 0;
}

/* ---------- Wait ---------- */

int vfmcap_vk_convert_wait(void)
{
    if (!ctx.initialized) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Not initialized");
        return -1;
    }

    if (!ctx.has_pending) return 0;

    VkResult result = vkWaitForFences(ctx.device, 1, &ctx.fence,
                                      VK_TRUE, 5000000000ULL);

    /* Release DMA-buf read access */
    if (ctx.pending_in_fd >= 0) {
        struct dma_buf_sync sync_end = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ
        };
        ioctl(ctx.pending_in_fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    }

    ctx.has_pending = 0;
    ctx.pending_in_fd = -1;

    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error),
                 "vkWaitForFences failed: %d (frame %lu)",
                 result, (unsigned long)ctx.frame_count);
        return -1;
    }

    vkResetFences(ctx.device, 1, &ctx.fence);
    ctx.frame_count++;

    return 0;
}

/* ---------- Synchronous conversion ---------- */

int vfmcap_vk_convert(int in_fd, int out_fd, uint32_t width,
                      uint32_t height, vfmcap_vk_fmt_t fmt,
                      uint32_t hdr_mode)
{
    int ret = vfmcap_vk_convert_submit(in_fd, out_fd, width, height, fmt, hdr_mode);
    if (ret != 0) return ret;
    return vfmcap_vk_convert_wait();
}

/* ---------- Cleanup ---------- */

void vfmcap_vk_cleanup(void)
{
    if (!ctx.initialized) return;

    vkDeviceWaitIdle(ctx.device);

    for (int i = 0; i < ctx.input_cache_count; i++) {
        cache_entry_destroy(&ctx.input_cache[i]);
    }
    ctx.input_cache_count = 0;

    cache_entry_destroy(&ctx.cached_output);

    if (ctx.fence != VK_NULL_HANDLE)
        vkDestroyFence(ctx.device, ctx.fence, NULL);
    if (ctx.pipeline_p010 != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx.device, ctx.pipeline_p010, NULL);
    if (ctx.pipeline_nv12 != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx.device, ctx.pipeline_nv12, NULL);
    if (ctx.pipeline_nv12_hdr != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx.device, ctx.pipeline_nv12_hdr, NULL);
    if (ctx.pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, NULL);
    if (ctx.shader_p010 != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx.device, ctx.shader_p010, NULL);
    if (ctx.shader_nv12 != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx.device, ctx.shader_nv12, NULL);
    if (ctx.shader_nv12_hdr != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx.device, ctx.shader_nv12_hdr, NULL);
    if (ctx.descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptor_set_layout, NULL);
    if (ctx.descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, NULL);
    if (ctx.command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(ctx.device, ctx.command_pool, NULL);
    if (ctx.device != VK_NULL_HANDLE)
        vkDestroyDevice(ctx.device, NULL);
    if (ctx.instance != VK_NULL_HANDLE)
        vkDestroyInstance(ctx.instance, NULL);

    memset(&ctx, 0, sizeof(ctx));
    fprintf(stderr, "[vfmcap-vk] Cleanup complete\n");
}

const char *vfmcap_vk_last_error(void)
{
    return ctx.last_error;
}
