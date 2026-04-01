/*
 * vfmcap_vulkan.c - Vulkan compute converter for AMLY -> P010/NV12
 *
 * Ported from amlvenc yuv422_converter_vulkan.c.
 * Key changes from amlvenc version:
 *   - Two shader pipelines: P010 (val << 6) and NV12 (val >> 2)
 *   - No wave_swap() — standard P010/NV12 output
 *   - No color space conversion — raw format packing only
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

#define DMABUF_CACHE_SIZE 2

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

/* ---------- DMA-buf import cache ---------- */

typedef struct {
    int             fd;
    int             fd_dup;
    ino_t           inode;      /* inode of the dma_buf file — unique per export */
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

    /* Two pipelines: P010 and NV12 */
    VkPipeline              pipeline_p010;
    VkPipeline              pipeline_nv12;
    VkShaderModule          shader_p010;
    VkShaderModule          shader_nv12;

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
    int                     pending_out_fd;
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
        close(fd);  /* ownership reverts to app on failure */
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

/**
 * get_fd_inode() - Get the inode number for a file descriptor.
 *
 * Each kernel dma_buf export creates a unique anon inode.  Two fds that
 * share the same inode refer to the same dma_buf (and therefore the same
 * physical CMA buffer).  If an fd number is reused after close+reopen,
 * the inode will differ, letting us detect stale cache entries.
 */
static ino_t get_fd_inode(int fd)
{
    struct stat st;
    if (fstat(fd, &st) == 0)
        return st.st_ino;
    return 0;
}

static int input_cache_get(int fd, VkDeviceSize size)
{
    ino_t fd_ino = get_fd_inode(fd);

    for (int i = 0; i < ctx.input_cache_count; i++) {
        if (ctx.input_cache[i].valid && ctx.input_cache[i].fd == fd &&
            ctx.input_cache[i].size == size) {
            /*
             * fd number matches — but does the underlying dma_buf?
             * After close(fd) + a new GET_DMABUF, the kernel can
             * reuse the same fd number for a completely different
             * CMA buffer.  Compare inodes to detect this.
             */
            if (fd_ino != 0 && ctx.input_cache[i].inode != 0 &&
                ctx.input_cache[i].inode != fd_ino) {
                /* Stale entry — destroy and re-import below */
                cache_entry_destroy(&ctx.input_cache[i]);
                /* Fall through to fresh import into this slot */
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
                ctx.input_cache[i].fd = fd;
                ctx.input_cache[i].fd_dup = fd_dup;
                ctx.input_cache[i].inode = fd_ino;
                ctx.input_cache[i].buffer = buffer;
                ctx.input_cache[i].memory = memory;
                ctx.input_cache[i].size = size;
                ctx.input_cache[i].valid = 1;
                ctx.input_cache[i].last_used = ctx.frame_count;
                return i;
            }
            /* Same inode — genuine cache hit */
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
    ctx.input_cache[slot].inode = fd_ino;
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
        /* Validate inode to catch stale entries */
        ino_t cur_ino = get_fd_inode(fd);
        if (cur_ino == 0 || ctx.cached_output.inode == 0 ||
            cur_ino == ctx.cached_output.inode) {
            return 0;
        }
        /* Stale — fall through to re-import */
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
    ctx.cached_output.inode = get_fd_inode(fd);
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
    ctx.pending_out_fd = -1;
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

    /* Pipeline layout: push constants = { width, height, pairs_per_row, reserved } */
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

    /* Load shaders */
    if (load_shader(amly_to_p010_spv, sizeof(amly_to_p010_spv),
                    &ctx.shader_p010) != 0) {
        return -1;
    }
    if (load_shader(amly_to_nv12_spv, sizeof(amly_to_nv12_spv),
                    &ctx.shader_nv12) != 0) {
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

    /* Create NV12 pipeline */
    pipeline_info.stage.module = ctx.shader_nv12;
    result = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &ctx.pipeline_nv12);
    VK_CHECK(result, "vkCreateComputePipelines(NV12)");

    ctx.initialized = 1;
    ctx.frame_count = 0;
    fprintf(stderr, "[vfmcap-vk] Initialized: %ux%u, P010+NV12 pipelines ready\n",
            width, height);

    return 0;
}

/* ---------- Async submit ---------- */

int vfmcap_vk_convert_submit(int in_fd, int out_fd, uint32_t width,
                             uint32_t height, vfmcap_vk_fmt_t fmt)
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

    /* Cached output import + DMA_BUF_SYNC write-access bracket (start) */
    if (output_cache_get(out_fd, output_size) != 0) return -1;
    VkBuffer out_buffer = ctx.cached_output.buffer;

    struct dma_buf_sync out_sync_start = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE
    };
    ioctl(out_fd, DMA_BUF_IOCTL_SYNC, &out_sync_start);

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

    /* Bind pipeline */
    VkPipeline pipeline = (fmt == VFMCAP_VK_FMT_P010) ?
                          ctx.pipeline_p010 : ctx.pipeline_nv12;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx.pipeline_layout, 0, 1,
                            &ctx.descriptor_set, 0, NULL);

    /* Push constants: { width, height, pairs_per_row, reserved } */
    uint32_t pairs_per_row = width / 2;
    uint32_t push_data[] = { width, height, pairs_per_row, 0u };
    vkCmdPushConstants(cmd, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_data), push_data);

    /* 2D dispatch
     * P010: local_size=(128,1,1), 1 pair/thread, 1 row/workgroup
     *       tile = 128 pairs, groups_x = ceil(pairs_per_row/128), groups_y = height
     * NV12: local_size=(64,1,1), 2 pairs/thread, 1 row/workgroup
     *       tile = 128 pairs, groups_x = ceil(pairs_per_row/128), groups_y = height
     */
    uint32_t groups_x = (pairs_per_row + 127) / 128;
    uint32_t groups_y = height;
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
    ctx.pending_out_fd = out_fd;
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

    /* Release DMA-buf read access on input */
    if (ctx.pending_in_fd >= 0) {
        struct dma_buf_sync sync_end = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ
        };
        ioctl(ctx.pending_in_fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    }

    /*
     * Immediately destroy the Vulkan cache entry for the just-completed
     * input frame so that Mali releases its dma_buf reference (acquired
     * via dup'd fd -> vkAllocateMemory -> dma_buf_get inside Mali).
     *
     * After vkFreeMemory, Mali's actual dma_buf_put is DEFERRED to an
     * internal worker thread.  If the caller immediately does close(fd)
     * + QBUF (which drop the userspace and kernel dma_buf refs), the
     * CMA buffer won't be recycled to vdin0 until Mali's worker runs.
     * At 60fps this backlog accumulates and eventually exhausts vdin0's
     * buffer pool.
     *
     * To fix this, after destroying the cache entry we call
     * vkDeviceWaitIdle() to force Mali to flush all pending internal
     * work, including the deferred dma_buf_put.  This ensures the CMA
     * buffer can be recycled as soon as the caller releases the frame.
     *
     * This also serves as a strong flush barrier for the output buffer:
     * vkDeviceWaitIdle drains Mali's L2 cache and write-combine buffers,
     * ensuring GPU-written data is visible in RAM before we signal
     * DMA_BUF_SYNC_END to downstream DMA consumers (VPU encoder).
     */
    if (ctx.pending_in_fd >= 0) {
        for (int i = 0; i < ctx.input_cache_count; i++) {
            if (ctx.input_cache[i].valid &&
                ctx.input_cache[i].fd == ctx.pending_in_fd) {
                cache_entry_destroy(&ctx.input_cache[i]);
                break;
            }
        }
    }
    /* Force Mali to complete ALL deferred work — including L2/WC flush
     * for output writes and deferred dma_buf_put from vkFreeMemory */
    vkDeviceWaitIdle(ctx.device);

    /*
     * ARM DSB + ISB to ensure all outstanding AXI write transactions
     * from Mali have committed to the memory controller before we hand
     * the buffer to the VPU encoder.  On A311D2 (Cortex-A73/A53 +
     * Mali-G52), the interconnect may still have in-flight write beats
     * after vkDeviceWaitIdle returns.
     *
     * Follow with a 500µs sleep to allow any remaining write buffer
     * drain through the NoC.  This is a diagnostic measure to isolate
     * whether the remaining ~1% tearing is an interconnect timing issue.
     */
    __sync_synchronize();
    usleep(500);

    /* Release DMA-buf write access on output AFTER vkDeviceWaitIdle —
     * this guarantees Mali's write-combine buffers and L2 cache have
     * been flushed to RAM, so the kernel's cache maintenance (triggered
     * by SYNC_END) operates on coherent data visible to DMA consumers. */
    if (ctx.pending_out_fd >= 0) {
        struct dma_buf_sync sync_end_wr = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE
        };
        ioctl(ctx.pending_out_fd, DMA_BUF_IOCTL_SYNC, &sync_end_wr);
    }

    ctx.has_pending = 0;
    ctx.pending_in_fd = -1;
    ctx.pending_out_fd = -1;

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
                      uint32_t height, vfmcap_vk_fmt_t fmt)
{
    int ret = vfmcap_vk_convert_submit(in_fd, out_fd, width, height, fmt);
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
    if (ctx.pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, NULL);
    if (ctx.shader_p010 != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx.device, ctx.shader_p010, NULL);
    if (ctx.shader_nv12 != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx.device, ctx.shader_nv12, NULL);
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
