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
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <fcntl.h>

#include "vfmcap_vulkan.h"

/* ---------- DMA-buf heap allocation ---------- */

static int dmabuf_heap_alloc(size_t size)
{
    int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0) {
        heap_fd = open("/dev/dma_heap/heap-system", O_RDONLY | O_CLOEXEC);
        if (heap_fd < 0) {
            fprintf(stderr, "[vfmcap-vk] Failed to open dma_heap: %s\n", strerror(errno));
            return -1;
        }
    }

    struct dma_heap_allocation_data alloc_data = {
        .len = size,
        .fd = 0,
        .fd_flags = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
        fprintf(stderr, "[vfmcap-vk] DMA_HEAP_IOCTL_ALLOC failed: %s\n", strerror(errno));
        close(heap_fd);
        return -1;
    }

    close(heap_fd);
    return alloc_data.fd;
}

/* ---------- Dynamic rendering fallback for Vulkan < 1.3 headers ---------- */

#if VK_HEADER_VERSION < 197
#define VK_STRUCTURE_TYPE_RENDERING_INFO                  ((VkStructureType)1000044000)
#define VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO       ((VkStructureType)1000044001)
#define VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO  ((VkStructureType)1000044002)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES ((VkStructureType)1000044003)
#define VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME           "VK_KHR_dynamic_rendering"

typedef struct VkPipelineRenderingCreateInfo {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           viewMask;
    uint32_t           colorAttachmentCount;
    const VkFormat*    pColorAttachmentFormats;
    VkFormat           depthAttachmentFormat;
    VkFormat           stencilAttachmentFormat;
} VkPipelineRenderingCreateInfo;

typedef struct VkRenderingAttachmentInfo {
    VkStructureType          sType;
    const void*              pNext;
    VkImageView              imageView;
    VkImageLayout            imageLayout;
    VkResolveModeFlagBits    resolveMode;
    VkImageView              resolveImageView;
    VkImageLayout            resolveImageLayout;
    VkAttachmentLoadOp       loadOp;
    VkAttachmentStoreOp      storeOp;
    VkClearValue             clearValue;
} VkRenderingAttachmentInfo;

typedef struct VkRenderingInfo {
    VkStructureType                      sType;
    const void*                          pNext;
    uint32_t                             flags;
    VkRect2D                             renderArea;
    uint32_t                             layerCount;
    uint32_t                             viewMask;
    uint32_t                             colorAttachmentCount;
    const VkRenderingAttachmentInfo*     pColorAttachments;
    const VkRenderingAttachmentInfo*     pDepthAttachment;
    const VkRenderingAttachmentInfo*     pStencilAttachment;
} VkRenderingInfo;

typedef struct VkPhysicalDeviceDynamicRenderingFeatures {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           dynamicRendering;
} VkPhysicalDeviceDynamicRenderingFeatures;

typedef void (VKAPI_PTR *PFN_vkCmdBeginRenderingKHR)(VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo);
typedef void (VKAPI_PTR *PFN_vkCmdEndRenderingKHR)(VkCommandBuffer commandBuffer);
#else
/* Headers already have core dynamic rendering types; just need KHR fn typedefs */
typedef void (VKAPI_PTR *PFN_vkCmdBeginRenderingKHR)(VkCommandBuffer commandBuffer, const VkRenderingInfo* pRenderingInfo);
typedef void (VKAPI_PTR *PFN_vkCmdEndRenderingKHR)(VkCommandBuffer commandBuffer);
#endif /* VK_HEADER_VERSION < 197 */

/* ---------- Configuration ---------- */

#ifndef VFMCAP_VK_DEBUG
#define VFMCAP_VK_DEBUG 0
#endif

#define DMABUF_CACHE_SIZE 2

#define VK_CHECK(result, msg) do { \
    if (result != VK_SUCCESS) { \
        snprintf(vk->last_error, sizeof(vk->last_error), "%s: %d", msg, result); \
        fprintf(stderr, "[vfmcap-vk] ERROR: %s\n", vk->last_error); \
        return -1; \
    } \
} while(0)

/* ---------- Embedded SPIR-V shaders ---------- */

#include "../shaders/amly_to_p010_spv.h"
#include "../shaders/amly_to_nv12_spv.h"
#include "../shaders/amly_decode_spv.h"
#include "../shaders/fullscreen.vert_spv.h"
#include "../shaders/nv12_y.frag_spv.h"
#include "../shaders/nv12_uv.frag_spv.h"
#include "../shaders/p010_y.frag_spv.h"
#include "../shaders/p010_uv.frag_spv.h"
#include "../shaders/nv12_y_from_r16.frag_spv.h"
#include "../shaders/nv12_uv_from_r16g16.frag_spv.h"
#include "../shaders/p010_y_from_r16.frag_spv.h"
#include "../shaders/p010_uv_from_r16g16.frag_spv.h"
#include "../shaders/yuv_from_yuva.frag_spv.h"

/* ---------- DMA-buf import cache (legacy VkBuffer for compute) ---------- */

typedef struct {
    int             fd;
    int             fd_dup;
    ino_t           inode;
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceSize    size;
    int             valid;
    uint64_t        last_used;
} DmabufCacheEntry;

/* ---------- DMA-buf VkImage import cache (graphics path) ---------- */

typedef struct {
    int             fd;
    int             fd_dup;
    ino_t           inode;
    VkImage         image;
    VkDeviceMemory  memory;
    VkImageView     view;
    uint32_t        width;
    uint32_t        height;
    VkFormat        format;
    uint64_t        modifier;
    int             valid;
    uint64_t        last_used;
} ImageCacheEntry;

/* ---------- Output pool types (must be defined before VulkanCtx) ---------- */

typedef struct {
    VkImage         image;
    VkDeviceMemory  memory;
    VkImageView     view;
    int             dmabuf_fd;      /* DMA-buf fd for image (dmabuf pool) or buffer (copy pool) */
    int             in_use;
    int             is_internal;    /* 1 = pure GPU-local image, no DMA-buf */
    /* Copy-out fields: internal image + DMA-buf backed buffer for formats
       that Mali cannot export as external VkImages (R16, R16G16). */
    VkBuffer        copy_buffer;
    VkDeviceMemory  copy_buffer_memory;
    VkDeviceSize    copy_buffer_size;
    int             has_copy_buffer; /* 1 = uses copy-out path */
} OutputPoolEntry;

typedef struct {
    OutputPoolEntry *entries;
    int              count;
    int              capacity;
    uint32_t         width;
    uint32_t         height;
    VkFormat         format;
} OutputPool;

/* ---------- Vulkan context ---------- */

struct VulkanCtx {
    VkInstance              instance;
    VkPhysicalDevice        physical_device;
    VkDevice                device;
    VkQueue                 compute_queue;
    VkQueue                 graphics_queue;
    uint32_t                compute_queue_family;
    uint32_t                graphics_queue_family;
    VkCommandPool           command_pool;
    VkDescriptorPool        descriptor_pool;
    VkDescriptorSetLayout   descriptor_set_layout;
    VkDescriptorSet         descriptor_set;
    VkDescriptorSetLayout   gfx_descriptor_set_layout;
    VkDescriptorSetLayout   gfx_descriptor_set_layout_10bit;
    VkDescriptorSet         gfx_descriptor_set;
    VkDescriptorSet         gfx_descriptor_set_10bit;
    VkCommandBuffer         command_buffer;
    VkFence                 fence;
    VkPipelineLayout        pipeline_layout;
    VkPipelineLayout        pipeline_layout_decode;
    VkPipelineLayout        gfx_pipeline_layout;
    VkPipelineLayout        gfx_pipeline_layout_10bit;

    /* Legacy compute pipelines */
    VkPipeline              pipeline_p010;
    VkPipeline              pipeline_nv12;
    VkShaderModule          shader_p010;
    VkShaderModule          shader_nv12;

    /* 10-bit compute-to-graphics decode pipeline (AMLY -> R16 Y + R16G16 UV) */
    VkPipeline              pipeline_amly_decode;
    VkShaderModule          shader_amly_decode;
    VkDescriptorSetLayout   compute_descriptor_set_layout_decode;
    VkDescriptorSet         compute_descriptor_set_decode;

    /* Graphics pipelines (dynamic rendering) for 8-bit Ycbcr path */
    VkPipeline              gfx_pipeline_nv12_y;
    VkPipeline              gfx_pipeline_nv12_uv;
    VkPipeline              gfx_pipeline_p010_y;
    VkPipeline              gfx_pipeline_p010_uv;

    /* Graphics pipelines (dynamic rendering) for 10-bit two-plane sampler path */
    VkPipeline              gfx_pipeline_nv12_y_10bit;
    VkPipeline              gfx_pipeline_nv12_uv_10bit;
    VkPipeline              gfx_pipeline_p010_y_10bit;
    VkPipeline              gfx_pipeline_p010_uv_10bit;

    VkShaderModule          gfx_shader_vert;
    VkShaderModule          gfx_shader_nv12_y;
    VkShaderModule          gfx_shader_nv12_uv;
    VkShaderModule          gfx_shader_p010_y;
    VkShaderModule          gfx_shader_p010_uv;
    VkShaderModule          gfx_shader_nv12_y_from_r16;
    VkShaderModule          gfx_shader_nv12_uv_from_r16g16;
    VkShaderModule          gfx_shader_p010_y_from_r16;
    VkShaderModule          gfx_shader_p010_uv_from_r16g16;

    /* Ycbcr sampler for NV12/NV21 input images */
    VkSampler               ycbcr_sampler;
    VkSamplerYcbcrConversion ycbcr_conversion;

    /* Regular linear sampler for 10-bit intermediate */
    VkSampler               regular_sampler;

    /* Dynamic rendering function pointers (for headers < 1.3) */
    PFN_vkCmdBeginRenderingKHR cmd_begin_rendering;
    PFN_vkCmdEndRenderingKHR   cmd_end_rendering;

    VkPhysicalDeviceMemoryProperties memory_props;

    uint32_t                width;
    uint32_t                height;
    int                     initialized;
    uint64_t                frame_count;
    char                    last_error[256];

    /* Cached output import (legacy compute) */
    DmabufCacheEntry        cached_output;

    /* Cached input imports (legacy compute) */
    DmabufCacheEntry        input_cache[DMABUF_CACHE_SIZE];
    int                     input_cache_count;

    /* Cached image imports (graphics path) */
    ImageCacheEntry         image_cache[DMABUF_CACHE_SIZE];
    int                     image_cache_count;

    /* Output pools for graphics rendering */
    OutputPool              pool_nv12_y;
    OutputPool              pool_nv12_uv;
    OutputPool              pool_p010_y;
    OutputPool              pool_p010_uv;
    OutputPool              pool_nv12_afbc;
    OutputPool              pool_a2b10g10r10_afbc;

    /* Intermediate pools for 10-bit compute->graphics chain */
    OutputPool              pool_intermediate_y;
    OutputPool              pool_intermediate_uv;

    /* Pending state for async submit/wait */
    int                     pending_in_fd;
    int                     pending_out_fd;
    int                     pending_out_fd2;
    int                     has_pending;
};

/* ---------- Helpers ---------- */

static int find_memory_type(VulkanCtx *vk, uint32_t type_filter, VkMemoryPropertyFlags props)
{
    for (uint32_t i = 0; i < vk->memory_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (vk->memory_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return -1;
}

/* ---------- DMA-buf import ---------- */

static int import_dmabuf(VulkanCtx *vk, int fd, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory)
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

    VkResult result = vkCreateBuffer(vk->device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        close(fd);
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateBuffer failed: %d", result);
        return -1;
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(vk->device, *buffer, &mem_reqs);

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
        .memoryTypeIndex = find_memory_type(vk, mem_reqs.memoryTypeBits, 0),
    };

    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        snprintf(vk->last_error, sizeof(vk->last_error), "No suitable memory type");
        vkDestroyBuffer(vk->device, *buffer, NULL);
        return -1;
    }

    result = vkAllocateMemory(vk->device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkAllocateMemory failed: %d", result);
        vkDestroyBuffer(vk->device, *buffer, NULL);
        close(fd);  /* ownership reverts to app on failure */
        return -1;
    }

    result = vkBindBufferMemory(vk->device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkBindBufferMemory failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        vkDestroyBuffer(vk->device, *buffer, NULL);
        return -1;
    }

    return 0;
}

/* ---------- DMA-buf VkImage import ---------- */

static int import_dmabuf_image(VulkanCtx *vk, int fd, uint32_t width, uint32_t height,
                               VkFormat format, uint64_t modifier,
                               VkImage *image, VkDeviceMemory *memory, VkImageView *view)
{
    VkExternalMemoryImageCreateInfo ext_mem_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &ext_mem_info,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = (modifier != 0) ? VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT : VK_IMAGE_TILING_LINEAR,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkImageDrmFormatModifierExplicitCreateInfoEXT modifier_info = {0};
    uint32_t drm_plane_count = 1;
    VkSubresourceLayout plane_layouts[4] = {0};

    if (modifier != 0) {
        modifier_info.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT;
        modifier_info.drmFormatModifier = modifier;
        modifier_info.drmFormatModifierPlaneCount = drm_plane_count;
        modifier_info.pPlaneLayouts = plane_layouts;
        image_info.pNext = &modifier_info;
    }

    VkResult result = vkCreateImage(vk->device, &image_info, NULL, image);
    if (result != VK_SUCCESS) {
        close(fd);
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateImage failed: %d", result);
        return -1;
    }

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(vk->device, *image, &mem_reqs);

    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd,
    };

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(vk, mem_reqs.memoryTypeBits, 0),
    };

    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        snprintf(vk->last_error, sizeof(vk->last_error), "No suitable memory type for image");
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    result = vkAllocateMemory(vk->device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkAllocateMemory(image) failed: %d", result);
        vkDestroyImage(vk->device, *image, NULL);
        close(fd);
        return -1;
    }

    result = vkBindImageMemory(vk->device, *image, *memory, 0);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkBindImageMemory failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    result = vkCreateImageView(vk->device, &view_info, NULL, view);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateImageView failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    return 0;
}

/* ---------- DMA-buf VkImage creation via dma_heap ---------- */

static int create_dmabuf_image(VulkanCtx *vk, uint32_t width, uint32_t height,
                               VkFormat format, uint64_t modifier,
                               VkImage *image, VkDeviceMemory *memory,
                               VkImageView *view, int *out_fd)
{
    (void)modifier;
    VkExternalMemoryImageCreateInfo ext_mem_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };

    uint64_t mod_linear = 0;
    VkImageDrmFormatModifierListCreateInfoEXT modifier_list = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .pNext = &ext_mem_info,
        .drmFormatModifierCount = 1,
        .pDrmFormatModifiers = &mod_linear,
    };

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &modifier_list,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkResult result = vkCreateImage(vk->device, &image_info, NULL, image);

    if (result != VK_SUCCESS) {
        image_info.tiling = VK_IMAGE_TILING_LINEAR;
        image_info.pNext = &ext_mem_info;
        result = vkCreateImage(vk->device, &image_info, NULL, image);
    }

    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateImage failed: %d", result);
        return -1;
    }

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(vk->device, *image, &mem_reqs);

    /* Allocate DMA-buf from heap */
    int dmabuf_fd = dmabuf_heap_alloc(mem_reqs.size);
    if (dmabuf_fd < 0) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "dmabuf_heap_alloc failed");
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    /* Vulkan takes ownership of imported fds, so dup it for our own use */
    int fd_for_vulkan = dup(dmabuf_fd);
    if (fd_for_vulkan < 0) {
        close(dmabuf_fd);
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "dup(dmabuf_fd) failed: %s", strerror(errno));
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd_for_vulkan,
    };

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(vk, mem_reqs.memoryTypeBits, 0),
    };

    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        snprintf(vk->last_error, sizeof(vk->last_error), "No suitable memory type for image");
        close(dmabuf_fd);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    result = vkAllocateMemory(vk->device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkAllocateMemory(image) failed: %d", result);
        close(dmabuf_fd);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    result = vkBindImageMemory(vk->device, *image, *memory, 0);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkBindImageMemory failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        close(dmabuf_fd);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    result = vkCreateImageView(vk->device, &view_info, NULL, view);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateImageView failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        close(dmabuf_fd);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    *out_fd = dmabuf_fd;
    return 0;
}

/* ---------- Internal Vulkan image (no DMA-buf, for intermediates) ---------- */

static int create_internal_image(VulkanCtx *vk, uint32_t width, uint32_t height,
                                  VkFormat format, VkImageUsageFlags usage,
                                  VkImage *image, VkDeviceMemory *memory,
                                  VkImageView *view)
{
    /* Use OPTIMAL tiling for color attachments (required on Mali-G52 for
     * R16_UNORM / R16G16_UNORM render targets).  LINEAR tiling is used for
     * storage/sampled intermediates so the compute shader can imageStore(). */
    VkImageTiling tiling = (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                         ? VK_IMAGE_TILING_OPTIMAL
                         : VK_IMAGE_TILING_LINEAR;

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkResult result = vkCreateImage(vk->device, &image_info, NULL, image);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateImage(internal) failed: %d", result);
        return -1;
    }

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(vk->device, *image, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(vk, mem_reqs.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };

    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        alloc_info.memoryTypeIndex = find_memory_type(vk, mem_reqs.memoryTypeBits, 0);
    }
    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "No suitable memory type for internal image");
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    result = vkAllocateMemory(vk->device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkAllocateMemory(internal) failed: %d", result);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    result = vkBindImageMemory(vk->device, *image, *memory, 0);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkBindImageMemory(internal) failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    result = vkCreateImageView(vk->device, &view_info, NULL, view);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateImageView(internal) failed: %d", result);
        vkFreeMemory(vk->device, *memory, NULL);
        vkDestroyImage(vk->device, *image, NULL);
        return -1;
    }

    return 0;
}

/* ---------- Cache management ---------- */

static void cache_entry_destroy(VulkanCtx *vk, DmabufCacheEntry *entry)
{
    if (!entry->valid) return;
    vkDestroyBuffer(vk->device, entry->buffer, NULL);
    vkFreeMemory(vk->device, entry->memory, NULL);
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

static int input_cache_get(VulkanCtx *vk, int fd, VkDeviceSize size)
{
    ino_t fd_ino = get_fd_inode(fd);

    for (int i = 0; i < vk->input_cache_count; i++) {
        if (vk->input_cache[i].valid && vk->input_cache[i].fd == fd &&
            vk->input_cache[i].size == size) {
            /*
             * fd number matches — but does the underlying dma_buf?
             * After close(fd) + a new GET_DMABUF, the kernel can
             * reuse the same fd number for a completely different
             * CMA buffer.  Compare inodes to detect this.
             */
            if (fd_ino != 0 && vk->input_cache[i].inode != 0 &&
                vk->input_cache[i].inode != fd_ino) {
                /* Stale entry — destroy and re-import below */
                cache_entry_destroy(vk, &vk->input_cache[i]);
                /* Fall through to fresh import into this slot */
                int fd_dup = dup(fd);
                if (fd_dup < 0) {
                    snprintf(vk->last_error, sizeof(vk->last_error),
                             "dup(input fd %d) failed: %s", fd, strerror(errno));
                    return -1;
                }
                VkBuffer buffer;
                VkDeviceMemory memory;
                if (import_dmabuf(vk, fd_dup, size, &buffer, &memory) != 0) {
                    return -1;
                }
                vk->input_cache[i].fd = fd;
                vk->input_cache[i].fd_dup = fd_dup;
                vk->input_cache[i].inode = fd_ino;
                vk->input_cache[i].buffer = buffer;
                vk->input_cache[i].memory = memory;
                vk->input_cache[i].size = size;
                vk->input_cache[i].valid = 1;
                vk->input_cache[i].last_used = vk->frame_count;
                return i;
            }
            /* Same inode — genuine cache hit */
            vk->input_cache[i].last_used = vk->frame_count;
            return i;
        }
    }

    int slot = -1;
    if (vk->input_cache_count < DMABUF_CACHE_SIZE) {
        slot = vk->input_cache_count++;
    } else {
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
            if (vk->input_cache[i].last_used < oldest) {
                oldest = vk->input_cache[i].last_used;
                slot = i;
            }
        }
        cache_entry_destroy(vk, &vk->input_cache[slot]);
    }

    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "dup(input fd %d) failed: %s", fd, strerror(errno));
        return -1;
    }

    VkBuffer buffer;
    VkDeviceMemory memory;
    if (import_dmabuf(vk, fd_dup, size, &buffer, &memory) != 0) {
        return -1;
    }

    vk->input_cache[slot].fd = fd;
    vk->input_cache[slot].fd_dup = fd_dup;
    vk->input_cache[slot].inode = fd_ino;
    vk->input_cache[slot].buffer = buffer;
    vk->input_cache[slot].memory = memory;
    vk->input_cache[slot].size = size;
    vk->input_cache[slot].valid = 1;
    vk->input_cache[slot].last_used = vk->frame_count;

    return slot;
}

static int output_cache_get(VulkanCtx *vk, int fd, VkDeviceSize size)
{
    if (vk->cached_output.valid && vk->cached_output.fd == fd &&
        vk->cached_output.size == size) {
        /* Validate inode to catch stale entries */
        ino_t cur_ino = get_fd_inode(fd);
        if (cur_ino == 0 || vk->cached_output.inode == 0 ||
            cur_ino == vk->cached_output.inode) {
            return 0;
        }
        /* Stale — fall through to re-import */
    }

    if (vk->cached_output.valid) {
        cache_entry_destroy(vk, &vk->cached_output);
    }

    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "dup(output fd %d) failed: %s", fd, strerror(errno));
        return -1;
    }

    VkBuffer buffer;
    VkDeviceMemory memory;
    if (import_dmabuf(vk, fd_dup, size, &buffer, &memory) != 0) {
        return -1;
    }

    vk->cached_output.fd = fd;
    vk->cached_output.fd_dup = fd_dup;
    vk->cached_output.inode = get_fd_inode(fd);
    vk->cached_output.buffer = buffer;
    vk->cached_output.memory = memory;
    vk->cached_output.size = size;
    vk->cached_output.valid = 1;

    return 0;
}

/* ---------- Image cache management ---------- */

static void image_cache_entry_destroy(VulkanCtx *vk, ImageCacheEntry *entry)
{
    if (!entry || !entry->valid) return;
    if (entry->view != VK_NULL_HANDLE)
        vkDestroyImageView(vk->device, entry->view, NULL);
    if (entry->image != VK_NULL_HANDLE)
        vkDestroyImage(vk->device, entry->image, NULL);
    if (entry->memory != VK_NULL_HANDLE)
        vkFreeMemory(vk->device, entry->memory, NULL);
    entry->valid = 0;
    entry->fd = -1;
    entry->fd_dup = -1;
}

static int image_cache_get(VulkanCtx *vk, int fd, uint32_t width, uint32_t height,
                           VkFormat format, uint64_t modifier,
                           VkImage *image, VkImageView *view)
{
    ino_t fd_ino = get_fd_inode(fd);

    for (int i = 0; i < vk->image_cache_count; i++) {
        if (vk->image_cache[i].valid && vk->image_cache[i].fd == fd &&
            vk->image_cache[i].width == width &&
            vk->image_cache[i].height == height &&
            vk->image_cache[i].format == format &&
            vk->image_cache[i].modifier == modifier) {
            if (fd_ino != 0 && vk->image_cache[i].inode != 0 &&
                vk->image_cache[i].inode != fd_ino) {
                image_cache_entry_destroy(vk, &vk->image_cache[i]);
                int fd_dup = dup(fd);
                if (fd_dup < 0) {
                    snprintf(vk->last_error, sizeof(vk->last_error),
                             "dup(image fd %d) failed: %s", fd, strerror(errno));
                    return -1;
                }
                if (import_dmabuf_image(vk, fd_dup, width, height, format, modifier,
                                        &vk->image_cache[i].image,
                                        &vk->image_cache[i].memory,
                                        &vk->image_cache[i].view) != 0) {
                    return -1;
                }
                vk->image_cache[i].fd = fd;
                vk->image_cache[i].fd_dup = fd_dup;
                vk->image_cache[i].inode = fd_ino;
                vk->image_cache[i].width = width;
                vk->image_cache[i].height = height;
                vk->image_cache[i].format = format;
                vk->image_cache[i].modifier = modifier;
                vk->image_cache[i].valid = 1;
                vk->image_cache[i].last_used = vk->frame_count;
                *image = vk->image_cache[i].image;
                *view = vk->image_cache[i].view;
                return 0;
            }
            vk->image_cache[i].last_used = vk->frame_count;
            *image = vk->image_cache[i].image;
            *view = vk->image_cache[i].view;
            return 0;
        }
    }

    int slot = -1;
    if (vk->image_cache_count < DMABUF_CACHE_SIZE) {
        slot = vk->image_cache_count++;
    } else {
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
            if (vk->image_cache[i].last_used < oldest) {
                oldest = vk->image_cache[i].last_used;
                slot = i;
            }
        }
        image_cache_entry_destroy(vk, &vk->image_cache[slot]);
    }

    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "dup(image fd %d) failed: %s", fd, strerror(errno));
        return -1;
    }

    if (import_dmabuf_image(vk, fd_dup, width, height, format, modifier,
                            &vk->image_cache[slot].image,
                            &vk->image_cache[slot].memory,
                            &vk->image_cache[slot].view) != 0) {
        return -1;
    }

    vk->image_cache[slot].fd = fd;
    vk->image_cache[slot].fd_dup = fd_dup;
    vk->image_cache[slot].inode = fd_ino;
    vk->image_cache[slot].width = width;
    vk->image_cache[slot].height = height;
    vk->image_cache[slot].format = format;
    vk->image_cache[slot].modifier = modifier;
    vk->image_cache[slot].valid = 1;
    vk->image_cache[slot].last_used = vk->frame_count;

    *image = vk->image_cache[slot].image;
    *view = vk->image_cache[slot].view;
    return 0;
}

/* ---------- Output pool management ---------- */

static void output_pool_destroy(VulkanCtx *vk, OutputPool *pool)
{
    if (!pool) return;
    for (int i = 0; i < pool->count; i++) {
        OutputPoolEntry *e = &pool->entries[i];
        if (e->view != VK_NULL_HANDLE)
            vkDestroyImageView(vk->device, e->view, NULL);
        if (e->image != VK_NULL_HANDLE)
            vkDestroyImage(vk->device, e->image, NULL);
        if (e->memory != VK_NULL_HANDLE)
            vkFreeMemory(vk->device, e->memory, NULL);
        if (e->copy_buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(vk->device, e->copy_buffer, NULL);
        if (e->copy_buffer_memory != VK_NULL_HANDLE)
            vkFreeMemory(vk->device, e->copy_buffer_memory, NULL);
        if (e->dmabuf_fd >= 0)
            close(e->dmabuf_fd);
    }
    free(pool->entries);
    pool->entries = NULL;
    pool->count = 0;
    pool->capacity = 0;
}

static int output_pool_create(VulkanCtx *vk, int capacity, uint32_t width, uint32_t height,
                              VkFormat format, OutputPool *pool)
{
    memset(pool, 0, sizeof(*pool));
    pool->entries = calloc(capacity, sizeof(OutputPoolEntry));
    if (!pool->entries) return -1;
    pool->capacity = capacity;
    pool->width = width;
    pool->height = height;
    pool->format = format;

    for (int i = 0; i < capacity; i++) {
        OutputPoolEntry *e = &pool->entries[i];
        e->dmabuf_fd = -1;
        if (create_dmabuf_image(vk, width, height, format, 0,
                                &e->image, &e->memory, &e->view, &e->dmabuf_fd) != 0) {
            output_pool_destroy(vk, pool);
            return -1;
        }
        pool->count++;
    }
    return 0;
}

static int output_pool_create_internal(VulkanCtx *vk, int capacity, uint32_t width,
                                        uint32_t height, VkFormat format,
                                        VkImageUsageFlags usage, OutputPool *pool)
{
    memset(pool, 0, sizeof(*pool));
    pool->entries = calloc(capacity, sizeof(OutputPoolEntry));
    if (!pool->entries) return -1;
    pool->capacity = capacity;
    pool->width = width;
    pool->height = height;
    pool->format = format;

    for (int i = 0; i < capacity; i++) {
        OutputPoolEntry *e = &pool->entries[i];
        e->dmabuf_fd = -1;
        e->is_internal = 1;
        if (create_internal_image(vk, width, height, format, usage,
                                  &e->image, &e->memory, &e->view) != 0) {
            output_pool_destroy(vk, pool);
            return -1;
        }
        pool->count++;
    }
    return 0;
}

/**
 * output_pool_create_copyout() - Create pool with internal render images + DMA-buf copy buffers.
 *
 * For formats that Mali-G52 cannot export as external DMA-buf VkImages
 * (R16_UNORM, R16G16_UNORM), we render into internal GPU-local images and then
 * copy to DMA-buf backed VkBuffers with vkCmdCopyImageToBuffer.
 *
 * Each entry has:
 *   - image/memory/view: internal GPU-local, used as COLOR_ATTACHMENT + TRANSFER_SRC
 *   - copy_buffer/copy_buffer_memory: DMA-buf backed buffer for downstream consumers
 *   - dmabuf_fd: fd of the copy_buffer DMA-buf (returned to caller)
 */
static int output_pool_create_copyout(VulkanCtx *vk, int capacity, uint32_t width,
                                       uint32_t height, VkFormat format,
                                       OutputPool *pool)
{
    memset(pool, 0, sizeof(*pool));
    pool->entries = calloc(capacity, sizeof(OutputPoolEntry));
    if (!pool->entries) return -1;
    pool->capacity = capacity;
    pool->width = width;
    pool->height = height;
    pool->format = format;

    /* Compute buffer size: width * height * bytes_per_pixel
     * R16_UNORM = 2 bytes/pixel, R16G16_UNORM = 4 bytes/pixel */
    uint32_t bpp = 0;
    switch (format) {
    case VK_FORMAT_R16_UNORM:     bpp = 2; break;
    case VK_FORMAT_R16G16_UNORM:  bpp = 4; break;
    case VK_FORMAT_R8_UNORM:      bpp = 1; break;
    case VK_FORMAT_R8G8_UNORM:    bpp = 2; break;
    default:
        fprintf(stderr, "[vfmcap-vk] output_pool_create_copyout: unsupported format %d\n", format);
        free(pool->entries);
        pool->entries = NULL;
        return -1;
    }

    VkDeviceSize buf_size = (VkDeviceSize)width * height * bpp;

    for (int i = 0; i < capacity; i++) {
        OutputPoolEntry *e = &pool->entries[i];
        e->dmabuf_fd = -1;

        /* Create internal image for rendering */
        VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (create_internal_image(vk, width, height, format, usage,
                                  &e->image, &e->memory, &e->view) != 0) {
            output_pool_destroy(vk, pool);
            return -1;
        }

        /* Allocate DMA-buf for the output buffer */
        int dmabuf_fd = dmabuf_heap_alloc(buf_size);
        if (dmabuf_fd < 0) {
            snprintf(vk->last_error, sizeof(vk->last_error),
                     "dmabuf_heap_alloc(copyout, %lu) failed", (unsigned long)buf_size);
            output_pool_destroy(vk, pool);
            return -1;
        }

        /* Import DMA-buf as VkBuffer for transfer destination */
        int fd_for_vulkan = dup(dmabuf_fd);
        if (fd_for_vulkan < 0) {
            close(dmabuf_fd);
            snprintf(vk->last_error, sizeof(vk->last_error),
                     "dup(copyout fd) failed: %s", strerror(errno));
            output_pool_destroy(vk, pool);
            return -1;
        }

        VkExternalMemoryBufferCreateInfo ext_buf_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };
        VkBufferCreateInfo buf_create_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &ext_buf_info,
            .size = buf_size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkResult result = vkCreateBuffer(vk->device, &buf_create_info, NULL, &e->copy_buffer);
        if (result != VK_SUCCESS) {
            close(fd_for_vulkan);
            close(dmabuf_fd);
            snprintf(vk->last_error, sizeof(vk->last_error),
                     "vkCreateBuffer(copyout) failed: %d", result);
            output_pool_destroy(vk, pool);
            return -1;
        }

        VkMemoryRequirements buf_mem_reqs;
        vkGetBufferMemoryRequirements(vk->device, e->copy_buffer, &buf_mem_reqs);

        VkImportMemoryFdInfoKHR import_info = {
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
            .fd = fd_for_vulkan,
        };

        VkDeviceSize alloc_size = buf_mem_reqs.size;
        if (buf_size > alloc_size) alloc_size = buf_size;

        VkMemoryAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &import_info,
            .allocationSize = alloc_size,
            .memoryTypeIndex = find_memory_type(vk, buf_mem_reqs.memoryTypeBits, 0),
        };

        if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
            close(dmabuf_fd);
            vkDestroyBuffer(vk->device, e->copy_buffer, NULL);
            e->copy_buffer = VK_NULL_HANDLE;
            snprintf(vk->last_error, sizeof(vk->last_error),
                     "No suitable memory type for copyout buffer");
            output_pool_destroy(vk, pool);
            return -1;
        }

        result = vkAllocateMemory(vk->device, &alloc_info, NULL, &e->copy_buffer_memory);
        if (result != VK_SUCCESS) {
            close(dmabuf_fd);
            vkDestroyBuffer(vk->device, e->copy_buffer, NULL);
            e->copy_buffer = VK_NULL_HANDLE;
            snprintf(vk->last_error, sizeof(vk->last_error),
                     "vkAllocateMemory(copyout) failed: %d", result);
            output_pool_destroy(vk, pool);
            return -1;
        }

        result = vkBindBufferMemory(vk->device, e->copy_buffer, e->copy_buffer_memory, 0);
        if (result != VK_SUCCESS) {
            close(dmabuf_fd);
            snprintf(vk->last_error, sizeof(vk->last_error),
                     "vkBindBufferMemory(copyout) failed: %d", result);
            output_pool_destroy(vk, pool);
            return -1;
        }

        e->dmabuf_fd = dmabuf_fd;
        e->copy_buffer_size = buf_size;
        e->has_copy_buffer = 1;
        e->is_internal = 0;  /* has DMA-buf fd for downstream */
        pool->count++;
    }
    return 0;
}

static OutputPoolEntry *output_pool_acquire(OutputPool *pool)
{
    if (!pool) return NULL;
    for (int i = 0; i < pool->count; i++) {
        if (!pool->entries[i].in_use) {
            pool->entries[i].in_use = 1;
            return &pool->entries[i];
        }
    }
    return NULL;
}

static void output_pool_release(OutputPool *pool, OutputPoolEntry *entry)
{
    if (!pool || !entry) return;
    for (int i = 0; i < pool->count; i++) {
        if (&pool->entries[i] == entry) {
            pool->entries[i].in_use = 0;
            return;
        }
    }
}

/* ---------- Shader loading ---------- */

static int load_shader(VulkanCtx *vk, const unsigned char *spv_data, size_t spv_size,
                       VkShaderModule *shader_module)
{
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode = (const uint32_t *)spv_data,
    };

    VkResult result = vkCreateShaderModule(vk->device, &create_info,
                                           NULL, shader_module);
    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCreateShaderModule failed: %d", result);
        return -1;
    }
    return 0;
}

/* ---------- Initialization ---------- */

int vfmcap_vk_init(VulkanCtx **vk_out, uint32_t width, uint32_t height, vfmcap_vk_fmt_t fmt)
{
    (void)fmt;
    VulkanCtx *vk = calloc(1, sizeof(VulkanCtx));
    if (!vk) return -1;
    *vk_out = vk;  /* Set early so error messages are retrievable */

    vk->width = width;
    vk->height = height;
    vk->input_cache_count = 0;
    vk->image_cache_count = 0;
    vk->cached_output.valid = 0;
    vk->cached_output.fd = -1;
    vk->pending_in_fd = -1;
    vk->pending_out_fd = -1;
    vk->has_pending = 0;

    for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
        vk->input_cache[i].valid = 0;
        vk->input_cache[i].fd = -1;
        vk->image_cache[i].valid = 0;
        vk->image_cache[i].fd = -1;
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

    result = vkCreateInstance(&instance_info, NULL, &vk->instance);
    VK_CHECK(result, "vkCreateInstance");

    /* Physical device selection */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(vk->instance, &dev_count, NULL);
    if (dev_count == 0) {
        snprintf(vk->last_error, sizeof(vk->last_error), "No Vulkan devices");
        return -1;
    }

    VkPhysicalDevice *devices = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(vk->instance, &dev_count, devices);

    for (uint32_t i = 0; i < dev_count; i++) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, NULL);
        VkQueueFamilyProperties *qf_props = malloc(qf_count * sizeof(*qf_props));
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, qf_props);

        for (uint32_t j = 0; j < qf_count; j++) {
            if ((qf_props[j].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
                == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
                vk->physical_device = devices[i];
                vk->compute_queue_family = j;
                vk->graphics_queue_family = j;
                break;
            }
        }
        free(qf_props);
        if (vk->physical_device != VK_NULL_HANDLE) break;
    }
    free(devices);

    if (vk->physical_device == VK_NULL_HANDLE) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "No device with graphics+compute support");
        return -1;
    }

    vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &vk->memory_props);

    /* Logical device */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    const char *dev_exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
    };

    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
        .dynamicRendering = VK_TRUE,
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &dynamic_rendering,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = 7,
        .ppEnabledExtensionNames = dev_exts,
    };

    result = vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device);
    VK_CHECK(result, "vkCreateDevice");

    vkGetDeviceQueue(vk->device, vk->compute_queue_family, 0, &vk->compute_queue);
    vkGetDeviceQueue(vk->device, vk->graphics_queue_family, 0, &vk->graphics_queue);

    /* Load dynamic rendering function pointers */
    vk->cmd_begin_rendering = (PFN_vkCmdBeginRenderingKHR)
        vkGetDeviceProcAddr(vk->device, "vkCmdBeginRenderingKHR");
    if (!vk->cmd_begin_rendering) {
        vk->cmd_begin_rendering = (PFN_vkCmdBeginRenderingKHR)
            vkGetDeviceProcAddr(vk->device, "vkCmdBeginRendering");
    }
    vk->cmd_end_rendering = (PFN_vkCmdEndRenderingKHR)
        vkGetDeviceProcAddr(vk->device, "vkCmdEndRenderingKHR");
    if (!vk->cmd_end_rendering) {
        vk->cmd_end_rendering = (PFN_vkCmdEndRenderingKHR)
            vkGetDeviceProcAddr(vk->device, "vkCmdEndRendering");
    }
    if (!vk->cmd_begin_rendering || !vk->cmd_end_rendering) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkCmdBeginRendering/vkCmdEndRendering not available");
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }

    /* Command pool */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = vk->compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    };

    result = vkCreateCommandPool(vk->device, &pool_info, NULL, &vk->command_pool);
    VK_CHECK(result, "vkCreateCommandPool");

    /* Command buffer */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    result = vkAllocateCommandBuffers(vk->device, &cmd_alloc, &vk->command_buffer);
    VK_CHECK(result, "vkAllocateCommandBuffers");

    /* Fence */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    result = vkCreateFence(vk->device, &fence_info, NULL, &vk->fence);
    VK_CHECK(result, "vkCreateFence");

    /* Descriptor pool */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 6,
        .poolSizeCount = 3,
        .pPoolSizes = pool_sizes,
    };

    result = vkCreateDescriptorPool(vk->device, &desc_pool_info, NULL,
                                    &vk->descriptor_pool);
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

    result = vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL,
                                         &vk->descriptor_set_layout);
    VK_CHECK(result, "vkCreateDescriptorSetLayout");

    /* Descriptor set */
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk->descriptor_set_layout,
    };

    result = vkAllocateDescriptorSets(vk->device, &desc_alloc, &vk->descriptor_set);
    VK_CHECK(result, "vkAllocateDescriptorSets");

    /* Descriptor set layout for 10-bit compute-to-graphics decode: storage buffer + 2 storage images */
    VkDescriptorSetLayoutBinding decode_bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };

    VkDescriptorSetLayoutCreateInfo decode_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = decode_bindings,
    };

    result = vkCreateDescriptorSetLayout(vk->device, &decode_layout_info, NULL,
                                         &vk->compute_descriptor_set_layout_decode);
    VK_CHECK(result, "vkCreateDescriptorSetLayout(decode)");

    VkDescriptorSetAllocateInfo decode_desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk->compute_descriptor_set_layout_decode,
    };

    result = vkAllocateDescriptorSets(vk->device, &decode_desc_alloc, &vk->compute_descriptor_set_decode);
    VK_CHECK(result, "vkAllocateDescriptorSets(decode)");

    /* Pipeline layout: push constants = { width, height, pairs_per_row, reserved } */
    VkPushConstantRange push_constant = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 4,
    };

    VkPipelineLayoutCreateInfo pl_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant,
    };

    result = vkCreatePipelineLayout(vk->device, &pl_layout_info, NULL,
                                    &vk->pipeline_layout);
    VK_CHECK(result, "vkCreatePipelineLayout");

    /* Pipeline layout for 10-bit compute-to-graphics decode */
    VkPipelineLayoutCreateInfo decode_pl_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->compute_descriptor_set_layout_decode,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant,
    };

    result = vkCreatePipelineLayout(vk->device, &decode_pl_layout_info, NULL,
                                    &vk->pipeline_layout_decode);
    VK_CHECK(result, "vkCreatePipelineLayout(decode)");

    /* Ycbcr sampler for NV12/NV21 input DMA-bufs */
    VkSamplerYcbcrConversionCreateInfo ycbcr_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
        .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
        .components = {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN,
        .yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN,
        .chromaFilter = VK_FILTER_LINEAR,
        .forceExplicitReconstruction = VK_FALSE,
    };

    result = vkCreateSamplerYcbcrConversion(vk->device, &ycbcr_info, NULL,
                                            &vk->ycbcr_conversion);
    if (result != VK_SUCCESS) {
        /* Non-fatal: some drivers may not support Ycbcr for this format */
        vk->ycbcr_conversion = VK_NULL_HANDLE;
    }

    VkSamplerYcbcrConversionInfo ycbcr_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
        .conversion = vk->ycbcr_conversion,
    };

    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = (vk->ycbcr_conversion != VK_NULL_HANDLE) ? &ycbcr_sampler_info : NULL,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    };

    result = vkCreateSampler(vk->device, &sampler_info, NULL, &vk->ycbcr_sampler);
    if (result != VK_SUCCESS) {
        vk->ycbcr_sampler = VK_NULL_HANDLE;
    }

    /* Graphics descriptor set layout: combined image sampler at binding 0 */
    VkDescriptorSetLayoutBinding gfx_bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
          .pImmutableSamplers = (vk->ycbcr_sampler != VK_NULL_HANDLE) ? &vk->ycbcr_sampler : NULL },
    };

    VkDescriptorSetLayoutCreateInfo gfx_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = gfx_bindings,
    };

    result = vkCreateDescriptorSetLayout(vk->device, &gfx_layout_info, NULL,
                                         &vk->gfx_descriptor_set_layout);
    VK_CHECK(result, "vkCreateDescriptorSetLayout(gfx)");

    VkDescriptorSetAllocateInfo gfx_desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk->gfx_descriptor_set_layout,
    };

    result = vkAllocateDescriptorSets(vk->device, &gfx_desc_alloc, &vk->gfx_descriptor_set);
    VK_CHECK(result, "vkAllocateDescriptorSets(gfx)");

    /* Graphics pipeline layout */
    VkPipelineLayoutCreateInfo gfx_pl_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->gfx_descriptor_set_layout,
    };

    result = vkCreatePipelineLayout(vk->device, &gfx_pl_layout_info, NULL,
                                    &vk->gfx_pipeline_layout);
    VK_CHECK(result, "vkCreatePipelineLayout(gfx)");

    /* Regular linear sampler for 10-bit intermediate */
    VkSamplerCreateInfo regular_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    };
    result = vkCreateSampler(vk->device, &regular_sampler_info, NULL, &vk->regular_sampler);
    VK_CHECK(result, "vkCreateSampler(regular)");

    /* Graphics descriptor set layout for 10-bit path: two regular combined image samplers (Y + UV) */
    VkDescriptorSetLayoutBinding gfx_bindings_10bit[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
          .pImmutableSamplers = &vk->regular_sampler },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
          .pImmutableSamplers = &vk->regular_sampler },
    };

    VkDescriptorSetLayoutCreateInfo gfx_layout_info_10bit = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = gfx_bindings_10bit,
    };

    result = vkCreateDescriptorSetLayout(vk->device, &gfx_layout_info_10bit, NULL,
                                         &vk->gfx_descriptor_set_layout_10bit);
    VK_CHECK(result, "vkCreateDescriptorSetLayout(gfx-10bit)");

    VkDescriptorSetAllocateInfo gfx_desc_alloc_10bit = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk->gfx_descriptor_set_layout_10bit,
    };

    result = vkAllocateDescriptorSets(vk->device, &gfx_desc_alloc_10bit, &vk->gfx_descriptor_set_10bit);
    VK_CHECK(result, "vkAllocateDescriptorSets(gfx-10bit)");

    /* Graphics pipeline layout for 10-bit path */
    VkPipelineLayoutCreateInfo gfx_pl_layout_info_10bit = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->gfx_descriptor_set_layout_10bit,
    };

    result = vkCreatePipelineLayout(vk->device, &gfx_pl_layout_info_10bit, NULL,
                                    &vk->gfx_pipeline_layout_10bit);
    VK_CHECK(result, "vkCreatePipelineLayout(gfx-10bit)");

    /* Load shaders */
    if (load_shader(vk, amly_to_p010_spv, sizeof(amly_to_p010_spv),
                    &vk->shader_p010) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, amly_to_nv12_spv, sizeof(amly_to_nv12_spv),
                    &vk->shader_nv12) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }

    /* Create P010 pipeline */
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk->shader_p010,
            .pName = "main",
        },
        .layout = vk->pipeline_layout,
    };

    result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &vk->pipeline_p010);
    VK_CHECK(result, "vkCreateComputePipelines(P010)");

    /* Create NV12 pipeline */
    pipeline_info.stage.module = vk->shader_nv12;
    result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &vk->pipeline_nv12);
    VK_CHECK(result, "vkCreateComputePipelines(NV12)");

    /* Create AMLY->R16 Y + R16G16 UV decode pipeline */
    if (load_shader(vk, amly_decode_spv, sizeof(amly_decode_spv),
                    &vk->shader_amly_decode) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }

    VkComputePipelineCreateInfo decode_pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk->shader_amly_decode,
            .pName = "main",
        },
        .layout = vk->pipeline_layout_decode,
    };

    result = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1,
                                      &decode_pipeline_info, NULL, &vk->pipeline_amly_decode);
    VK_CHECK(result, "vkCreateComputePipelines(decode)");

    /* ---------- Graphics pipelines (dynamic rendering) ---------- */
    if (load_shader(vk, fullscreen_vert_spv, sizeof(fullscreen_vert_spv),
                    &vk->gfx_shader_vert) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, nv12_y_frag_spv, sizeof(nv12_y_frag_spv),
                    &vk->gfx_shader_nv12_y) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, nv12_uv_frag_spv, sizeof(nv12_uv_frag_spv),
                    &vk->gfx_shader_nv12_uv) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, p010_y_frag_spv, sizeof(p010_y_frag_spv),
                    &vk->gfx_shader_p010_y) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, p010_uv_frag_spv, sizeof(p010_uv_frag_spv),
                    &vk->gfx_shader_p010_uv) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, nv12_y_from_r16_frag_spv, sizeof(nv12_y_from_r16_frag_spv),
                    &vk->gfx_shader_nv12_y_from_r16) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, nv12_uv_from_r16g16_frag_spv, sizeof(nv12_uv_from_r16g16_frag_spv),
                    &vk->gfx_shader_nv12_uv_from_r16g16) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, p010_y_from_r16_frag_spv, sizeof(p010_y_from_r16_frag_spv),
                    &vk->gfx_shader_p010_y_from_r16) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }
    if (load_shader(vk, p010_uv_from_r16g16_frag_spv, sizeof(p010_uv_from_r16g16_frag_spv),
                    &vk->gfx_shader_p010_uv_from_r16g16) != 0) {
        *vk_out = NULL;
        vfmcap_vk_cleanup(vk);
        return -1;
    }

    VkPipelineShaderStageCreateInfo vert_stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vk->gfx_shader_vert,
        .pName = "main",
    };

    VkPipelineVertexInputStateCreateInfo empty_vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkViewport viewport = { 0, 0, (float)width, (float)height, 0.0f, 1.0f };
    VkRect2D scissor = { {0, 0}, {width, height} };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineColorBlendAttachmentState cb_attach = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &cb_attach,
    };

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dyn_states,
    };

    VkFormat nv12_y_fmt = VK_FORMAT_R8_UNORM;
    VkPipelineRenderingCreateInfo nv12_y_render = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &nv12_y_fmt,
    };

    VkGraphicsPipelineCreateInfo gfx_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &nv12_y_render,
        .stageCount = 2,
        .pStages = (VkPipelineShaderStageCreateInfo[]){
            vert_stage,
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
              VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_nv12_y, "main", NULL },
        },
        .pVertexInputState = &empty_vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pColorBlendState = &cb,
        .pDynamicState = &dyn,
        .layout = vk->gfx_pipeline_layout,
    };

    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_nv12_y);
    VK_CHECK(result, "vkCreateGraphicsPipelines(NV12 Y)");

    VkFormat nv12_uv_fmt = VK_FORMAT_R8G8_UNORM;
    VkPipelineRenderingCreateInfo nv12_uv_render = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &nv12_uv_fmt,
    };
    gfx_info.pNext = &nv12_uv_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_nv12_uv, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_nv12_uv);
    VK_CHECK(result, "vkCreateGraphicsPipelines(NV12 UV)");

    VkFormat p010_y_fmt = VK_FORMAT_R16_UNORM;
    VkPipelineRenderingCreateInfo p010_y_render = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &p010_y_fmt,
    };
    gfx_info.pNext = &p010_y_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_p010_y, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_p010_y);
    VK_CHECK(result, "vkCreateGraphicsPipelines(P010 Y)");

    VkFormat p010_uv_fmt = VK_FORMAT_R16G16_UNORM;
    VkPipelineRenderingCreateInfo p010_uv_render = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &p010_uv_fmt,
    };
    gfx_info.pNext = &p010_uv_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_p010_uv, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_p010_uv);
    VK_CHECK(result, "vkCreateGraphicsPipelines(P010 UV)");

    /* Create 10-bit variant graphics pipelines with two-plane regular sampler layout */
    gfx_info.pNext = &nv12_y_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_nv12_y_from_r16, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout_10bit;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_nv12_y_10bit);
    VK_CHECK(result, "vkCreateGraphicsPipelines(NV12 Y 10bit)");

    gfx_info.pNext = &nv12_uv_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_nv12_uv_from_r16g16, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout_10bit;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_nv12_uv_10bit);
    VK_CHECK(result, "vkCreateGraphicsPipelines(NV12 UV 10bit)");

    gfx_info.pNext = &p010_y_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_p010_y_from_r16, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout_10bit;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_p010_y_10bit);
    VK_CHECK(result, "vkCreateGraphicsPipelines(P010 Y 10bit)");

    gfx_info.pNext = &p010_uv_render;
    gfx_info.pStages = (VkPipelineShaderStageCreateInfo[]){
        vert_stage,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, vk->gfx_shader_p010_uv_from_r16g16, "main", NULL },
    };
    gfx_info.layout = vk->gfx_pipeline_layout_10bit;
    result = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &gfx_info,
                                       NULL, &vk->gfx_pipeline_p010_uv_10bit);
    VK_CHECK(result, "vkCreateGraphicsPipelines(P010 UV 10bit)");

    /* Create output pools based on requested format */
    if (fmt == VFMCAP_VK_FMT_NV12 || fmt == VFMCAP_VK_FMT_NV21) {
        if (output_pool_create(vk, 4, width, height, VK_FORMAT_R8_UNORM,
                               &vk->pool_nv12_y) != 0 ||
            output_pool_create(vk, 4, width / 2, height / 2, VK_FORMAT_R8G8_UNORM,
                               &vk->pool_nv12_uv) != 0) {
            vfmcap_vk_cleanup(vk);
            return -1;
        }
    } else if (fmt == VFMCAP_VK_FMT_P010) {
        /* Mali-G52 cannot create DMA-buf-exportable VkImages for R16_UNORM /
         * R16G16_UNORM.  Use copyout pools: render to internal image, then
         * vkCmdCopyImageToBuffer into a DMA-buf backed VkBuffer. */
        if (output_pool_create_copyout(vk, 4, width, height, VK_FORMAT_R16_UNORM,
                                       &vk->pool_p010_y) != 0 ||
            output_pool_create_copyout(vk, 4, width / 2, height / 2, VK_FORMAT_R16G16_UNORM,
                                       &vk->pool_p010_uv) != 0) {
            vfmcap_vk_cleanup(vk);
            return -1;
        }
    }

    /* Create R16 Y + R16G16 UV intermediate pools for 10-bit compute->graphics chain */
    if (output_pool_create_internal(vk, 4, width, height, VK_FORMAT_R16_UNORM,
                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     &vk->pool_intermediate_y) != 0 ||
        output_pool_create_internal(vk, 4, width / 2, height / 2, VK_FORMAT_R16G16_UNORM,
                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     &vk->pool_intermediate_uv) != 0) {
        vfmcap_vk_cleanup(vk);
        return -1;
    }

    vk->initialized = 1;
    vk->frame_count = 0;

    fprintf(stderr, "[vfmcap-vk] Initialized: %ux%u, P010+NV12+GFX pipelines ready\n",
            width, height);

    return 0;
}

/* ---------- Async submit ---------- */

int vfmcap_vk_convert_submit(VulkanCtx *vk, int in_fd, int out_fd, uint32_t width,
                             uint32_t height, vfmcap_vk_fmt_t fmt)
{
    if (!vk || !vk->initialized) {
        if (vk) snprintf(vk->last_error, sizeof(vk->last_error), "Not initialized");
        return -1;
    }

    VkResult result;

    /* Fence management */
    if (vk->frame_count == 0) {
        vkResetFences(vk->device, 1, &vk->fence);
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
    int in_idx = input_cache_get(vk, in_fd, input_size);
    if (in_idx < 0) return -1;

    struct dma_buf_sync sync_start = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
    };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_start);

    VkBuffer in_buffer = vk->input_cache[in_idx].buffer;

    /* Cached output import + DMA_BUF_SYNC write-access bracket (start) */
    if (output_cache_get(vk, out_fd, output_size) != 0) return -1;
    VkBuffer out_buffer = vk->cached_output.buffer;

    struct dma_buf_sync out_sync_start = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE
    };
    ioctl(out_fd, DMA_BUF_IOCTL_SYNC, &out_sync_start);

    /* Record command buffer */
    VkCommandBuffer cmd = vk->command_buffer;

    result = vkResetCommandPool(vk->device, vk->command_pool, 0);
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
          .dstSet = vk->descriptor_set, .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->descriptor_set, .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->descriptor_set, .dstBinding = 2,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[2] },
    };

    vkUpdateDescriptorSets(vk->device, 3, writes, 0, NULL);

    /* Bind pipeline */
    VkPipeline pipeline = (fmt == VFMCAP_VK_FMT_P010) ?
                          vk->pipeline_p010 : vk->pipeline_nv12;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pipeline_layout, 0, 1,
                            &vk->descriptor_set, 0, NULL);

    /* Push constants: { width, height, pairs_per_row, reserved } */
    uint32_t pairs_per_row = width / 2;
    uint32_t push_data[] = { width, height, pairs_per_row, 0u };
    vkCmdPushConstants(cmd, vk->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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

    /* Submit to graphics queue */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    result = vkQueueSubmit(vk->graphics_queue, 1, &submit_info, vk->fence);
    VK_CHECK(result, "vkQueueSubmit(graphics)");

    vk->pending_in_fd = in_fd;
    vk->pending_out_fd = out_fd;
    vk->pending_out_fd2 = -1;
    vk->has_pending = 1;

    return 0;
}

/* ---------- Wait for GPU completion ---------- */

int vfmcap_vk_convert_wait(VulkanCtx *vk)
{
    if (!vk || !vk->initialized) {
        if (vk) snprintf(vk->last_error, sizeof(vk->last_error), "Not initialized");
        return -1;
    }

    if (!vk->has_pending) return 0;

    VkResult result = vkWaitForFences(vk->device, 1, &vk->fence,
                                      VK_TRUE, 5000000000ULL);

    /* Release DMA-buf read access on input */
    if (vk->pending_in_fd >= 0) {
        struct dma_buf_sync sync_end = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ
        };
        ioctl(vk->pending_in_fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    }

    /*
     * Destroy the Vulkan cache entry for the just-completed input frame
     * so that Mali releases its dma_buf reference.  After vkFreeMemory,
     * Mali defers the actual dma_buf_put to an internal worker thread.
     * vkDeviceWaitIdle below forces that flush.
     */
    if (vk->pending_in_fd >= 0) {
        for (int i = 0; i < vk->input_cache_count; i++) {
            if (vk->input_cache[i].valid &&
                vk->input_cache[i].fd == vk->pending_in_fd) {
                cache_entry_destroy(vk, &vk->input_cache[i]);
                break;
            }
        }
    }

    /* Force Mali to complete ALL deferred work — including L2/WC flush
     * for output writes and deferred dma_buf_put from vkFreeMemory */
    vkDeviceWaitIdle(vk->device);

    /* ARM DSB to ensure all outstanding AXI write transactions from Mali
     * have committed to the memory controller before downstream DMA. */
    __sync_synchronize();
    usleep(500);

    /* Release DMA-buf write access on output(s) AFTER vkDeviceWaitIdle —
     * ensures Mali's write-combine buffers and L2 cache have been flushed
     * to RAM so kernel cache maintenance operates on coherent data. */
    if (vk->pending_out_fd >= 0) {
        struct dma_buf_sync sync_end_wr = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE
        };
        ioctl(vk->pending_out_fd, DMA_BUF_IOCTL_SYNC, &sync_end_wr);
    }
    if (vk->pending_out_fd2 >= 0) {
        struct dma_buf_sync sync_end_wr = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE
        };
        ioctl(vk->pending_out_fd2, DMA_BUF_IOCTL_SYNC, &sync_end_wr);
    }

    vk->has_pending = 0;
    vk->pending_in_fd = -1;
    vk->pending_out_fd = -1;
    vk->pending_out_fd2 = -1;

    if (result != VK_SUCCESS) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "vkWaitForFences failed: %d (frame %lu)",
                 result, (unsigned long)vk->frame_count);
        return -1;
    }

    vkResetFences(vk->device, 1, &vk->fence);
    vk->frame_count++;

    return 0;
}

/* ---------- Synchronous conversion ---------- */

int vfmcap_vk_convert(VulkanCtx *vk, int in_fd, int out_fd, uint32_t width,
                      uint32_t height, vfmcap_vk_fmt_t fmt)
{
    int ret = vfmcap_vk_convert_submit(vk, in_fd, out_fd, width, height, fmt);
    if (ret != 0) return ret;
    return vfmcap_vk_convert_wait(vk);
}

/* ---------- 8-bit graphics render submit ---------- */

int vfmcap_vk_render_submit(VulkanCtx *vk, int in_fd,
                            int out_y_fd, int out_uv_fd,
                            uint32_t src_width, uint32_t src_height,
                            uint32_t dst_width, uint32_t dst_height,
                            vfmcap_vk_fmt_t fmt)
{
    if (!vk || !vk->initialized) {
        if (vk) snprintf(vk->last_error, sizeof(vk->last_error), "Not initialized");
        return -1;
    }

    VkPipeline y_pipeline, uv_pipeline;
    VkFormat out_y_fmt, out_uv_fmt;

    if (fmt == VFMCAP_VK_FMT_NV12 || fmt == VFMCAP_VK_FMT_NV21) {
        y_pipeline = vk->gfx_pipeline_nv12_y;
        uv_pipeline = vk->gfx_pipeline_nv12_uv;
        out_y_fmt = VK_FORMAT_R8_UNORM;
        out_uv_fmt = VK_FORMAT_R8G8_UNORM;
    } else if (fmt == VFMCAP_VK_FMT_P010) {
        y_pipeline = vk->gfx_pipeline_p010_y;
        uv_pipeline = vk->gfx_pipeline_p010_uv;
        out_y_fmt = VK_FORMAT_R16_UNORM;
        out_uv_fmt = VK_FORMAT_R16G16_UNORM;
    } else {
        snprintf(vk->last_error, sizeof(vk->last_error), "Unsupported render format");
        return -1;
    }

    /* Import NV12/NV21 input as YCbCr VkImage */
    VkImage in_image;
    VkImageView in_view;
    VkFormat in_fmt = (fmt == VFMCAP_VK_FMT_NV21) ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM
                                                    : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    if (image_cache_get(vk, in_fd, src_width, src_height, in_fmt, 0,
                        &in_image, &in_view) != 0) {
        return -1;
    }

    /* Import output DMA-buf fds as VkImages */
    VkImage out_y_image, out_uv_image;
    VkImageView out_y_view, out_uv_view;
    if (image_cache_get(vk, out_y_fd, dst_width, dst_height, out_y_fmt, 0,
                        &out_y_image, &out_y_view) != 0) {
        return -1;
    }
    if (image_cache_get(vk, out_uv_fd, dst_width / 2, dst_height / 2, out_uv_fmt, 0,
                        &out_uv_image, &out_uv_view) != 0) {
        return -1;
    }

    VkResult result;

    /* Fence management */
    if (vk->frame_count == 0) {
        vkResetFences(vk->device, 1, &vk->fence);
    }

    /* DMA-buf sync: start read on input, write on outputs */
    struct dma_buf_sync sync_start_rd = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
    };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_start_rd);

    struct dma_buf_sync sync_start_wr = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE
    };
    ioctl(out_y_fd, DMA_BUF_IOCTL_SYNC, &sync_start_wr);
    ioctl(out_uv_fd, DMA_BUF_IOCTL_SYNC, &sync_start_wr);

    /* Record command buffer */
    VkCommandBuffer cmd = vk->command_buffer;

    result = vkResetCommandPool(vk->device, vk->command_pool, 0);
    VK_CHECK(result, "vkResetCommandPool");

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    result = vkBeginCommandBuffer(cmd, &begin_info);
    VK_CHECK(result, "vkBeginCommandBuffer");

    /* Update graphics descriptor set: YCbCr input sampler */
    VkDescriptorImageInfo in_img_info = {
        .sampler = vk->ycbcr_sampler,
        .imageView = in_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkWriteDescriptorSet gfx_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = vk->gfx_descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &in_img_info,
    };
    vkUpdateDescriptorSets(vk->device, 1, &gfx_write, 0, NULL);

    /* Transition input to SHADER_READ, output Y to COLOR_ATTACHMENT */
    VkImageMemoryBarrier in_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image = in_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    VkImageMemoryBarrier out_y_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .image = out_y_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    VkImageMemoryBarrier barriers[2] = { in_barrier, out_y_barrier };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, NULL, 0, NULL, 2, barriers);

    /* Y pass */
    VkRenderingAttachmentInfo y_attach = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = out_y_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo y_render = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { {0, 0}, {dst_width, dst_height} },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &y_attach,
    };

    vk->cmd_begin_rendering(cmd, &y_render);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, y_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk->gfx_pipeline_layout, 0, 1,
                            &vk->gfx_descriptor_set, 0, NULL);
    VkViewport viewport_y = { 0.0f, 0.0f, (float)dst_width, (float)dst_height, 0.0f, 1.0f };
    VkRect2D scissor_y = { {0, 0}, {dst_width, dst_height} };
    vkCmdSetViewport(cmd, 0, 1, &viewport_y);
    vkCmdSetScissor(cmd, 0, 1, &scissor_y);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vk->cmd_end_rendering(cmd);

    /* Transition Y to GENERAL, UV to COLOR_ATTACHMENT */
    VkImageMemoryBarrier y_post = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = out_y_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    VkImageMemoryBarrier out_uv_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .image = out_uv_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    VkImageMemoryBarrier barriers2[2] = { y_post, out_uv_barrier };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, NULL, 0, NULL, 2, barriers2);

    /* UV pass */
    VkRenderingAttachmentInfo uv_attach = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = out_uv_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo uv_render = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { {0, 0}, {dst_width / 2, dst_height / 2} },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uv_attach,
    };

    vk->cmd_begin_rendering(cmd, &uv_render);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, uv_pipeline);
    VkViewport viewport_uv = { 0.0f, 0.0f, (float)(dst_width / 2), (float)(dst_height / 2), 0.0f, 1.0f };
    VkRect2D scissor_uv = { {0, 0}, {dst_width / 2, dst_height / 2} };
    vkCmdSetViewport(cmd, 0, 1, &viewport_uv);
    vkCmdSetScissor(cmd, 0, 1, &scissor_uv);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vk->cmd_end_rendering(cmd);

    /* Final barrier: UV to GENERAL */
    VkImageMemoryBarrier uv_final = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = out_uv_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 1, &uv_final);

    result = vkEndCommandBuffer(cmd);
    VK_CHECK(result, "vkEndCommandBuffer");

    /* Submit to graphics queue */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    result = vkQueueSubmit(vk->graphics_queue, 1, &submit_info, vk->fence);
    VK_CHECK(result, "vkQueueSubmit(graphics)");

    vk->pending_in_fd = in_fd;
    vk->pending_out_fd = out_y_fd;
    vk->pending_out_fd2 = out_uv_fd;
    vk->has_pending = 1;

    return 0;
}

/* ---------- Render with output pool ---------- */

int vfmcap_vk_render_and_wait(VulkanCtx *vk, int in_fd,
                              uint32_t src_width, uint32_t src_height,
                              uint32_t dst_width, uint32_t dst_height,
                              vfmcap_vk_fmt_t fmt,
                              int *out_y_fd, int *out_uv_fd)
{
    if (!vk || !vk->initialized) {
        if (vk) snprintf(vk->last_error, sizeof(vk->last_error), "Not initialized");
        return -1;
    }

    OutputPool *y_pool = NULL;
    OutputPool *uv_pool = NULL;

    if (fmt == VFMCAP_VK_FMT_NV12 || fmt == VFMCAP_VK_FMT_NV21) {
        y_pool = &vk->pool_nv12_y;
        uv_pool = &vk->pool_nv12_uv;
    } else if (fmt == VFMCAP_VK_FMT_P010) {
        y_pool = &vk->pool_p010_y;
        uv_pool = &vk->pool_p010_uv;
    } else {
        snprintf(vk->last_error, sizeof(vk->last_error), "Unsupported render format");
        return -1;
    }

    OutputPoolEntry *y_entry = output_pool_acquire(y_pool);
    OutputPoolEntry *uv_entry = output_pool_acquire(uv_pool);
    if (!y_entry || !uv_entry) {
        if (y_entry) output_pool_release(y_pool, y_entry);
        if (uv_entry) output_pool_release(uv_pool, uv_entry);
        snprintf(vk->last_error, sizeof(vk->last_error), "Output pool exhausted");
        return -1;
    }

    int ret = vfmcap_vk_render_submit(vk, in_fd, y_entry->dmabuf_fd, uv_entry->dmabuf_fd,
                                      src_width, src_height, dst_width, dst_height, fmt);
    if (ret != 0) {
        output_pool_release(y_pool, y_entry);
        output_pool_release(uv_pool, uv_entry);
        return -1;
    }

    ret = vfmcap_vk_convert_wait(vk);
    if (ret != 0) {
        output_pool_release(y_pool, y_entry);
        output_pool_release(uv_pool, uv_entry);
        return -1;
    }

    *out_y_fd = y_entry->dmabuf_fd;
    *out_uv_fd = uv_entry->dmabuf_fd;
    return 0;
}

void vfmcap_vk_release_output(VulkanCtx *vk, int y_fd, int uv_fd, vfmcap_vk_fmt_t fmt)
{
    if (!vk) return;

    OutputPool *y_pool = NULL;
    OutputPool *uv_pool = NULL;

    if (fmt == VFMCAP_VK_FMT_NV12 || fmt == VFMCAP_VK_FMT_NV21) {
        y_pool = &vk->pool_nv12_y;
        uv_pool = &vk->pool_nv12_uv;
    } else if (fmt == VFMCAP_VK_FMT_P010) {
        y_pool = &vk->pool_p010_y;
        uv_pool = &vk->pool_p010_uv;
    } else {
        return;
    }

    for (int i = 0; i < y_pool->count; i++) {
        if (y_pool->entries[i].dmabuf_fd == y_fd && y_pool->entries[i].in_use) {
            y_pool->entries[i].in_use = 0;
            break;
        }
    }
    for (int i = 0; i < uv_pool->count; i++) {
        if (uv_pool->entries[i].dmabuf_fd == uv_fd && uv_pool->entries[i].in_use) {
            uv_pool->entries[i].in_use = 0;
            break;
        }
    }
}

/* ---------- 10-bit compute-to-graphics render ---------- */

int vfmcap_vk_render_10bit_and_wait(VulkanCtx *vk, int in_fd,
                                     uint32_t src_width, uint32_t src_height,
                                     uint32_t dst_width, uint32_t dst_height,
                                     vfmcap_vk_fmt_t fmt,
                                     int *out_y_fd, int *out_uv_fd)
{
    if (!vk || !vk->initialized) {
        if (vk) snprintf(vk->last_error, sizeof(vk->last_error), "Not initialized");
        return -1;
    }

    VkPipeline y_pipeline, uv_pipeline;
    OutputPool *y_pool = NULL;
    OutputPool *uv_pool = NULL;

    if (fmt == VFMCAP_VK_FMT_NV12 || fmt == VFMCAP_VK_FMT_NV21) {
        y_pipeline = vk->gfx_pipeline_nv12_y_10bit;
        uv_pipeline = vk->gfx_pipeline_nv12_uv_10bit;
        y_pool = &vk->pool_nv12_y;
        uv_pool = &vk->pool_nv12_uv;
    } else if (fmt == VFMCAP_VK_FMT_P010) {
        y_pipeline = vk->gfx_pipeline_p010_y_10bit;
        uv_pipeline = vk->gfx_pipeline_p010_uv_10bit;
        y_pool = &vk->pool_p010_y;
        uv_pool = &vk->pool_p010_uv;
    } else {
        snprintf(vk->last_error, sizeof(vk->last_error), "Unsupported render format");
        return -1;
    }

    /* Import AMLY input buffer */
    VkDeviceSize input_size = (VkDeviceSize)src_width * src_height * 5 / 2;
    int in_idx = input_cache_get(vk, in_fd, input_size);
    if (in_idx < 0) return -1;
    VkBuffer in_buffer = vk->input_cache[in_idx].buffer;

    /* DMA-buf sync: start read access on input */
    struct dma_buf_sync sync_start_rd = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
    };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_start_rd);

    /* Acquire intermediate (R16 Y + R16G16 UV) and output pool entries */
    OutputPoolEntry *intermediate_y_entry = output_pool_acquire(&vk->pool_intermediate_y);
    OutputPoolEntry *intermediate_uv_entry = output_pool_acquire(&vk->pool_intermediate_uv);
    OutputPoolEntry *y_entry = output_pool_acquire(y_pool);
    OutputPoolEntry *uv_entry = output_pool_acquire(uv_pool);
    if (!intermediate_y_entry || !intermediate_uv_entry || !y_entry || !uv_entry) {
        if (intermediate_y_entry) output_pool_release(&vk->pool_intermediate_y, intermediate_y_entry);
        if (intermediate_uv_entry) output_pool_release(&vk->pool_intermediate_uv, intermediate_uv_entry);
        if (y_entry) output_pool_release(y_pool, y_entry);
        if (uv_entry) output_pool_release(uv_pool, uv_entry);
        snprintf(vk->last_error, sizeof(vk->last_error), "Output pool exhausted");
        return -1;
    }

    VkImage intermediate_y_image = intermediate_y_entry->image;
    VkImageView intermediate_y_view = intermediate_y_entry->view;
    VkImage intermediate_uv_image = intermediate_uv_entry->image;
    VkImageView intermediate_uv_view = intermediate_uv_entry->view;

    /* Get output images and views.
     * For copyout pool entries (has_copy_buffer=1), the image/view are stored
     * directly in the pool entry (internal GPU-local).
     * For regular DMA-buf pool entries, import via image cache. */
    VkImage out_y_image, out_uv_image;
    VkImageView out_y_view, out_uv_view;
    int uses_copyout = y_entry->has_copy_buffer;

    /* Both copyout and DMA-buf pool entries have image + view from creation */
    out_y_image = y_entry->image;
    out_y_view = y_entry->view;
    out_uv_image = uv_entry->image;
    out_uv_view = uv_entry->view;

    VkResult result;

    /* Fence management */
    if (vk->frame_count == 0) {
        vkResetFences(vk->device, 1, &vk->fence);
    }

    /* Record command buffer */
    VkCommandBuffer cmd = vk->command_buffer;

    result = vkResetCommandPool(vk->device, vk->command_pool, 0);
    VK_CHECK(result, "vkResetCommandPool");

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    result = vkBeginCommandBuffer(cmd, &begin_info);
    VK_CHECK(result, "vkBeginCommandBuffer");

    /* Update compute descriptors: buffer + Y image + UV image */
    VkDescriptorBufferInfo buffer_info = { in_buffer, 0, input_size };
    VkDescriptorImageInfo img_info_y = {
        .imageView = intermediate_y_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkDescriptorImageInfo img_info_uv = {
        .imageView = intermediate_uv_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkWriteDescriptorSet compute_writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->compute_descriptor_set_decode, .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->compute_descriptor_set_decode, .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = &img_info_y },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->compute_descriptor_set_decode, .dstBinding = 2,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = &img_info_uv },
    };
    vkUpdateDescriptorSets(vk->device, 3, compute_writes, 0, NULL);

    /* Bind compute pipeline and dispatch */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_amly_decode);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pipeline_layout_decode, 0, 1,
                            &vk->compute_descriptor_set_decode, 0, NULL);

    uint32_t pairs_per_row = src_width / 2;
    uint32_t push_data[] = { src_width, src_height, pairs_per_row, 0u };
    vkCmdPushConstants(cmd, vk->pipeline_layout_decode, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_data), push_data);

    uint32_t groups_x = (pairs_per_row + 127) / 128;
    uint32_t groups_y = src_height;
    vkCmdDispatch(cmd, groups_x, groups_y, 1);

    /* Transition intermediate Y/UV images to SHADER_READ_ONLY_OPTIMAL for graphics */
    VkImageMemoryBarrier intermediate_y_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image = intermediate_y_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    VkImageMemoryBarrier intermediate_uv_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image = intermediate_uv_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    /* Transition output Y to COLOR_ATTACHMENT_OPTIMAL */
    VkImageMemoryBarrier out_y_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .image = out_y_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    VkImageMemoryBarrier barriers[3] = { intermediate_y_barrier, intermediate_uv_barrier, out_y_barrier };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, NULL, 0, NULL, 3, barriers);

    /* Update graphics descriptor set with intermediate Y and UV image views */
    VkDescriptorImageInfo img_info_gfx_y = {
        .sampler = vk->regular_sampler,
        .imageView = intermediate_y_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo img_info_gfx_uv = {
        .sampler = vk->regular_sampler,
        .imageView = intermediate_uv_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkWriteDescriptorSet gfx_writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->gfx_descriptor_set_10bit,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &img_info_gfx_y },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vk->gfx_descriptor_set_10bit,
          .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &img_info_gfx_uv },
    };
    vkUpdateDescriptorSets(vk->device, 2, gfx_writes, 0, NULL);

    /* Y pass */
    VkRenderingAttachmentInfo y_attach = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = out_y_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };

    VkRenderingInfo y_render = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { {0, 0}, {dst_width, dst_height} },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &y_attach,
    };

    vk->cmd_begin_rendering(cmd, &y_render);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, y_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk->gfx_pipeline_layout_10bit, 0, 1,
                            &vk->gfx_descriptor_set_10bit, 0, NULL);
    VkViewport viewport_y = { 0.0f, 0.0f, (float)dst_width, (float)dst_height, 0.0f, 1.0f };
    VkRect2D scissor_y = { {0, 0}, {dst_width, dst_height} };
    vkCmdSetViewport(cmd, 0, 1, &viewport_y);
    vkCmdSetScissor(cmd, 0, 1, &scissor_y);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vk->cmd_end_rendering(cmd);

    /* Transition output Y: COLOR_ATTACHMENT -> TRANSFER_SRC (if copyout) or GENERAL */
    VkImageLayout y_final_layout = uses_copyout ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                                : VK_IMAGE_LAYOUT_GENERAL;
    VkAccessFlags y_final_access = uses_copyout ? VK_ACCESS_TRANSFER_READ_BIT : 0;
    VkPipelineStageFlags y_final_stage = uses_copyout ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                                      : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

    VkImageMemoryBarrier y_post_render = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = y_final_access,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = y_final_layout,
        .image = out_y_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    /* Transition output UV to COLOR_ATTACHMENT_OPTIMAL */
    VkImageMemoryBarrier out_uv_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .image = out_uv_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    VkImageMemoryBarrier barriers2[2] = { y_post_render, out_uv_barrier };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         y_final_stage,
                         0, 0, NULL, 0, NULL, 2, barriers2);

    /* UV pass */
    VkRenderingAttachmentInfo uv_attach = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = out_uv_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };

    VkRenderingInfo uv_render = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { {0, 0}, {dst_width / 2, dst_height / 2} },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &uv_attach,
    };

    vk->cmd_begin_rendering(cmd, &uv_render);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, uv_pipeline);
    VkViewport viewport_uv = { 0.0f, 0.0f, (float)(dst_width / 2), (float)(dst_height / 2), 0.0f, 1.0f };
    VkRect2D scissor_uv = { {0, 0}, {dst_width / 2, dst_height / 2} };
    vkCmdSetViewport(cmd, 0, 1, &viewport_uv);
    vkCmdSetScissor(cmd, 0, 1, &scissor_uv);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vk->cmd_end_rendering(cmd);

    if (uses_copyout) {
        /* Copyout path: transition UV image to TRANSFER_SRC, then copy both
         * images to their DMA-buf backed buffers. */
        VkImageMemoryBarrier uv_to_transfer = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = out_uv_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, NULL, 0, NULL, 1, &uv_to_transfer);

        /* Copy Y image -> Y buffer */
        VkBufferImageCopy y_copy = {
            .bufferOffset = 0,
            .bufferRowLength = dst_width,
            .bufferImageHeight = dst_height,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {dst_width, dst_height, 1},
        };
        vkCmdCopyImageToBuffer(cmd, out_y_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               y_entry->copy_buffer, 1, &y_copy);

        /* Copy UV image -> UV buffer */
        VkBufferImageCopy uv_copy = {
            .bufferOffset = 0,
            .bufferRowLength = dst_width / 2,
            .bufferImageHeight = dst_height / 2,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {dst_width / 2, dst_height / 2, 1},
        };
        vkCmdCopyImageToBuffer(cmd, out_uv_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               uv_entry->copy_buffer, 1, &uv_copy);

        /* Memory barrier to ensure transfer is visible to host/downstream */
        VkMemoryBarrier transfer_done = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &transfer_done, 0, NULL, 0, NULL);
    } else {
        /* Regular DMA-buf image path: transition UV to GENERAL */
        VkImageMemoryBarrier uv_to_general = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .image = out_uv_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, NULL, 0, NULL, 1, &uv_to_general);
    }

    result = vkEndCommandBuffer(cmd);
    VK_CHECK(result, "vkEndCommandBuffer");

    /* DMA-buf sync: start write access on output copyout buffers before GPU submit
     * (needed for cached DMA-buf heap buffers to ensure CPU cache coherency) */
    if (uses_copyout) {
        struct dma_buf_sync sync_start_wr = {
            .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE
        };
        ioctl(y_entry->dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync_start_wr);
        ioctl(uv_entry->dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync_start_wr);
    }

    /* Submit to graphics queue */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    result = vkQueueSubmit(vk->graphics_queue, 1, &submit_info, vk->fence);
    VK_CHECK(result, "vkQueueSubmit(graphics-10bit)");

    vk->pending_in_fd = in_fd;
    vk->pending_out_fd = y_entry->dmabuf_fd;
    vk->pending_out_fd2 = uv_entry->dmabuf_fd;
    vk->has_pending = 1;

    /* Wait for completion */
    int ret = vfmcap_vk_convert_wait(vk);
    if (ret != 0) {
        output_pool_release(&vk->pool_intermediate_y, intermediate_y_entry);
        output_pool_release(&vk->pool_intermediate_uv, intermediate_uv_entry);
        output_pool_release(y_pool, y_entry);
        output_pool_release(uv_pool, uv_entry);
        return -1;
    }

    /* Release intermediate pools immediately — they are internal temporaries */
    output_pool_release(&vk->pool_intermediate_y, intermediate_y_entry);
    output_pool_release(&vk->pool_intermediate_uv, intermediate_uv_entry);

    *out_y_fd = y_entry->dmabuf_fd;
    *out_uv_fd = uv_entry->dmabuf_fd;
    return 0;
}

/* ---------- 10-bit compute conversion (integrated path) ---------- */

int vfmcap_vk_compute_10bit_and_wait(VulkanCtx *vk, int in_fd,
                                     uint32_t width, uint32_t height,
                                     vfmcap_vk_fmt_t fmt,
                                     int *out_fd)
{
    if (!vk || !vk->initialized) {
        if (vk) snprintf(vk->last_error, sizeof(vk->last_error), "Not initialized");
        return -1;
    }

    VkDeviceSize y_plane_size, uv_plane_size, output_size;
    if (fmt == VFMCAP_VK_FMT_P010) {
        y_plane_size = (VkDeviceSize)width * height * 2;
        uv_plane_size = (VkDeviceSize)width * height;
    } else if (fmt == VFMCAP_VK_FMT_NV12) {
        y_plane_size = (VkDeviceSize)width * height;
        uv_plane_size = (VkDeviceSize)width * height / 2;
    } else {
        snprintf(vk->last_error, sizeof(vk->last_error), "Unsupported format");
        return -1;
    }
    output_size = y_plane_size + uv_plane_size;

    int dmabuf_fd = dmabuf_heap_alloc(output_size);
    if (dmabuf_fd < 0) {
        snprintf(vk->last_error, sizeof(vk->last_error),
                 "dmabuf_heap_alloc(output) failed");
        return -1;
    }

    int ret = vfmcap_vk_convert_submit(vk, in_fd, dmabuf_fd, width, height, fmt);
    if (ret != 0) {
        close(dmabuf_fd);
        return -1;
    }

    ret = vfmcap_vk_convert_wait(vk);
    if (ret != 0) {
        close(dmabuf_fd);
        return -1;
    }

    *out_fd = dmabuf_fd;
    return 0;
}

/* ---------- Cleanup ---------- */

void vfmcap_vk_cleanup(VulkanCtx *vk)
{
    if (!vk) return;
    if (!vk->initialized) {
        free(vk);
        return;
    }

    vkDeviceWaitIdle(vk->device);

    for (int i = 0; i < vk->input_cache_count; i++) {
        cache_entry_destroy(vk, &vk->input_cache[i]);
    }
    vk->input_cache_count = 0;

    for (int i = 0; i < vk->image_cache_count; i++) {
        image_cache_entry_destroy(vk, &vk->image_cache[i]);
    }
    vk->image_cache_count = 0;

    output_pool_destroy(vk, &vk->pool_nv12_y);
    output_pool_destroy(vk, &vk->pool_nv12_uv);
    output_pool_destroy(vk, &vk->pool_p010_y);
    output_pool_destroy(vk, &vk->pool_p010_uv);
    output_pool_destroy(vk, &vk->pool_nv12_afbc);
    output_pool_destroy(vk, &vk->pool_a2b10g10r10_afbc);
    output_pool_destroy(vk, &vk->pool_intermediate_y);
    output_pool_destroy(vk, &vk->pool_intermediate_uv);

    cache_entry_destroy(vk, &vk->cached_output);

    if (vk->fence != VK_NULL_HANDLE)
        vkDestroyFence(vk->device, vk->fence, NULL);
    if (vk->ycbcr_sampler != VK_NULL_HANDLE)
        vkDestroySampler(vk->device, vk->ycbcr_sampler, NULL);
    if (vk->ycbcr_conversion != VK_NULL_HANDLE)
        vkDestroySamplerYcbcrConversion(vk->device, vk->ycbcr_conversion, NULL);
    if (vk->regular_sampler != VK_NULL_HANDLE)
        vkDestroySampler(vk->device, vk->regular_sampler, NULL);
    if (vk->gfx_pipeline_layout_10bit != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(vk->device, vk->gfx_pipeline_layout_10bit, NULL);
    if (vk->gfx_descriptor_set_layout_10bit != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(vk->device, vk->gfx_descriptor_set_layout_10bit, NULL);
    if (vk->gfx_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(vk->device, vk->gfx_pipeline_layout, NULL);
    if (vk->gfx_descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(vk->device, vk->gfx_descriptor_set_layout, NULL);
    if (vk->gfx_pipeline_nv12_y != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_nv12_y, NULL);
    if (vk->gfx_pipeline_nv12_uv != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_nv12_uv, NULL);
    if (vk->gfx_pipeline_p010_y != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_p010_y, NULL);
    if (vk->gfx_pipeline_p010_uv != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_p010_uv, NULL);
    if (vk->gfx_pipeline_p010_y_10bit != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_p010_y_10bit, NULL);
    if (vk->gfx_pipeline_p010_uv_10bit != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_p010_uv_10bit, NULL);
    if (vk->gfx_pipeline_nv12_y_10bit != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_nv12_y_10bit, NULL);
    if (vk->gfx_pipeline_nv12_uv_10bit != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->gfx_pipeline_nv12_uv_10bit, NULL);
    if (vk->gfx_shader_vert != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_vert, NULL);
    if (vk->gfx_shader_nv12_y != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_nv12_y, NULL);
    if (vk->gfx_shader_nv12_uv != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_nv12_uv, NULL);
    if (vk->gfx_shader_p010_y != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_p010_y, NULL);
    if (vk->gfx_shader_p010_uv != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_p010_uv, NULL);
    if (vk->gfx_shader_nv12_y_from_r16 != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_nv12_y_from_r16, NULL);
    if (vk->gfx_shader_nv12_uv_from_r16g16 != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_nv12_uv_from_r16g16, NULL);
    if (vk->gfx_shader_p010_y_from_r16 != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_p010_y_from_r16, NULL);
    if (vk->gfx_shader_p010_uv_from_r16g16 != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->gfx_shader_p010_uv_from_r16g16, NULL);
    if (vk->pipeline_p010 != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->pipeline_p010, NULL);
    if (vk->pipeline_nv12 != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->pipeline_nv12, NULL);
    if (vk->pipeline_layout_decode != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(vk->device, vk->pipeline_layout_decode, NULL);
    if (vk->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->pipeline_amly_decode != VK_NULL_HANDLE)
        vkDestroyPipeline(vk->device, vk->pipeline_amly_decode, NULL);
    if (vk->shader_amly_decode != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->shader_amly_decode, NULL);
    if (vk->shader_p010 != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->shader_p010, NULL);
    if (vk->shader_nv12 != VK_NULL_HANDLE)
        vkDestroyShaderModule(vk->device, vk->shader_nv12, NULL);
    if (vk->compute_descriptor_set_layout_decode != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(vk->device, vk->compute_descriptor_set_layout_decode, NULL);
    if (vk->descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(vk->device, vk->descriptor_set_layout, NULL);
    if (vk->descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(vk->device, vk->descriptor_pool, NULL);
    if (vk->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device != VK_NULL_HANDLE)
        vkDestroyDevice(vk->device, NULL);
    if (vk->instance != VK_NULL_HANDLE)
        vkDestroyInstance(vk->instance, NULL);

    memset(vk, 0, sizeof(*vk));
    fprintf(stderr, "[vfmcap-vk] Cleanup complete\n");
}

const char *vfmcap_vk_last_error(VulkanCtx *vk)
{
    if (vk)
        return vk->last_error;
    return "No Vulkan context";
}
