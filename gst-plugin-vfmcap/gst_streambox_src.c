/*
 * gst_streambox_src.c - Unified GStreamer HDMI capture source element
 *
 * Dual-path capture from Amlogic A311D2 (T7):
 *
 *   Path A (vfmcap): Zero-copy raw capture via libvfmcap SDK.
 *     /dev/video_cap -> AMLY 10-bit -> Vulkan GPU -> P010/NV12 DMA-buf
 *     Ultra low latency. No color processing. BT.2020 PQ passthrough.
 *
 *   Path B (vdin1): Color-processed capture via VPP loopback.
 *     /dev/video71 -> NV21 8-bit (HDR->SDR already applied by VPP hardware)
 *     Higher latency. Good looking frames out of the box.
 *
 * Pipeline examples:
 *   gst-launch-1.0 streamboxsrc source=vfmcap output-format=p010 ! \
 *       video/x-raw,format=P010_10LE ! amlvenc ! ...
 *   gst-launch-1.0 streamboxsrc source=vdin1 ! \
 *       video/x-raw,format=NV21 ! amlvenc ! ...
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>

#include <vulkan/vulkan.h>
#include <linux/dma-buf.h>

#include "gst_streambox_src.h"
#include "vdin1_amly_to_p010_spv.h"

GST_DEBUG_CATEGORY_STATIC(gst_streambox_src_debug);
#define GST_CAT_DEFAULT gst_streambox_src_debug

/* ---------- Timing helpers ---------- */

static inline guint64
_get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (guint64)ts.tv_sec * 1000000ULL + (guint64)ts.tv_nsec / 1000ULL;
}

/* ---------- Constants ---------- */

#define DEFAULT_DEVICE_VFMCAP  "/dev/video_cap"
#define DEFAULT_DEVICE_VDIN1   "/dev/video71"
#define DEFAULT_NUM_BUFFERS    6
#define DEFAULT_PATHA_POOL_SIZE 12
#define DEFAULT_OUTPUT_FMT     GST_STREAMBOX_OUTPUT_NV12
#define DEFAULT_SOURCE_MODE    GST_STREAMBOX_SOURCE_VFMCAP

#define DMA_HEAP_PATH          "/dev/dma_heap/system-uncached"
#define DMA_HEAP_CMA_PATH      "/dev/dma_heap/heap-codecmm"

/*
 * vdin1 V4L2 input index for VPP post-blend loopback.
 * This captures the full VPP output (after HDR->SDR tone mapping,
 * gamut conversion, and all color management by TruLife Image Engine).
 * Index determined by VIDIOC_ENUMINPUT on /dev/video71.
 */
#define VDIN1_INPUT_VPP_POST_BLEND 6

/* Path to read HDMI RX signal info (active resolution) */
#define HDMIRX_INFO_PATH "/sys/class/hdmirx/hdmirx0/info"

/* Poll vdin1 G_FMT every N frames to detect resolution change */
#define VDIN1_FMT_POLL_INTERVAL 60

/* Signal event actions returned by handle_signal_event() */
typedef enum {
    SIGNAL_EVENT_NONE        = 0,  /* No actionable event */
    SIGNAL_EVENT_CHANGED     = 1,  /* Source changed — need reconfigure */
    SIGNAL_EVENT_LOST        = 2,  /* Signal lost (no lock) */
} SignalEventAction;

/* ---------- Properties ---------- */

enum
{
    PROP_0,
    PROP_SOURCE,
    PROP_DEVICE,
    PROP_NUM_BUFFERS,
    PROP_PATHA_POOL_SIZE,
    PROP_OUTPUT_FORMAT,
    PROP_TARGET_WIDTH,
    PROP_TARGET_HEIGHT,
    PROP_TARGET_FPS,
    PROP_COLOR_MODE,
    PROP_VDIN1_INPUT,
};

/* ---------- GType for enums ---------- */

#define GST_TYPE_STREAMBOX_SOURCE_MODE (gst_streambox_source_mode_get_type())

static GType
gst_streambox_source_mode_get_type(void)
{
    static GType type = 0;
    if (g_once_init_enter(&type)) {
        static const GEnumValue values[] = {
            { GST_STREAMBOX_SOURCE_VFMCAP, "Raw capture via vfm_cap (Path A)", "vfmcap" },
            { GST_STREAMBOX_SOURCE_VDIN1, "Color-processed via vdin1 (Path B)", "vdin1" },
            { 0, NULL, NULL }
        };
        GType tmp = g_enum_register_static("GstStreamboxSourceMode", values);
        g_once_init_leave(&type, tmp);
    }
    return type;
}

#define GST_TYPE_STREAMBOX_OUTPUT_FORMAT (gst_streambox_output_format_get_type())

static GType
gst_streambox_output_format_get_type(void)
{
    static GType type = 0;
    if (g_once_init_enter(&type)) {
        static const GEnumValue values[] = {
            { GST_STREAMBOX_OUTPUT_NV12, "NV12 (8-bit)", "nv12" },
            { GST_STREAMBOX_OUTPUT_P010, "P010 (10-bit)", "p010" },
            { 0, NULL, NULL }
        };
        GType tmp = g_enum_register_static("GstStreamboxOutputFormat", values);
        g_once_init_leave(&type, tmp);
    }
    return type;
}

/* ---------- Pad template ---------- */

/*
 * Source pad template: covers both paths.
 * Path A produces NV12 or P010_10LE (via GPU conversion).
 * Path B produces NV21 (VPP output format).
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, "
        "format = (string) { NV12, P010_10LE, NV21 }, "
        "width = (int) [ 1, 4096 ], "
        "height = (int) [ 1, 2160 ], "
        "framerate = (fraction) [ 0/1, 240/1 ]"
    )
);

/* ---------- GObject boilerplate ---------- */

#define gst_streambox_src_parent_class parent_class
G_DEFINE_TYPE(GstStreamboxSrc, gst_streambox_src, GST_TYPE_PUSH_SRC);

/* ---------- Forward declarations ---------- */

static void     gst_streambox_src_set_property(GObject *object, guint prop_id,
                                                const GValue *value, GParamSpec *pspec);
static void     gst_streambox_src_get_property(GObject *object, guint prop_id,
                                                GValue *value, GParamSpec *pspec);
static void     gst_streambox_src_finalize(GObject *object);

static gboolean gst_streambox_src_start(GstBaseSrc *basesrc);
static gboolean gst_streambox_src_stop(GstBaseSrc *basesrc);
static gboolean gst_streambox_src_unlock(GstBaseSrc *basesrc);
static gboolean gst_streambox_src_unlock_stop(GstBaseSrc *basesrc);
static GstCaps *gst_streambox_src_get_caps(GstBaseSrc *basesrc, GstCaps *filter);
static GstCaps *gst_streambox_src_fixate(GstBaseSrc *basesrc, GstCaps *caps);
static GstFlowReturn gst_streambox_src_create(GstPushSrc *pushsrc, GstBuffer **buf);

/* Path-specific helpers */
static gboolean start_path_a(GstStreamboxSrc *self);
static gboolean start_path_b(GstStreamboxSrc *self);
static void     stop_path_a(GstStreamboxSrc *self);
static void     stop_path_b(GstStreamboxSrc *self);
static GstFlowReturn create_path_a(GstStreamboxSrc *self, GstBuffer **buf);
static GstFlowReturn create_path_b(GstStreamboxSrc *self, GstBuffer **buf);

/* Forward declarations for P010 output pool */
static gboolean p010_pool_init(GstStreamboxSrc *self, guint32 buf_size);
static void     p010_pool_cleanup(GstStreamboxSrc *self);

/* Forward declaration for signal change message */
static void post_signal_change_message(GstStreamboxSrc *self, const gchar *reason);

/* ---------- DMA-heap allocation ---------- */

/*
 * Allocate from /dev/dma_heap/system-uncached (scatter-gather, non-contiguous).
 * NOTE: NOT suitable for encoder hardware input — use alloc_cma_dmabuf instead.
 * Kept for potential future non-encoder DMA-buf use cases.
 */
static int __attribute__((unused))
alloc_output_dmabuf(GstStreamboxSrc *self, guint32 size)
{
    if (self->heap_fd < 0) {
        self->heap_fd = open(DMA_HEAP_PATH, O_RDWR);
        if (self->heap_fd < 0) {
            GST_ERROR_OBJECT(self, "Cannot open %s: %s",
                             DMA_HEAP_PATH, strerror(errno));
            return -1;
        }
    }

    struct dma_heap_allocation_data alloc = {
        .len = size,
        .fd_flags = O_CLOEXEC | O_RDWR,
        .heap_flags = 0,
    };

    if (ioctl(self->heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        GST_ERROR_OBJECT(self, "DMA_HEAP_IOCTL_ALLOC(%u) failed: %s",
                         size, strerror(errno));
        return -1;
    }

    return alloc.fd;
}

/*
 * Allocate a CMA-backed DMA-buf from /dev/dma_heap/heap-codecmm.
 * Required for vdin1 DMABUF mode: the vdin1 driver extracts a single
 * physical address from sg_page(sgl), so the buffer must be physically
 * contiguous (CMA).
 */
static int
alloc_cma_dmabuf(GstStreamboxSrc *self, guint32 size)
{
    if (self->heap_cma_fd < 0) {
        self->heap_cma_fd = open(DMA_HEAP_CMA_PATH, O_RDWR);
        if (self->heap_cma_fd < 0) {
            GST_ERROR_OBJECT(self, "Cannot open %s: %s",
                             DMA_HEAP_CMA_PATH, strerror(errno));
            return -1;
        }
    }

    struct dma_heap_allocation_data alloc = {
        .len = size,
        .fd_flags = O_CLOEXEC | O_RDWR,
        .heap_flags = 0,
    };

    if (ioctl(self->heap_cma_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        GST_ERROR_OBJECT(self, "CMA DMA_HEAP_IOCTL_ALLOC(%u) failed: %s",
                         size, strerror(errno));
        return -1;
    }

    return alloc.fd;
}

/* ====================================================================
 * VULKAN COMPUTE: AMLY -> P010 for Path B 10-bit
 *
 * vdin1 outputs AMLY (40-bit packed YUV422 10-bit, little-endian).
 * We use a Vulkan compute shader to convert to P010 (semi-planar 10-bit).
 * Modeled closely after libvfmcap's vfmcap_vulkan.c but integrated
 * directly into the GStreamer plugin with per-instance state.
 * ==================================================================== */

/* Vulkan error check macro using GST_ERROR_OBJECT */
#define VK_CHECK_GST(self, result, msg) do { \
    if ((result) != VK_SUCCESS) { \
        GST_ERROR_OBJECT(self, "%s: VkResult=%d", msg, (int)(result)); \
        return FALSE; \
    } \
} while(0)

/* ---------- Vulkan helpers ---------- */

static gint
vdin1_vk_find_memory_type(GstStreamboxSrc *self, uint32_t type_filter,
                           VkMemoryPropertyFlags props)
{
    for (uint32_t i = 0; i < self->vk_memory_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (self->vk_memory_props.memoryTypes[i].propertyFlags & props) == props) {
            return (gint)i;
        }
    }
    return -1;
}

/* Import a DMA-buf fd into Vulkan. fd is consumed on success. */
static gboolean
vdin1_vk_import_dmabuf(GstStreamboxSrc *self, int fd, VkDeviceSize size,
                        VkBuffer *buffer, VkDeviceMemory *memory)
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

    VkResult result = vkCreateBuffer(self->vk_device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkCreateBuffer failed: %d", result);
        close(fd);
        return FALSE;
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(self->vk_device, *buffer, &mem_reqs);

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

    gint mem_type = vdin1_vk_find_memory_type(self, mem_reqs.memoryTypeBits, 0);
    if (mem_type < 0) {
        GST_ERROR_OBJECT(self, "No suitable memory type for DMA-buf import");
        vkDestroyBuffer(self->vk_device, *buffer, NULL);
        return FALSE;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = alloc_size,
        .memoryTypeIndex = (uint32_t)mem_type,
    };

    result = vkAllocateMemory(self->vk_device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkAllocateMemory(DMA-buf) failed: %d", result);
        vkDestroyBuffer(self->vk_device, *buffer, NULL);
        return FALSE;
    }

    result = vkBindBufferMemory(self->vk_device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkBindBufferMemory failed: %d", result);
        vkFreeMemory(self->vk_device, *memory, NULL);
        vkDestroyBuffer(self->vk_device, *buffer, NULL);
        return FALSE;
    }

    return TRUE;
}

/* ---------- DMA-buf cache for Vulkan ---------- */

static void
vdin1_vk_cache_entry_destroy(GstStreamboxSrc *self, Vdin1VkCacheEntry *entry)
{
    if (!entry->valid) return;
    vkDestroyBuffer(self->vk_device, entry->buffer, NULL);
    vkFreeMemory(self->vk_device, entry->memory, NULL);
    entry->valid = 0;
    entry->fd = -1;
    entry->fd_dup = -1;
}

/* Get or create cached input DMA-buf import. Returns cache index or -1. */
static gint
vdin1_vk_input_cache_get(GstStreamboxSrc *self, int fd, VkDeviceSize size)
{
    /* Search existing */
    for (gint i = 0; i < self->vk_input_cache_count; i++) {
        if (self->vk_input_cache[i].valid &&
            self->vk_input_cache[i].fd == fd &&
            self->vk_input_cache[i].size == size) {
            self->vk_input_cache[i].last_used = self->vk_frame_count;
            return i;
        }
    }

    /* Find or evict slot */
    gint slot = -1;
    if (self->vk_input_cache_count < VDIN1_VK_DMABUF_CACHE_SIZE) {
        slot = self->vk_input_cache_count++;
    } else {
        guint64 oldest = G_MAXUINT64;
        for (gint i = 0; i < VDIN1_VK_DMABUF_CACHE_SIZE; i++) {
            if (self->vk_input_cache[i].last_used < oldest) {
                oldest = self->vk_input_cache[i].last_used;
                slot = i;
            }
        }
        vdin1_vk_cache_entry_destroy(self, &self->vk_input_cache[slot]);
    }

    /* dup fd because vkAllocateMemory consumes it */
    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        GST_ERROR_OBJECT(self, "dup(input fd %d) failed: %s", fd, strerror(errno));
        return -1;
    }

    VkBuffer buffer;
    VkDeviceMemory memory;
    if (!vdin1_vk_import_dmabuf(self, fd_dup, size, &buffer, &memory)) {
        return -1;
    }

    self->vk_input_cache[slot].fd = fd;
    self->vk_input_cache[slot].fd_dup = fd_dup;
    self->vk_input_cache[slot].buffer = buffer;
    self->vk_input_cache[slot].memory = memory;
    self->vk_input_cache[slot].size = size;
    self->vk_input_cache[slot].valid = 1;
    self->vk_input_cache[slot].last_used = self->vk_frame_count;

    return slot;
}

/* ---------- Vulkan init ---------- */

static gboolean
vdin1_vk_init(GstStreamboxSrc *self)
{
    if (self->vk_initialized)
        return TRUE;

    VkResult result;

    /* Instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "StreamboxSrcVdin1",
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

    result = vkCreateInstance(&instance_info, NULL, &self->vk_instance);
    VK_CHECK_GST(self, result, "vkCreateInstance");

    /* Physical device selection — find compute queue */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(self->vk_instance, &dev_count, NULL);
    if (dev_count == 0) {
        GST_ERROR_OBJECT(self, "No Vulkan physical devices found");
        return FALSE;
    }

    VkPhysicalDevice *devices = g_new(VkPhysicalDevice, dev_count);
    vkEnumeratePhysicalDevices(self->vk_instance, &dev_count, devices);

    self->vk_physical_device = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < dev_count; i++) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, NULL);
        VkQueueFamilyProperties *qf_props = g_new(VkQueueFamilyProperties, qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, qf_props);

        for (uint32_t j = 0; j < qf_count; j++) {
            if (qf_props[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                self->vk_physical_device = devices[i];
                self->vk_queue_family = j;
                break;
            }
        }
        g_free(qf_props);
        if (self->vk_physical_device != VK_NULL_HANDLE)
            break;
    }
    g_free(devices);

    if (self->vk_physical_device == VK_NULL_HANDLE) {
        GST_ERROR_OBJECT(self, "No Vulkan device with compute support");
        return FALSE;
    }

    vkGetPhysicalDeviceMemoryProperties(self->vk_physical_device,
                                         &self->vk_memory_props);

    /* Logical device */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = self->vk_queue_family,
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

    result = vkCreateDevice(self->vk_physical_device, &device_info, NULL,
                            &self->vk_device);
    VK_CHECK_GST(self, result, "vkCreateDevice");

    vkGetDeviceQueue(self->vk_device, self->vk_queue_family, 0,
                     &self->vk_compute_queue);

    /* Command pool */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = self->vk_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    };

    result = vkCreateCommandPool(self->vk_device, &pool_info, NULL,
                                  &self->vk_command_pool);
    VK_CHECK_GST(self, result, "vkCreateCommandPool");

    /* Command buffers (2 for double-buffered async pipeline) */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = self->vk_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2,
    };

    result = vkAllocateCommandBuffers(self->vk_device, &cmd_alloc,
                                       self->vk_cmd);
    VK_CHECK_GST(self, result, "vkAllocateCommandBuffers(2)");

    /* Fences (2, start signaled for first reset) */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (int fi = 0; fi < 2; fi++) {
        result = vkCreateFence(self->vk_device, &fence_info, NULL, &self->vk_fences[fi]);
        VK_CHECK_GST(self, result, "vkCreateFence");
    }
    self->vk_slot = 0;

    /* Descriptor pool: 3 storage buffers x 2 sets (double-buffered) */
    VkDescriptorPoolSize desc_pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 },
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 1,
        .pPoolSizes = desc_pool_sizes,
    };

    result = vkCreateDescriptorPool(self->vk_device, &desc_pool_info, NULL,
                                     &self->vk_descriptor_pool);
    VK_CHECK_GST(self, result, "vkCreateDescriptorPool");

    /* Descriptor set layout: 3 storage buffers (input, Y-out, UV-out) */
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

    result = vkCreateDescriptorSetLayout(self->vk_device, &layout_info, NULL,
                                          &self->vk_descriptor_set_layout);
    VK_CHECK_GST(self, result, "vkCreateDescriptorSetLayout");

    /* Descriptor sets (2 for double-buffered pipeline) */
    VkDescriptorSetLayout layouts[2] = {
        self->vk_descriptor_set_layout,
        self->vk_descriptor_set_layout,
    };
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = self->vk_descriptor_pool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts,
    };

    result = vkAllocateDescriptorSets(self->vk_device, &desc_alloc,
                                       self->vk_descriptor_sets);
    VK_CHECK_GST(self, result, "vkAllocateDescriptorSets(2)");

    /* Pipeline layout: push constants = { width, height, pairs_per_row, reserved } */
    VkPushConstantRange push_constant = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 4,
    };

    VkPipelineLayoutCreateInfo pl_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &self->vk_descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant,
    };

    result = vkCreatePipelineLayout(self->vk_device, &pl_layout_info, NULL,
                                     &self->vk_pipeline_layout);
    VK_CHECK_GST(self, result, "vkCreatePipelineLayout");

    /* Load P010 shader */
    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(vdin1_amly_to_p010_spv),
        .pCode = (const uint32_t *)vdin1_amly_to_p010_spv,
    };

    result = vkCreateShaderModule(self->vk_device, &shader_info, NULL,
                                   &self->vk_shader_p010);
    VK_CHECK_GST(self, result, "vkCreateShaderModule(P010)");

    /* Create P010 compute pipeline */
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = self->vk_shader_p010,
            .pName = "main",
        },
        .layout = self->vk_pipeline_layout,
    };

    result = vkCreateComputePipelines(self->vk_device, VK_NULL_HANDLE, 1,
                                       &pipeline_info, NULL,
                                       &self->vk_pipeline_p010);
    VK_CHECK_GST(self, result, "vkCreateComputePipelines(P010)");

    /* Init caches */
    self->vk_input_cache_count = 0;
    self->vk_output_cache.valid = 0;
    self->vk_output_cache.fd = -1;
    self->vk_pending_in_fd = -1;
    self->vk_has_pending = FALSE;
    self->vk_frame_count = 0;
    self->vk_async_pending = FALSE;
    self->vk_async_out_pool_idx = -1;
    self->vk_async_vdin1_idx = 0;
    self->vk_async_in_fd = -1;
    self->vk_async_pts = 0;
    self->vk_async_duration = 0;

    for (gint i = 0; i < VDIN1_VK_DMABUF_CACHE_SIZE; i++) {
        self->vk_input_cache[i].valid = 0;
        self->vk_input_cache[i].fd = -1;
    }

    self->vk_initialized = TRUE;

    GST_INFO_OBJECT(self, "Vulkan P010 pipeline initialized for vdin1 10-bit");
    return TRUE;
}

/* ---------- Vulkan async submit (non-blocking) ---------- */

/*
 * Record and submit GPU work for AMLY->P010 conversion on the given slot.
 * Returns immediately after vkQueueSubmit — does NOT wait for the fence.
 * Caller must later call vdin1_vk_wait_async() before reusing this slot.
 */
static gboolean
vdin1_vk_submit_async(GstStreamboxSrc *self, int in_fd, guint out_pool_idx,
                       guint32 width, guint32 height, guint slot)
{
    if (!self->vk_initialized) {
        GST_ERROR_OBJECT(self, "Vulkan not initialized");
        return FALSE;
    }

    if (out_pool_idx >= self->p010_out_count ||
        !self->vk_output_pool_cache[out_pool_idx].valid) {
        GST_ERROR_OBJECT(self, "Invalid output pool index %u", out_pool_idx);
        return FALSE;
    }

    VkResult result;

    /* Calculate buffer sizes */
    VkDeviceSize input_size = (VkDeviceSize)width * height * 5 / 2;  /* AMLY: 20 bpp */
    VkDeviceSize y_plane_size = (VkDeviceSize)width * height * 2;    /* P010 Y: 16-bit */
    VkDeviceSize uv_plane_size = (VkDeviceSize)width * height;       /* P010 UV: 16-bit, half height */

    /* Import input DMA-buf (cached) */
    gint in_idx = vdin1_vk_input_cache_get(self, in_fd, input_size);
    if (in_idx < 0) return FALSE;

    /* DMA_BUF_SYNC start read */
    struct dma_buf_sync sync_start = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
    };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_start);

    VkBuffer in_buffer = self->vk_input_cache[in_idx].buffer;
    VkBuffer out_buffer = self->vk_output_pool_cache[out_pool_idx].buffer;

    VkCommandBuffer cmd = self->vk_cmd[slot];
    VkFence fence = self->vk_fences[slot];
    VkDescriptorSet desc_set = self->vk_descriptor_sets[slot];

    /* Reset fence */
    vkResetFences(self->vk_device, 1, &fence);

    /* Record command buffer */
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    result = vkResetCommandBuffer(cmd, 0);
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkResetCommandBuffer[%u] failed: %d", slot, result);
        goto sync_end;
    }

    result = vkBeginCommandBuffer(cmd, &begin_info);
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkBeginCommandBuffer[%u] failed: %d", slot, result);
        goto sync_end;
    }

    /* Update descriptors for this slot */
    VkDescriptorBufferInfo buffer_infos[] = {
        { in_buffer, 0, input_size },
        { out_buffer, 0, y_plane_size },
        { out_buffer, y_plane_size, uv_plane_size },
    };

    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = desc_set, .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = desc_set, .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = desc_set, .dstBinding = 2,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buffer_infos[2] },
    };

    vkUpdateDescriptorSets(self->vk_device, 3, writes, 0, NULL);

    /* Bind pipeline and descriptors */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      self->vk_pipeline_p010);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            self->vk_pipeline_layout, 0, 1,
                            &desc_set, 0, NULL);

    /* Push constants: { width, height, pairs_per_row, reserved } */
    uint32_t pairs_per_row = width / 2;
    uint32_t push_data[] = { width, height, pairs_per_row, 0u };
    vkCmdPushConstants(cmd, self->vk_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

    /* Dispatch: local_size=(128,1,1), groups_x = ceil(pairs_per_row/128), groups_y = height */
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
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkEndCommandBuffer[%u] failed: %d", slot, result);
        goto sync_end;
    }

    /* Submit (non-blocking) */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    result = vkQueueSubmit(self->vk_compute_queue, 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkQueueSubmit[%u] failed: %d", slot, result);
        goto sync_end;
    }

    return TRUE;

sync_end:
    {
        struct dma_buf_sync sync_end_s = {
            .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ
        };
        ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_end_s);
    }
    return FALSE;
}

/* ---------- Vulkan async wait (blocking) ---------- */

/*
 * Wait for previously submitted GPU work on the given slot to complete.
 * Also releases DMA_BUF_SYNC on the input fd.
 */
static gboolean
vdin1_vk_wait_async(GstStreamboxSrc *self, guint slot, int in_fd)
{
    VkFence fence = self->vk_fences[slot];

    /* Wait for completion (5 second timeout) */
    VkResult result = vkWaitForFences(self->vk_device, 1, &fence,
                                       VK_TRUE, 5000000000ULL);

    /* Release DMA-buf read access */
    struct dma_buf_sync sync_end_s = {
        .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ
    };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_end_s);

    if (result != VK_SUCCESS) {
        GST_ERROR_OBJECT(self, "vkWaitForFences[%u] failed: %d (frame %lu)",
                         slot, result, (unsigned long)self->vk_frame_count);
        return FALSE;
    }

    return TRUE;
}

/* ---------- Vulkan synchronous convert (for priming frame) ---------- */

static gboolean
vdin1_vk_convert_sync(GstStreamboxSrc *self, int in_fd, guint out_pool_idx,
                       guint32 width, guint32 height)
{
    guint slot = self->vk_slot;

    guint64 t0 = _get_time_us();

    if (!vdin1_vk_submit_async(self, in_fd, out_pool_idx, width, height, slot))
        return FALSE;

    guint64 t1 = _get_time_us();

    if (!vdin1_vk_wait_async(self, slot, in_fd))
        return FALSE;

    guint64 t2 = _get_time_us();

    if (self->vk_frame_count < 10 || self->vk_frame_count % 100 == 0) {
        GST_LOG_OBJECT(self,
            "VK TIMING frame %lu (sync): submit=%luus fence_wait=%luus TOTAL=%luus",
            (unsigned long)self->vk_frame_count,
            (unsigned long)(t1 - t0),
            (unsigned long)(t2 - t1),
            (unsigned long)(t2 - t0));
    }

    self->vk_frame_count++;
    return TRUE;
}

/* ---------- Vulkan cleanup ---------- */

static void
vdin1_vk_cleanup(GstStreamboxSrc *self)
{
    if (!self->vk_initialized) return;

    vkDeviceWaitIdle(self->vk_device);

    for (gint i = 0; i < self->vk_input_cache_count; i++) {
        vdin1_vk_cache_entry_destroy(self, &self->vk_input_cache[i]);
    }
    self->vk_input_cache_count = 0;

    vdin1_vk_cache_entry_destroy(self, &self->vk_output_cache);

    for (int fi = 0; fi < 2; fi++) {
        if (self->vk_fences[fi] != VK_NULL_HANDLE)
            vkDestroyFence(self->vk_device, self->vk_fences[fi], NULL);
        self->vk_fences[fi] = VK_NULL_HANDLE;
    }
    if (self->vk_pipeline_p010 != VK_NULL_HANDLE)
        vkDestroyPipeline(self->vk_device, self->vk_pipeline_p010, NULL);
    if (self->vk_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(self->vk_device, self->vk_pipeline_layout, NULL);
    if (self->vk_shader_p010 != VK_NULL_HANDLE)
        vkDestroyShaderModule(self->vk_device, self->vk_shader_p010, NULL);
    if (self->vk_descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(self->vk_device,
                                      self->vk_descriptor_set_layout, NULL);
    if (self->vk_descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(self->vk_device, self->vk_descriptor_pool, NULL);
    if (self->vk_command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(self->vk_device, self->vk_command_pool, NULL);
    if (self->vk_device != VK_NULL_HANDLE)
        vkDestroyDevice(self->vk_device, NULL);
    if (self->vk_instance != VK_NULL_HANDLE)
        vkDestroyInstance(self->vk_instance, NULL);

    self->vk_instance = VK_NULL_HANDLE;
    self->vk_device = VK_NULL_HANDLE;
    self->vk_physical_device = VK_NULL_HANDLE;
    self->vk_pipeline_p010 = VK_NULL_HANDLE;
    self->vk_pipeline_layout = VK_NULL_HANDLE;
    self->vk_shader_p010 = VK_NULL_HANDLE;
    self->vk_descriptor_set_layout = VK_NULL_HANDLE;
    self->vk_descriptor_pool = VK_NULL_HANDLE;
    self->vk_command_pool = VK_NULL_HANDLE;
    self->vk_initialized = FALSE;
    self->vk_async_pending = FALSE;

    GST_INFO_OBJECT(self, "Vulkan cleanup complete (%lu frames converted)",
                     (unsigned long)self->vk_frame_count);
}

/* ---------- V4L2 helpers (Path B) ---------- */

static int
xioctl(int fd, unsigned long request, void *arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

/*
 * Query HDMI RX source resolution from sysfs.
 * Parses /sys/class/hdmirx/hdmirx0/info for Hactive/Vactive lines.
 * Returns TRUE if a valid resolution was found.
 */
static gboolean
hdmirx_get_source_resolution(GstStreamboxSrc *self, guint *w, guint *h)
{
    FILE *f = fopen(HDMIRX_INFO_PATH, "r");
    if (!f) {
        GST_WARNING_OBJECT(self, "Cannot open %s: %s",
                           HDMIRX_INFO_PATH, strerror(errno));
        return FALSE;
    }

    gboolean got_w = FALSE, got_h = FALSE;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        guint val;
        if (sscanf(line, "Hactive: %u", &val) == 1) {
            *w = val;
            got_w = TRUE;
        } else if (sscanf(line, "Vactive: %u", &val) == 1) {
            *h = val;
            got_h = TRUE;
        }
        if (got_w && got_h)
            break;
    }
    fclose(f);

    if (got_w && got_h && *w > 0 && *h > 0) {
        GST_INFO_OBJECT(self, "HDMI RX source: %ux%u", *w, *h);
        return TRUE;
    }

    GST_WARNING_OBJECT(self, "Could not parse HDMI RX resolution from %s",
                       HDMIRX_INFO_PATH);
    return FALSE;
}

/*
 * Read HDMI RX frame rate from sysfs info.
 * Returns the frame rate as a GStreamer fraction (fps_n/fps_d).
 * The sysfs "Frame Rate:" value is in hundredths (e.g. 5992 = 59.92 Hz,
 * 14400 = 144.00 Hz, 24000 = 240.00 Hz).
 * Returns TRUE if successfully read, FALSE otherwise.
 */
static gboolean
hdmirx_get_source_framerate(GstStreamboxSrc *self, guint *fps_n, guint *fps_d)
{
    FILE *f = fopen(HDMIRX_INFO_PATH, "r");
    if (!f) {
        GST_WARNING_OBJECT(self, "Cannot open %s for framerate: %s",
                           HDMIRX_INFO_PATH, strerror(errno));
        return FALSE;
    }

    guint frame_rate = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        guint val;
        if (sscanf(line, "Frame Rate: %u", &val) == 1) {
            frame_rate = val;
            break;
        }
    }
    fclose(f);

    if (frame_rate > 0) {
        /* Convert from hundredths to fraction.
         * Common values: 5994 -> 60000/1001, 2997 -> 30000/1001,
         * 6000 -> 60/1, 12000 -> 120/1, 14400 -> 144/1, 24000 -> 240/1
         */
        if (frame_rate % 100 == 0) {
            *fps_n = frame_rate / 100;
            *fps_d = 1;
        } else if (frame_rate == 5994 || frame_rate == 2997 ||
                   frame_rate == 11988 || frame_rate == 23976) {
            /* NTSC fractional rates */
            *fps_n = (frame_rate + 6) / 100 * 1000; /* round up: 5994->60000 */
            *fps_d = 1001;
        } else {
            /* Generic: use frame_rate/100 as integer fps */
            *fps_n = (frame_rate + 50) / 100;
            *fps_d = 1;
        }
        GST_INFO_OBJECT(self, "HDMI RX frame rate: %u (raw) -> %u/%u fps",
                        frame_rate, *fps_n, *fps_d);
        return TRUE;
    }

    GST_WARNING_OBJECT(self, "Could not parse HDMI RX frame rate from %s",
                       HDMIRX_INFO_PATH);
    return FALSE;
}

/*
 * Read HDMI RX signal info and determine the GStreamer colorimetry string
 * for the current capture path.
 *
 * Parses /sys/class/hdmirx/hdmirx0/info for:
 *   - Color Space: e.g. "0-RGB", "1-YUV422", "2-YUV444", "3-YUV420"
 *   - Color Depth: 8, 10, or 12
 *   - HDR EOTF: e.g. "SDR", "SMPTE_ST_2084" (PQ), "HLG", "HDR10PLUS"
 *
 * GStreamer colorimetry format: "range/matrix/transfer/primaries"
 * where each component can be a GstVideoColorimetry enum name.
 *
 * Mapping logic:
 *   Path A (vfmcap, raw passthrough, no color processing):
 *     - Always preserves original colorimetry from source
 *     - HDR PQ source  -> bt2100-pq  (BT.2020 primaries, PQ transfer, BT.2020 matrix)
 *     - HDR HLG source -> bt2100-hlg (BT.2020 primaries, HLG transfer, BT.2020 matrix)
 *     - SDR source     -> bt709 or bt601 depending on resolution
 *
 *   Path B 8-bit (vdin1, NV21, VPP color-processed):
 *     - VPP has already done HDR->SDR tone mapping and BT.2020->BT.709 gamut mapping
 *     - Output is always BT.709/sRGB regardless of input
 *
 *   Path B 10-bit (vdin1, AMLY->P010, raw capture before VPP color processing):
 *     - Same as Path A: preserves original source colorimetry
 *
 * Stores result in self->colorimetry[].
 */
static void
hdmirx_detect_colorimetry(GstStreamboxSrc *self)
{
    guint color_depth = 8;
    gchar hdr_eotf[64] = "SDR";
    gchar color_space[64] = "";

    /* Parse HDMI RX sysfs */
    FILE *f = fopen(HDMIRX_INFO_PATH, "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            guint val;
            if (sscanf(line, "Color Depth: %u", &val) == 1) {
                color_depth = val;
            } else if (sscanf(line, " HDR EOTF: %63[^\n]", hdr_eotf) == 1) {
                /* captured */
            } else if (sscanf(line, " Color Space: %63[^\n]", color_space) == 1) {
                /* captured */
            }
        }
        fclose(f);
    } else {
        GST_WARNING_OBJECT(self, "Cannot read %s for colorimetry detection: %s",
                           HDMIRX_INFO_PATH, strerror(errno));
    }

    GST_INFO_OBJECT(self, "HDMI RX signal: color_depth=%u hdr_eotf=%s color_space=%s",
                     color_depth, hdr_eotf, color_space);

    /*
     * Determine colorimetry based on capture path and signal info.
     *
     * GStreamer colorimetry string format: "range/matrix/transfer/primaries"
     *
     * For HDR10 PQ (SMPTE ST 2084):
     *   bt2100-pq = "full/bt2020/smpte2084/bt2020"
     *   (GstVideoColorimetry has GST_VIDEO_COLORIMETRY_BT2100_PQ = "bt2100-pq")
     *
     * For HLG:
     *   bt2100-hlg = "full/bt2020/arib-std-b67/bt2020"
     *   (GstVideoColorimetry has GST_VIDEO_COLORIMETRY_BT2100_HLG = "bt2100-hlg")
     *
     * For SDR BT.709:
     *   bt709 = "limited/bt709/bt709/bt709"
     *   (GstVideoColorimetry has GST_VIDEO_COLORIMETRY_BT709 = "bt709")
     *
     * For SDR BT.601 (SD content):
     *   smpte240m = "limited/smpte240m/smpte240m/smpte240m" or
     *   bt601 = "limited/bt601/bt709/bt470bg" -- but bt709 is fine for SD over HDMI
     */

    if (self->source_mode == GST_STREAMBOX_SOURCE_VDIN1 && !self->vdin1_10bit) {
        /*
         * Path B 8-bit: VPP has performed HDR->SDR tone mapping and
         * BT.2020->BT.709 gamut mapping.  Output is always BT.709/SDR
         * regardless of the original HDMI source color space.
         */
        g_strlcpy(self->colorimetry, "bt709", sizeof(self->colorimetry));
        GST_INFO_OBJECT(self, "Path B 8-bit: colorimetry=bt709 (VPP color-processed)");
    } else {
        /*
         * Path A (all formats) and Path B 10-bit: raw capture, no color
         * processing.  Preserve original source colorimetry.
         */
        gboolean is_hdr_pq = (g_strstr_len(hdr_eotf, -1, "2084") != NULL ||
                               g_strstr_len(hdr_eotf, -1, "HDR10") != NULL ||
                               g_strstr_len(hdr_eotf, -1, "SMPTE_ST_2048") != NULL);
        gboolean is_hlg = (g_strstr_len(hdr_eotf, -1, "HLG") != NULL);

        if (is_hdr_pq) {
            /*
             * HDR10 PQ: BT.2020 primaries, SMPTE ST 2084 (PQ) transfer, BT.2020 matrix.
             * Use the standard GStreamer "bt2100-pq" colorimetry name which expands to
             * the full colorimetry needed by H.265 VUI for correct HDR playback.
             */
            g_strlcpy(self->colorimetry, "bt2100-pq", sizeof(self->colorimetry));
            GST_INFO_OBJECT(self, "HDR PQ source: colorimetry=bt2100-pq");
        } else if (is_hlg) {
            g_strlcpy(self->colorimetry, "bt2100-hlg", sizeof(self->colorimetry));
            GST_INFO_OBJECT(self, "HLG source: colorimetry=bt2100-hlg");
        } else if (color_depth > 8) {
            /*
             * 10/12-bit SDR: likely BT.2020 container even without HDR transfer.
             * Use BT.2020 primaries/matrix with BT.709 transfer (SDR in wide gamut).
             */
            g_strlcpy(self->colorimetry, "bt2020", sizeof(self->colorimetry));
            GST_INFO_OBJECT(self, "10-bit SDR source: colorimetry=bt2020");
        } else {
            /* 8-bit SDR: standard BT.709 */
            g_strlcpy(self->colorimetry, "bt709", sizeof(self->colorimetry));
            GST_INFO_OBJECT(self, "SDR source: colorimetry=bt709");
        }
    }

    /* Debug override: STREAMBOX_COLORIMETRY env var forces a specific string.
     * Useful to test whether colour corruption is a signalling mismatch
     * (e.g. STREAMBOX_COLORIMETRY=bt709 while HDMI signals HDR PQ). */
    {
        const gchar *ovr = g_getenv("STREAMBOX_COLORIMETRY");
        if (ovr && *ovr) {
            g_strlcpy(self->colorimetry, ovr, sizeof(self->colorimetry));
            GST_WARNING_OBJECT(self, "colorimetry forced by env: %s", ovr);
        }
    }
}

/*
 * Select V4L2 input on vdin1.
 * For VPP post-blend loopback, use VDIN1_INPUT_VPP_POST_BLEND.
 * This sets the driver's work_mode to V4L and configures the loopback port.
 */
static gboolean
vdin1_set_input(GstStreamboxSrc *self, guint input_index)
{
    int idx = (int)input_index;
    if (xioctl(self->vdin1_fd, VIDIOC_S_INPUT, &idx) < 0) {
        GST_ERROR_OBJECT(self, "VIDIOC_S_INPUT(%u) failed: %s",
                         input_index, strerror(errno));
        return FALSE;
    }
    GST_INFO_OBJECT(self, "vdin1: selected input %u",
                    input_index);
    return TRUE;
}

/* Get current format from vdin1 via VIDIOC_G_FMT (multi-plane) */
static gboolean
vdin1_get_format(GstStreamboxSrc *self, guint *w, guint *h, guint32 *pixfmt,
                 guint *num_planes)
{
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    if (xioctl(self->vdin1_fd, VIDIOC_G_FMT, &fmt) < 0) {
        GST_WARNING_OBJECT(self, "VIDIOC_G_FMT failed: %s", strerror(errno));
        return FALSE;
    }

    *w = fmt.fmt.pix_mp.width;
    *h = fmt.fmt.pix_mp.height;
    *pixfmt = fmt.fmt.pix_mp.pixelformat;
    *num_planes = fmt.fmt.pix_mp.num_planes;

    GST_DEBUG_OBJECT(self, "vdin1 G_FMT: %ux%u pixfmt=0x%08x planes=%u",
                     *w, *h, *pixfmt, *num_planes);
    return TRUE;
}

/*
 * Set format on vdin1 via VIDIOC_S_FMT (multi-plane).
 * 8-bit mode: Request NV21 (VPP-processed output).
 * 10-bit mode (output-format=p010): Request AMLY (40-bit packed YUV422 10-bit).
 */
#define V4L2_PIX_FMT_AMLY v4l2_fourcc('A','M','L','Y')

static gboolean
vdin1_set_format(GstStreamboxSrc *self, guint w, guint h)
{
    gboolean want_10bit = (self->output_fmt == GST_STREAMBOX_OUTPUT_P010);

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = w;
    fmt.fmt.pix_mp.height = h;
    fmt.fmt.pix_mp.pixelformat = want_10bit ? V4L2_PIX_FMT_AMLY : V4L2_PIX_FMT_NV21;
    fmt.fmt.pix_mp.num_planes = 1;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;

    GST_INFO_OBJECT(self, "vdin1: requesting %s format %ux%u",
                     want_10bit ? "AMLY (10-bit)" : "NV21 (8-bit)", w, h);

    if (xioctl(self->vdin1_fd, VIDIOC_S_FMT, &fmt) < 0) {
        GST_ERROR_OBJECT(self, "VIDIOC_S_FMT failed: %s", strerror(errno));
        return FALSE;
    }

    /* Re-read what driver actually set (captures sizeimage with padding) */
    guint np = 0;
    if (!vdin1_get_format(self, &self->width, &self->height,
                          &self->vdin1_pixfmt, &np))
        return FALSE;

    /* Capture sizeimage from the format the driver returned */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(self->vdin1_fd, VIDIOC_G_FMT, &fmt) == 0) {
        self->vdin1_sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
    } else {
        /* Fallback based on format */
        if (want_10bit)
            self->vdin1_sizeimage = w * h * 5 / 2;  /* AMLY: 20 bpp */
        else
            self->vdin1_sizeimage = w * h * 3 / 2;  /* NV21: 12 bpp */
    }

    /* Check if we actually got AMLY when we asked for 10-bit */
    self->vdin1_10bit = (self->vdin1_pixfmt == V4L2_PIX_FMT_AMLY);

    if (want_10bit && !self->vdin1_10bit) {
        GST_WARNING_OBJECT(self, "Requested AMLY but got pixfmt=0x%08x, "
                           "falling back to 8-bit mode", self->vdin1_pixfmt);
    }

    if (self->vdin1_10bit) {
        self->vdin1_amly_sizeimage = self->vdin1_sizeimage;
        GST_INFO_OBJECT(self, "vdin1: AMLY 10-bit sizeimage=%u "
                         "(expected=%u, bytesperline=%u)",
                         self->vdin1_sizeimage,
                         self->width * self->height * 5 / 2,
                         self->width * 5 / 2);
    } else {
        self->vdin1_amly_sizeimage = 0;
        GST_INFO_OBJECT(self, "vdin1: NV21 sizeimage=%u (actual NV21=%u, padding=%u)",
                         self->vdin1_sizeimage,
                         self->width * self->height * 3 / 2,
                         self->vdin1_sizeimage - self->width * self->height * 3 / 2);
    }

    return TRUE;
}

static gboolean
vdin1_reqbufs(GstStreamboxSrc *self, guint count)
{
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_DMABUF;

    if (xioctl(self->vdin1_fd, VIDIOC_REQBUFS, &req) < 0) {
        GST_ERROR_OBJECT(self, "VIDIOC_REQBUFS(%u, DMABUF) failed: %s",
                         count, strerror(errno));
        return FALSE;
    }

    if (req.count < 3) {
        GST_ERROR_OBJECT(self, "Insufficient buffer count: %u (need >= 3)",
                         req.count);
        return FALSE;
    }

    self->vdin1_n_bufs = req.count;
    GST_INFO_OBJECT(self, "vdin1: DMABUF mode, %u buffers", req.count);
    return TRUE;
}

/*
 * Allocate DMA-bufs from /dev/dma_heap/system-uncached for vdin1 DMABUF mode.
 * vdin1 will DMA-write directly into these buffers — zero CPU memcpy.
 * Must use driver's sizeimage (includes alignment padding).
 */
static gboolean
vdin1_alloc_dmabufs(GstStreamboxSrc *self)
{
    guint32 buf_size = self->vdin1_sizeimage;

    for (guint i = 0; i < self->vdin1_n_bufs; i++) {
        int fd = alloc_cma_dmabuf(self, buf_size);
        if (fd < 0) {
            GST_ERROR_OBJECT(self, "Failed to allocate DMA-buf %u (%u bytes)",
                             i, buf_size);
            return FALSE;
        }
        self->vdin1_bufs[i].dma_fd = fd;
        self->vdin1_bufs[i].size = buf_size;
        self->vdin1_bufs[i].queued = FALSE;

        GST_DEBUG_OBJECT(self, "vdin1: DMA-buf %u fd=%d size=%u",
                         i, fd, buf_size);
    }
    return TRUE;
}

static void
vdin1_free_dmabufs(GstStreamboxSrc *self)
{
    for (guint i = 0; i < self->vdin1_n_bufs; i++) {
        if (self->vdin1_bufs[i].dma_fd >= 0) {
            close(self->vdin1_bufs[i].dma_fd);
            self->vdin1_bufs[i].dma_fd = -1;
        }
        self->vdin1_bufs[i].queued = FALSE;
    }
    self->vdin1_n_bufs = 0;
}

static gboolean
vdin1_qbuf_all(GstStreamboxSrc *self)
{
    for (guint i = 0; i < self->vdin1_n_bufs; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        memset(&buf, 0, sizeof(buf));
        memset(&planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_DMABUF;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 1;
        planes[0].m.fd = self->vdin1_bufs[i].dma_fd;
        planes[0].length = self->vdin1_bufs[i].size;

        if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &buf) < 0) {
            GST_ERROR_OBJECT(self, "VIDIOC_QBUF(%u, fd=%d) failed: %s",
                             i, self->vdin1_bufs[i].dma_fd, strerror(errno));
            return FALSE;
        }
        self->vdin1_bufs[i].queued = TRUE;
    }
    return TRUE;
}

static gboolean
vdin1_streamon(GstStreamboxSrc *self)
{
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(self->vdin1_fd, VIDIOC_STREAMON, &type) < 0) {
        GST_ERROR_OBJECT(self, "VIDIOC_STREAMON failed: %s", strerror(errno));
        return FALSE;
    }
    GST_INFO_OBJECT(self, "vdin1: STREAMON");
    return TRUE;
}

static void
vdin1_streamoff(GstStreamboxSrc *self)
{
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (self->vdin1_fd >= 0)
        xioctl(self->vdin1_fd, VIDIOC_STREAMOFF, &type);
    GST_INFO_OBJECT(self, "vdin1: STREAMOFF");
}

/* ---------- decide_allocation ----------
 * We manage our own DMA-buf allocation in create_path_a/create_path_b,
 * so we don't need basesrc to set up a bufferpool.  Returning TRUE
 * without configuring a pool tells basesrc to skip pool activation.
 */
static gboolean
gst_streambox_src_decide_allocation(GstBaseSrc *src, GstQuery *query)
{
    /* Remove any pools proposed by downstream — we allocate ourselves */
    while (gst_query_get_n_allocation_pools(query) > 0)
        gst_query_remove_nth_allocation_pool(query, 0);

    return TRUE;
}

/* ---------- Class init ---------- */

static void
gst_streambox_src_class_init(GstStreamboxSrcClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS(klass);
    GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = gst_streambox_src_set_property;
    gobject_class->get_property = gst_streambox_src_get_property;
    gobject_class->finalize = gst_streambox_src_finalize;

    g_object_class_install_property(gobject_class, PROP_SOURCE,
        g_param_spec_enum("source", "Capture Source",
            "Capture path: vfmcap (raw, low latency) or vdin1 (color-processed)",
            GST_TYPE_STREAMBOX_SOURCE_MODE,
            DEFAULT_SOURCE_MODE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_DEVICE,
        g_param_spec_string("device", "Device",
            "V4L2 device path (auto-detected from source if not set)",
            NULL,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_NUM_BUFFERS,
        g_param_spec_uint("capture-buffers", "Capture Buffers",
            "Number of V4L2 capture buffers",
            2, 16, DEFAULT_NUM_BUFFERS,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_PATHA_POOL_SIZE,
        g_param_spec_uint("patha-pool-size", "Path A Pool Size",
            "Number of preallocated Path A output DMA-bufs for downstream processing",
            6, PATHA_OUT_POOL_SIZE_MAX, DEFAULT_PATHA_POOL_SIZE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_OUTPUT_FORMAT,
        g_param_spec_enum("output-format", "Output Format",
            "Output pixel format: NV12/NV21 (8-bit) or P010 (10-bit). "
            "Path A: GPU converts AMLY->NV12 or P010. "
            "Path B: NV21 zero-copy (8-bit) or GPU converts AMLY->P010 (10-bit).",
            GST_TYPE_STREAMBOX_OUTPUT_FORMAT,
            DEFAULT_OUTPUT_FMT,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_TARGET_WIDTH,
        g_param_spec_uint("target-width", "Target Width",
            "Target output width in pixels (0 = match source)",
            0, 8192, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_TARGET_HEIGHT,
        g_param_spec_uint("target-height", "Target Height",
            "Target output height in pixels (0 = match source)",
            0, 8192, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_TARGET_FPS,
        g_param_spec_float("target-fps", "Target FPS",
            "Target output framerate in fps (0 = match source)",
            0.0f, 240.0f, 0.0f,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_COLOR_MODE,
        g_param_spec_uint("color-mode", "Color Mode",
            "HDR/color conversion mode: 0=passthrough, 1=HDR10->SDR, 2=HLG->SDR",
            0, 2, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_VDIN1_INPUT,
        g_param_spec_uint("vdin1-input", "VDIN1 Input Index",
            "V4L2 input index for vdin1 loopback source. "
            "0=VD1, 1=VD2, 3=OSD1, 4=OSD2, 6=VPP post-blend (default). "
            "See VIDIOC_ENUMINPUT on /dev/video71 for available inputs.",
            0, 15, VDIN1_INPUT_VPP_POST_BLEND,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    gst_element_class_set_static_metadata(element_class,
        "StreamBox HDMI Capture Source",
        "Source/Video",
        "Dual-path HDMI capture: raw (vfm_cap) or color-processed (vdin1)",
        "StreamBox");

    gst_element_class_add_static_pad_template(element_class, &src_template);

    basesrc_class->start = GST_DEBUG_FUNCPTR(gst_streambox_src_start);
    basesrc_class->stop = GST_DEBUG_FUNCPTR(gst_streambox_src_stop);
    basesrc_class->unlock = GST_DEBUG_FUNCPTR(gst_streambox_src_unlock);
    basesrc_class->unlock_stop = GST_DEBUG_FUNCPTR(gst_streambox_src_unlock_stop);
    basesrc_class->get_caps = GST_DEBUG_FUNCPTR(gst_streambox_src_get_caps);
    basesrc_class->fixate = GST_DEBUG_FUNCPTR(gst_streambox_src_fixate);
    basesrc_class->decide_allocation = GST_DEBUG_FUNCPTR(gst_streambox_src_decide_allocation);

    pushsrc_class->create = GST_DEBUG_FUNCPTR(gst_streambox_src_create);
}

/* ---------- Instance init ---------- */

static void
gst_streambox_src_init(GstStreamboxSrc *self)
{
    self->source_mode = DEFAULT_SOURCE_MODE;
    self->device = NULL;  /* auto-detect */
    self->num_buffers = DEFAULT_NUM_BUFFERS;
    self->patha_out_pool_size = DEFAULT_PATHA_POOL_SIZE;
    self->output_fmt = DEFAULT_OUTPUT_FMT;
    self->target_width = 0;
    self->target_height = 0;
    self->target_fps = 0.0f;
    self->color_mode = 0;
    self->vdin1_input = VDIN1_INPUT_VPP_POST_BLEND;

    self->sig_state = GST_STREAMBOX_STATE_IDLE;
    g_mutex_init(&self->state_lock);

    self->streaming = FALSE;
    self->width = 0;
    self->height = 0;
    self->fps_n = 60;
    self->fps_d = 1;
    self->caps_set = FALSE;
    self->frame_count = 0;
    self->colorimetry[0] = '\0';

    /* Path A */
    self->cap_ctx = NULL;
    self->heap_fd = -1;
    self->out_buf_size = 0;
    self->patha_out_count = 0;
    g_mutex_init(&self->patha_out_lock);
    for (guint i = 0; i < PATHA_OUT_POOL_SIZE_MAX; i++) {
        self->patha_out_fds[i] = -1;
        self->patha_out_free[i] = FALSE;
    }

    /* Path B */
    self->signal_monitor_fd = -1;
    self->vdin1_fd = -1;
    self->heap_cma_fd = -1;
    self->vdin1_n_bufs = 0;
    self->vdin1_pixfmt = 0;
    self->vdin1_num_planes = 0;
    self->vdin1_sizeimage = 0;
    self->vdin1_prev_width = 0;
    self->vdin1_prev_height = 0;
    self->vdin1_prev_pixfmt = 0;
    self->vdin1_fmt_poll_counter = 0;
    for (guint i = 0; i < STREAMBOX_VDIN1_MAX_BUFFERS; i++) {
        self->vdin1_bufs[i].dma_fd = -1;
        self->vdin1_bufs[i].size = 0;
        self->vdin1_bufs[i].queued = FALSE;
    }

    /* Path B Vulkan */
    self->vdin1_10bit = FALSE;
    self->vdin1_amly_sizeimage = 0;
    self->vk_instance = VK_NULL_HANDLE;
    self->vk_physical_device = VK_NULL_HANDLE;
    self->vk_device = VK_NULL_HANDLE;
    self->vk_compute_queue = VK_NULL_HANDLE;
    self->vk_queue_family = 0;
    self->vk_command_pool = VK_NULL_HANDLE;
    self->vk_cmd[0] = VK_NULL_HANDLE;
    self->vk_cmd[1] = VK_NULL_HANDLE;
    self->vk_fences[0] = VK_NULL_HANDLE;
    self->vk_fences[1] = VK_NULL_HANDLE;
    self->vk_slot = 0;
    self->vk_descriptor_pool = VK_NULL_HANDLE;
    self->vk_descriptor_set_layout = VK_NULL_HANDLE;
    self->vk_descriptor_sets[0] = VK_NULL_HANDLE;
    self->vk_descriptor_sets[1] = VK_NULL_HANDLE;
    self->vk_pipeline_layout = VK_NULL_HANDLE;
    self->vk_pipeline_p010 = VK_NULL_HANDLE;
    self->vk_shader_p010 = VK_NULL_HANDLE;
    self->vk_initialized = FALSE;
    self->vk_frame_count = 0;
    self->vk_input_cache_count = 0;
    self->vk_output_cache.valid = 0;
    self->vk_output_cache.fd = -1;
    self->vk_pending_in_fd = -1;
    self->vk_has_pending = FALSE;
    self->vk_async_pending = FALSE;
    self->vk_async_out_pool_idx = -1;
    self->vk_async_vdin1_idx = 0;
    self->vk_async_in_fd = -1;
    self->vk_async_pts = 0;
    self->vk_async_duration = 0;

    /* P010 output pool */
    self->p010_out_count = 0;
    self->p010_out_size = 0;
    g_mutex_init(&self->p010_out_lock);
    for (guint i = 0; i < P010_OUT_POOL_SIZE; i++) {
        self->p010_out_fds[i] = -1;
        self->p010_out_free[i] = FALSE;
        self->vk_output_pool_cache[i].valid = 0;
        self->vk_output_pool_cache[i].fd = -1;
    }

    /* Flush pipe */
    self->flushing = FALSE;
    self->flush_pipefd[0] = -1;
    self->flush_pipefd[1] = -1;

    /* We are a live source */
    gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
    gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
    gst_base_src_set_do_timestamp(GST_BASE_SRC(self), TRUE);
}

/* ---------- Properties ---------- */

static void
gst_streambox_src_set_property(GObject *object, guint prop_id,
                                const GValue *value, GParamSpec *pspec)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(object);

    switch (prop_id) {
    case PROP_SOURCE:
        self->source_mode = g_value_get_enum(value);
        break;
    case PROP_DEVICE:
        g_free(self->device);
        self->device = g_value_dup_string(value);
        break;
    case PROP_NUM_BUFFERS:
        self->num_buffers = g_value_get_uint(value);
        break;
    case PROP_PATHA_POOL_SIZE:
        self->patha_out_pool_size = g_value_get_uint(value);
        break;
    case PROP_OUTPUT_FORMAT:
        self->output_fmt = g_value_get_enum(value);
        break;
    case PROP_TARGET_WIDTH:
        self->target_width = g_value_get_uint(value);
        break;
    case PROP_TARGET_HEIGHT:
        self->target_height = g_value_get_uint(value);
        break;
    case PROP_TARGET_FPS:
        self->target_fps = g_value_get_float(value);
        break;
    case PROP_COLOR_MODE:
        self->color_mode = g_value_get_uint(value);
        break;
    case PROP_VDIN1_INPUT:
        self->vdin1_input = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_streambox_src_get_property(GObject *object, guint prop_id,
                                GValue *value, GParamSpec *pspec)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(object);

    switch (prop_id) {
    case PROP_SOURCE:
        g_value_set_enum(value, self->source_mode);
        break;
    case PROP_DEVICE:
        if (self->device) {
            g_value_set_string(value, self->device);
        } else {
            g_value_set_string(value,
                self->source_mode == GST_STREAMBOX_SOURCE_VFMCAP
                    ? DEFAULT_DEVICE_VFMCAP : DEFAULT_DEVICE_VDIN1);
        }
        break;
    case PROP_NUM_BUFFERS:
        g_value_set_uint(value, self->num_buffers);
        break;
    case PROP_PATHA_POOL_SIZE:
        g_value_set_uint(value, self->patha_out_pool_size);
        break;
    case PROP_OUTPUT_FORMAT:
        g_value_set_enum(value, self->output_fmt);
        break;
    case PROP_TARGET_WIDTH:
        g_value_set_uint(value, self->target_width);
        break;
    case PROP_TARGET_HEIGHT:
        g_value_set_uint(value, self->target_height);
        break;
    case PROP_TARGET_FPS:
        g_value_set_float(value, self->target_fps);
        break;
    case PROP_COLOR_MODE:
        g_value_set_uint(value, self->color_mode);
        break;
    case PROP_VDIN1_INPUT:
        g_value_set_uint(value, self->vdin1_input);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_streambox_src_finalize(GObject *object)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(object);
    g_free(self->device);
    g_mutex_clear(&self->state_lock);
    g_mutex_clear(&self->patha_out_lock);

    if (self->flush_pipefd[0] >= 0)
        close(self->flush_pipefd[0]);
    if (self->flush_pipefd[1] >= 0)
        close(self->flush_pipefd[1]);

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* ---------- Resolve device path ---------- */

static const gchar *
resolve_device(GstStreamboxSrc *self)
{
    if (self->device && self->device[0])
        return self->device;
    return self->source_mode == GST_STREAMBOX_SOURCE_VFMCAP
        ? DEFAULT_DEVICE_VFMCAP : DEFAULT_DEVICE_VDIN1;
}

/* ---------- Start / Stop ---------- */

static gboolean
gst_streambox_src_start(GstBaseSrc *basesrc)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(basesrc);

    GST_INFO_OBJECT(self, "Starting: source=%s device=%s buffers=%u",
                     self->source_mode == GST_STREAMBOX_SOURCE_VFMCAP
                         ? "vfmcap" : "vdin1",
                     resolve_device(self), self->num_buffers);

    /* Create flush pipe */
    if (self->flush_pipefd[0] < 0) {
        if (pipe2(self->flush_pipefd, O_CLOEXEC | O_NONBLOCK) < 0) {
            GST_ERROR_OBJECT(self, "pipe2 failed: %s", strerror(errno));
            return FALSE;
        }
    }
    self->flushing = FALSE;

    self->frame_count = 0;
    self->caps_set = FALSE;
    self->colorimetry[0] = '\0';

    if (self->source_mode == GST_STREAMBOX_SOURCE_VFMCAP)
        return start_path_a(self);
    else
        return start_path_b(self);
}

static gboolean
gst_streambox_src_stop(GstBaseSrc *basesrc)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(basesrc);

    GST_INFO_OBJECT(self, "Stopping: %lu frames captured",
                     (unsigned long)self->frame_count);

    if (self->source_mode == GST_STREAMBOX_SOURCE_VFMCAP)
        stop_path_a(self);
    else
        stop_path_b(self);

    self->streaming = FALSE;
    self->caps_set = FALSE;
    self->sig_state = GST_STREAMBOX_STATE_IDLE;

    return TRUE;
}

/* ---------- Unlock (for flushing / state changes) ---------- */

static gboolean
gst_streambox_src_unlock(GstBaseSrc *basesrc)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(basesrc);
    GST_DEBUG_OBJECT(self, "unlock");

    self->flushing = TRUE;

    /* Write a byte to the flush pipe to wake up poll() */
    if (self->flush_pipefd[1] >= 0) {
        char c = 'x';
        if (write(self->flush_pipefd[1], &c, 1) < 0)
            GST_WARNING_OBJECT(self, "flush pipe write: %s", strerror(errno));
    }

    return TRUE;
}

static gboolean
gst_streambox_src_unlock_stop(GstBaseSrc *basesrc)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(basesrc);
    GST_DEBUG_OBJECT(self, "unlock_stop");

    self->flushing = FALSE;

    /* Drain the flush pipe */
    if (self->flush_pipefd[0] >= 0) {
        char buf[16];
        while (read(self->flush_pipefd[0], buf, sizeof(buf)) > 0)
            ;
    }

    return TRUE;
}

/* ---------- Caps negotiation ---------- */

static const gchar *
get_caps_format_string(GstStreamboxSrc *self)
{
    if (self->source_mode == GST_STREAMBOX_SOURCE_VDIN1) {
        return self->vdin1_10bit ? "P010_10LE" : "NV21";
    }
    /* Path A */
    return (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
           ? "P010_10LE" : "NV12";
}

static GstCaps *
gst_streambox_src_get_caps(GstBaseSrc *basesrc, GstCaps *filter)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(basesrc);
    GstCaps *caps;

    if (self->streaming && self->width > 0 && self->height > 0) {
        caps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, get_caps_format_string(self),
            "width", G_TYPE_INT, (gint)self->width,
            "height", G_TYPE_INT, (gint)self->height,
            "framerate", GST_TYPE_FRACTION, (gint)self->fps_n, (gint)self->fps_d,
            NULL);

        /* Add colorimetry if detected */
        if (self->colorimetry[0] != '\0') {
            GstVideoColorimetry cinfo;
            if (gst_video_colorimetry_from_string(&cinfo, self->colorimetry)) {
                gchar *cstr = gst_video_colorimetry_to_string(&cinfo);
                if (cstr) {
                    gst_caps_set_simple(caps, "colorimetry", G_TYPE_STRING, cstr, NULL);
                    g_free(cstr);
                }
            }
        }
    } else {
        caps = gst_pad_get_pad_template_caps(GST_BASE_SRC_PAD(basesrc));
    }

    if (filter) {
        GstCaps *intersection = gst_caps_intersect_full(caps, filter,
                                                        GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(caps);
        caps = intersection;
    }

    GST_DEBUG_OBJECT(self, "get_caps: %" GST_PTR_FORMAT, caps);
    return caps;
}

static GstCaps *
gst_streambox_src_fixate(GstBaseSrc *basesrc, GstCaps *caps)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(basesrc);
    GstStructure *s;

    caps = gst_caps_make_writable(caps);
    s = gst_caps_get_structure(caps, 0);

    if (self->width > 0 && self->height > 0) {
        gst_structure_fixate_field_nearest_int(s, "width", self->width);
        gst_structure_fixate_field_nearest_int(s, "height", self->height);
    }

    if (self->fps_n > 0) {
        gst_structure_fixate_field_nearest_fraction(s, "framerate",
                                                     self->fps_n, self->fps_d);
    }

    caps = GST_BASE_SRC_CLASS(parent_class)->fixate(basesrc, caps);
    GST_DEBUG_OBJECT(self, "fixate: %" GST_PTR_FORMAT, caps);
    return caps;
}

/* ---------- Push caps helper ---------- */

static void
push_current_caps(GstStreamboxSrc *self)
{
    const gchar *fmt_str = get_caps_format_string(self);

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, fmt_str,
        "width", G_TYPE_INT, (gint)self->width,
        "height", G_TYPE_INT, (gint)self->height,
        "framerate", GST_TYPE_FRACTION, (gint)self->fps_n, (gint)self->fps_d,
        NULL);

    /* Add colorimetry if detected from HDMI RX signal */
    if (self->colorimetry[0] != '\0') {
        GstVideoColorimetry cinfo;
        if (gst_video_colorimetry_from_string(&cinfo, self->colorimetry)) {
            gchar *cstr = gst_video_colorimetry_to_string(&cinfo);
            if (cstr) {
                gst_caps_set_simple(caps, "colorimetry", G_TYPE_STRING, cstr, NULL);
                g_free(cstr);
            }
        } else {
            GST_WARNING_OBJECT(self, "Failed to parse colorimetry '%s'",
                               self->colorimetry);
        }
    }

    GST_INFO_OBJECT(self, "Setting caps: %" GST_PTR_FORMAT, caps);
    gst_base_src_set_caps(GST_BASE_SRC(self), caps);
    gst_caps_unref(caps);
    self->caps_set = TRUE;
}

/* ====================================================================
 * PATH A: vfm_cap via libvfmcap (raw, low latency)
 * ==================================================================== */

static gboolean
start_path_a(GstStreamboxSrc *self)
{
    const gchar *dev = resolve_device(self);

    vfmcap_config_t cfg = {0};
    cfg.output_format = (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
                         ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
    cfg.target_width = self->target_width;
    cfg.target_height = self->target_height;
    cfg.target_fps = self->target_fps;
    cfg.color_mode = (vfmcap_color_mode_t)self->color_mode;

    self->cap_ctx = vfmcap_open(dev, &cfg);
    if (!self->cap_ctx) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("Failed to open %s", dev),
                          ("%s", vfmcap_last_error(NULL)));
        return FALSE;
    }

    int ret = vfmcap_start(self->cap_ctx, self->num_buffers);
    if (ret != VFMCAP_OK) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("Failed to start streaming"),
                          ("%s", vfmcap_last_error(self->cap_ctx)));
        vfmcap_close(self->cap_ctx);
        self->cap_ctx = NULL;
        return FALSE;
    }

    /*
     * Acquire a test frame to get the actual signal resolution.
     * V4L2 G_FMT returns defaults until the first vframe arrives from vdin0.
     */
    vfmcap_frame_t test_frame;
    ret = vfmcap_acquire_frame(self->cap_ctx, &test_frame, 3000);
    if (ret == VFMCAP_OK) {
        self->width = test_frame.width;
        self->height = test_frame.height;
        /* Read actual framerate from HDMI RX, fall back to 60fps */
        if (!hdmirx_get_source_framerate(self, &self->fps_n, &self->fps_d)) {
            self->fps_n = 60;
            self->fps_d = 1;
        }
        vfmcap_release_frame(self->cap_ctx, &test_frame);
    } else {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                          ("No signal or cannot acquire test frame"),
                          ("%s", vfmcap_last_error(self->cap_ctx)));
        vfmcap_close(self->cap_ctx);
        self->cap_ctx = NULL;
        return FALSE;
    }

    /* Detect colorimetry from HDMI RX signal for caps */
    hdmirx_detect_colorimetry(self);

    self->streaming = TRUE;
    g_mutex_lock(&self->state_lock);
    self->sig_state = GST_STREAMBOX_STATE_STREAMING;
    g_mutex_unlock(&self->state_lock);

    GST_INFO_OBJECT(self, "Path A started: %ux%u @ %u/%u, output=%s, target=%ux%u fps=%.1f color=%u",
                     self->width, self->height, self->fps_n, self->fps_d,
                     self->output_fmt == GST_STREAMBOX_OUTPUT_P010 ? "P010" : "NV12",
                     self->target_width, self->target_height,
                     self->target_fps, self->color_mode);
    return TRUE;
}

static void
stop_path_a(GstStreamboxSrc *self)
{
    if (self->cap_ctx) {
        /* vfmcap_close() handles the full teardown in safe order:
         *   1. Vulkan cleanup (releases GPU-imported DMA-buf references)
         *   2. STREAMOFF + REQBUFS(0) (frees kernel CMA buffers)
         *   3. close(fd)
         * Do NOT call vfmcap_stop() separately — that would free kernel
         * CMA buffers before Vulkan has released its imported references,
         * causing a use-after-free kernel oops on shutdown. */
        vfmcap_close(self->cap_ctx);
        self->cap_ctx = NULL;
    }

    if (self->heap_fd >= 0) {
        close(self->heap_fd);
        self->heap_fd = -1;
    }
}

typedef struct {
    GstStreamboxSrc *self;
    vfmcap_frame_t   frame;
} PathAFrameContext;

static void
patha_frame_release(gpointer data)
{
    PathAFrameContext *ctx = (PathAFrameContext *)data;
    if (ctx->self && ctx->self->cap_ctx) {
        vfmcap_release_frame(ctx->self->cap_ctx, &ctx->frame);
    }
    if (ctx->self)
        gst_object_unref(ctx->self);
    g_free(ctx);
}

static GstFlowReturn
create_path_a(GstStreamboxSrc *self, GstBuffer **buf)
{
    if (!self->cap_ctx)
        return GST_FLOW_ERROR;

    /* Push caps on first frame */
    if (!self->caps_set)
        push_current_caps(self);

    /* Acquire frame (conversion integrated if configured) */
    vfmcap_frame_t frame;
    int ret = vfmcap_acquire_frame(self->cap_ctx, &frame, 1000);

    if (self->flushing)
        return GST_FLOW_FLUSHING;

    if (ret == VFMCAP_ERR_TIMEOUT) {
        GST_WARNING_OBJECT(self, "Frame acquire timeout (frame %lu)",
                           (unsigned long)self->frame_count);
        return GST_FLOW_ERROR;
    }

    if (ret == VFMCAP_ERR_NOSIG) {
        GST_WARNING_OBJECT(self, "No signal — posting hdmi-signal-change and exiting");
        g_mutex_lock(&self->state_lock);
        self->sig_state = GST_STREAMBOX_STATE_WAITING;
        g_mutex_unlock(&self->state_lock);
        post_signal_change_message(self, "signal-lost");
        return GST_FLOW_EOS;
    }

    if (ret == VFMCAP_RECONFIGURED) {
        GST_INFO_OBJECT(self, "Signal reconfigured — exiting for restart");
        vfmcap_release_frame(self->cap_ctx, &frame);
        post_signal_change_message(self, "signal-changed");
        return GST_FLOW_EOS;
    }

    if (ret != VFMCAP_OK) {
        GST_ERROR_OBJECT(self, "acquire_frame failed: %s",
                         vfmcap_last_error(self->cap_ctx));
        return GST_FLOW_ERROR;
    }

    /* Check for resolution change */
    if (frame.width != self->width || frame.height != self->height) {
        GST_INFO_OBJECT(self, "Resolution changed: %ux%u -> %ux%u — exiting for restart",
                         self->width, self->height, frame.width, frame.height);
        vfmcap_release_frame(self->cap_ctx, &frame);
        post_signal_change_message(self, "signal-changed");
        return GST_FLOW_EOS;
    }

    /* Wrap the DMA-buf fd(s) in GstBuffer.
     * For integrated conversion, frame.dmabuf_fd is the output buffer
     * owned by libvfmcap. We dup it because gst_dmabuf_allocator_alloc
     * takes ownership of the fd. */
    int dup_fd = dup(frame.dmabuf_fd);
    if (dup_fd < 0) {
        GST_ERROR_OBJECT(self, "dup(frame.dmabuf_fd=%d) failed: %s",
                         frame.dmabuf_fd, strerror(errno));
        vfmcap_release_frame(self->cap_ctx, &frame);
        return GST_FLOW_ERROR;
    }

    GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new();
    GstMemory *mem = gst_dmabuf_allocator_alloc(dmabuf_alloc, dup_fd, frame.size);
    gst_object_unref(dmabuf_alloc);

    if (!mem) {
        GST_ERROR_OBJECT(self, "Failed to wrap DMA-buf fd as GstMemory");
        close(dup_fd);
        vfmcap_release_frame(self->cap_ctx, &frame);
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_buffer_new();
    gst_buffer_append_memory(buffer, mem);

    /* Add second plane memory if present (NV12/P010 linear) */
    if (frame.dmabuf_fd2 >= 0) {
        int dup_fd2 = dup(frame.dmabuf_fd2);
        if (dup_fd2 < 0) {
            GST_ERROR_OBJECT(self, "dup(frame.dmabuf_fd2=%d) failed: %s",
                             frame.dmabuf_fd2, strerror(errno));
            gst_buffer_unref(buffer);
            vfmcap_release_frame(self->cap_ctx, &frame);
            return GST_FLOW_ERROR;
        }
        gsize plane2_size;
        if (frame.pixelformat == V4L2_PIX_FMT_NV12)
            plane2_size = (gsize)frame.width * frame.height / 2;
        else if (frame.pixelformat == v4l2_fourcc('P', '0', '1', '0'))
            plane2_size = (gsize)frame.width * frame.height;
        else
            plane2_size = frame.size / 2;
        dmabuf_alloc = gst_dmabuf_allocator_new();
        GstMemory *mem2 = gst_dmabuf_allocator_alloc(dmabuf_alloc, dup_fd2, plane2_size);
        gst_object_unref(dmabuf_alloc);
        if (!mem2) {
            GST_ERROR_OBJECT(self, "Failed to wrap DMA-buf fd2 as GstMemory");
            close(dup_fd2);
            gst_buffer_unref(buffer);
            vfmcap_release_frame(self->cap_ctx, &frame);
            return GST_FLOW_ERROR;
        }
        gst_buffer_append_memory(buffer, mem2);
    }

    GST_BUFFER_PTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(GST_SECOND,
                                                             self->fps_d,
                                                             self->fps_n);

    /* Add video meta from frame format */
    GstVideoFormat gst_fmt;
    guint n_planes = 1;
    gsize offsets[GST_VIDEO_MAX_PLANES] = { 0, };
    gint strides[GST_VIDEO_MAX_PLANES] = { 0, };

    if (frame.pixelformat == v4l2_fourcc('P', '0', '1', '0')) {
        gst_fmt = GST_VIDEO_FORMAT_P010_10LE;
        n_planes = 2;
        strides[0] = frame.width * 2;
        strides[1] = frame.width * 2;
        offsets[0] = 0;
        offsets[1] = (frame.dmabuf_fd2 < 0) ? (gsize)frame.width * frame.height * 2 : 0;
    } else if (frame.pixelformat == V4L2_PIX_FMT_NV12) {
        gst_fmt = GST_VIDEO_FORMAT_NV12;
        n_planes = 2;
        strides[0] = frame.width;
        strides[1] = frame.width;
        offsets[0] = 0;
        offsets[1] = (gsize)frame.width * frame.height;
    } else if (frame.pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y')) {
        /* Raw AMLY passthrough — treat as single-plane opaque */
        gst_fmt = GST_VIDEO_FORMAT_UNKNOWN;
        n_planes = 1;
        strides[0] = frame.bytesperline;
        offsets[0] = 0;
    } else {
        /* Fallback for other formats */
        gst_fmt = GST_VIDEO_FORMAT_UNKNOWN;
        n_planes = 1;
        strides[0] = frame.bytesperline;
        offsets[0] = 0;
    }

    if (gst_fmt != GST_VIDEO_FORMAT_UNKNOWN) {
        gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                        gst_fmt, frame.width, frame.height,
                                        n_planes, offsets, strides);
    }

    /* Attach release callback so vfmcap_release_frame is called when
     * the downstream element unrefs the GstBuffer. */
    PathAFrameContext *pctx = g_new(PathAFrameContext, 1);
    pctx->self = GST_STREAMBOX_SRC(gst_object_ref(self));
    pctx->frame = frame;
    gst_mini_object_set_qdata(GST_MINI_OBJECT(buffer),
                               g_quark_from_static_string("patha-frame-ctx"),
                               pctx, patha_frame_release);

    self->frame_count++;

    if (self->frame_count == 1 || self->frame_count % 300 == 0) {
        GST_INFO_OBJECT(self, "Path A frame %lu: %ux%u pixfmt=%.4s",
                         (unsigned long)self->frame_count,
                         frame.width, frame.height,
                         (char *)&frame.pixelformat);
    }

    *buf = buffer;
    return GST_FLOW_OK;
}

/* ====================================================================
 * PATH B: vdin1 direct V4L2 (color-processed)
 * ==================================================================== */

static gboolean
start_path_b(GstStreamboxSrc *self)
{
    const gchar *dev = resolve_device(self);

    self->vdin1_fd = open(dev, O_RDWR | O_NONBLOCK);
    if (self->vdin1_fd < 0) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("Failed to open %s: %s", dev, strerror(errno)),
                          (NULL));
        return FALSE;
    }

    /* Query capabilities */
    struct v4l2_capability cap;
    if (xioctl(self->vdin1_fd, VIDIOC_QUERYCAP, &cap) < 0) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("VIDIOC_QUERYCAP failed: %s", strerror(errno)),
                          (NULL));
        close(self->vdin1_fd);
        self->vdin1_fd = -1;
        return FALSE;
    }

    GST_INFO_OBJECT(self, "vdin1: driver=%s card=%s bus=%s",
                     cap.driver, cap.card, cap.bus_info);

    /*
     * Step 1: Select V4L2 input for vdin1 loopback.
     * Default is VPP post-blend (index 6), which captures the full VPP
     * output after HDR->SDR tone mapping, gamut conversion, and all
     * color processing. Can be changed via the vdin1-input property.
     */
    if (!vdin1_set_input(self, self->vdin1_input)) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                          ("Cannot select vdin1 input %u", self->vdin1_input),
                          (NULL));
        close(self->vdin1_fd);
        self->vdin1_fd = -1;
        return FALSE;
    }

    /*
     * Step 2: Determine the source resolution.
     * Query HDMI RX for the actual input resolution so we can request
     * full-resolution output from vdin1 (no scaling).
     * If HDMI RX query fails, fall back to vdin1's G_FMT default.
     */
    guint src_w = 0, src_h = 0;
    if (!hdmirx_get_source_resolution(self, &src_w, &src_h)) {
        /* Fallback: read whatever vdin1 currently reports */
        guint np = 0;
        guint32 pf = 0;
        if (vdin1_get_format(self, &src_w, &src_h, &pf, &np)) {
            GST_INFO_OBJECT(self, "Using vdin1 G_FMT fallback: %ux%u", src_w, src_h);
        }
    }

    /* Default to 1920x1080 if nothing found */
    if (src_w == 0 || src_h == 0) {
        src_w = 1920;
        src_h = 1080;
        GST_WARNING_OBJECT(self, "No source resolution found, defaulting to %ux%u",
                           src_w, src_h);
    }

    /*
     * Step 3: Set output format (NV21 at full source resolution).
     * vdin1 can scale down but not up. By requesting the full source
     * resolution, we avoid any scaling and get the best quality.
     */
    if (!vdin1_set_format(self, src_w, src_h)) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                          ("Cannot set vdin1 format to %ux%u NV21", src_w, src_h),
                          (NULL));
        close(self->vdin1_fd);
        self->vdin1_fd = -1;
        return FALSE;
    }

    /* Save for format-change detection */
    self->vdin1_prev_width = self->width;
    self->vdin1_prev_height = self->height;
    self->vdin1_prev_pixfmt = self->vdin1_pixfmt;
    self->vdin1_fmt_poll_counter = 0;

    /* Detect colorimetry from HDMI RX signal for caps.
     * Must be after vdin1_set_format() which sets vdin1_10bit. */
    hdmirx_detect_colorimetry(self);

    /* Request buffers (DMABUF mode) */
    guint req_count = self->num_buffers;
    if (req_count > STREAMBOX_VDIN1_MAX_BUFFERS)
        req_count = STREAMBOX_VDIN1_MAX_BUFFERS;
    if (!vdin1_reqbufs(self, req_count))
        goto fail;

    /* Allocate DMA-bufs from heap */
    if (!vdin1_alloc_dmabufs(self))
        goto fail;

    /* Queue all buffers (passes DMA-buf fds to vdin1) */
    if (!vdin1_qbuf_all(self))
        goto fail;

    /* Start streaming */
    if (!vdin1_streamon(self))
        goto fail;

    /* Initialize Vulkan for 10-bit AMLY->P010 conversion */
    if (self->vdin1_10bit) {
        if (!vdin1_vk_init(self)) {
            GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                              ("Failed to initialize Vulkan for P010 conversion"),
                              (NULL));
            vdin1_streamoff(self);
            goto fail;
        }

        /* Pre-allocate P010 output buffer pool and import into Vulkan */
        guint32 p010_size = self->width * self->height * 3;  /* Y + UV */
        if (!p010_pool_init(self, p010_size)) {
            GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                              ("Failed to allocate P010 output buffer pool"),
                              (NULL));
            vdin1_streamoff(self);
            goto fail;
        }
    }

    /*
     * Open vfm_cap (/dev/video_cap) as a signal monitor.
     * vdin1's V4L2 driver does NOT support VIDIOC_SUBSCRIBE_EVENT, but
     * vfm_cap does.  vfm_cap monitors the vdin0 tvin state machine and
     * translates HDMI signal changes into V4L2_EVENT_SOURCE_CHANGE events.
     * We poll this fd alongside the vdin1 capture fd in create_path_b()
     * to get event-driven signal change detection with zero polling overhead.
     *
     * Note: vfm_cap thus serves a dual role:
     *   Path A: frame capture (MMAP/DMA-buf)
     *   Path B: signal monitor (V4L2 events only, no frame capture)
     */
    self->signal_monitor_fd = open(DEFAULT_DEVICE_VFMCAP, O_RDWR | O_NONBLOCK);
    if (self->signal_monitor_fd < 0) {
        GST_WARNING_OBJECT(self,
            "Cannot open %s for signal monitoring: %s (signal change detection disabled)",
            DEFAULT_DEVICE_VFMCAP, strerror(errno));
        /* Non-fatal: we fall back to G_FMT polling + poll timeout detection */
    } else {
        struct v4l2_event_subscription sub;
        memset(&sub, 0, sizeof(sub));
        sub.type = V4L2_EVENT_SOURCE_CHANGE;
        if (ioctl(self->signal_monitor_fd, VIDIOC_SUBSCRIBE_EVENT, &sub) < 0) {
            GST_WARNING_OBJECT(self,
                "VIDIOC_SUBSCRIBE_EVENT(SOURCE_CHANGE) failed: %s (signal change detection disabled)",
                strerror(errno));
            close(self->signal_monitor_fd);
            self->signal_monitor_fd = -1;
        } else {
            GST_INFO_OBJECT(self,
                "Signal monitor: subscribed to V4L2_EVENT_SOURCE_CHANGE on %s (fd=%d)",
                DEFAULT_DEVICE_VFMCAP, self->signal_monitor_fd);
        }
    }

    self->streaming = TRUE;
    g_mutex_lock(&self->state_lock);
    self->sig_state = GST_STREAMBOX_STATE_STREAMING;
    g_mutex_unlock(&self->state_lock);
    /* Read actual framerate from HDMI RX, fall back to 60fps */
    if (!hdmirx_get_source_framerate(self, &self->fps_n, &self->fps_d)) {
        self->fps_n = 60;
        self->fps_d = 1;
    }

    GST_INFO_OBJECT(self, "Path B started: %ux%u %s (source %ux%u, %u bufs)",
                     self->width, self->height,
                     self->vdin1_10bit ? "AMLY->P010" : "NV21",
                     src_w, src_h, self->vdin1_n_bufs);
    return TRUE;

fail:
    if (self->signal_monitor_fd >= 0) {
        close(self->signal_monitor_fd);
        self->signal_monitor_fd = -1;
    }
    p010_pool_cleanup(self);
    vdin1_vk_cleanup(self);
    vdin1_free_dmabufs(self);
    close(self->vdin1_fd);
    self->vdin1_fd = -1;
    return FALSE;
}

static void
stop_path_b(GstStreamboxSrc *self)
{
    /* Drain any pending async GPU work before cleanup */
    if (self->vk_async_pending && self->vk_initialized) {
        guint prev_slot = 1 - self->vk_slot;
        GST_INFO_OBJECT(self, "Draining pending async GPU work on slot %u", prev_slot);
        vdin1_vk_wait_async(self, prev_slot, self->vk_async_in_fd);

        /* Re-QBUF the vdin1 buffer that was held */
        guint idx = self->vk_async_vdin1_idx;
        if (self->vdin1_fd >= 0 && idx < self->vdin1_n_bufs) {
            struct v4l2_buffer qbuf;
            struct v4l2_plane qplanes[1];
            memset(&qbuf, 0, sizeof(qbuf));
            memset(&qplanes, 0, sizeof(qplanes));
            qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            qbuf.memory = V4L2_MEMORY_DMABUF;
            qbuf.index = idx;
            qbuf.m.planes = qplanes;
            qbuf.length = 1;
            qplanes[0].m.fd = self->vdin1_bufs[idx].dma_fd;
            qplanes[0].length = self->vdin1_bufs[idx].size;
            xioctl(self->vdin1_fd, VIDIOC_QBUF, &qbuf);
        }

        /* Return output buffer to pool */
        if (self->vk_async_out_pool_idx >= 0 &&
            (guint)self->vk_async_out_pool_idx < self->p010_out_count) {
            g_mutex_lock(&self->p010_out_lock);
            self->p010_out_free[self->vk_async_out_pool_idx] = TRUE;
            g_mutex_unlock(&self->p010_out_lock);
        }

        self->vk_async_pending = FALSE;
    }

    /* Clean up P010 output pool (must be before Vulkan cleanup since pool
     * entries hold Vulkan objects) */
    p010_pool_cleanup(self);

    /* Clean up Vulkan before releasing DMA-bufs */
    vdin1_vk_cleanup(self);

    if (self->vdin1_fd >= 0) {
        vdin1_streamoff(self);

        /* Release V4L2 buffers */
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 0;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        req.memory = V4L2_MEMORY_DMABUF;
        xioctl(self->vdin1_fd, VIDIOC_REQBUFS, &req);

        /* Free DMA-bufs */
        vdin1_free_dmabufs(self);

        close(self->vdin1_fd);
        self->vdin1_fd = -1;
    }

    if (self->heap_cma_fd >= 0) {
        close(self->heap_cma_fd);
        self->heap_cma_fd = -1;
    }

    if (self->heap_fd >= 0) {
        close(self->heap_fd);
        self->heap_fd = -1;
    }

    /* Close signal monitor */
    if (self->signal_monitor_fd >= 0) {
        close(self->signal_monitor_fd);
        self->signal_monitor_fd = -1;
    }
}

/*
 * Check if vdin1's format has changed by polling G_FMT.
 * Returns TRUE if format changed (caller should reconfigure).
 */
static gboolean
vdin1_check_format_change(GstStreamboxSrc *self)
{
    self->vdin1_fmt_poll_counter++;
    if (self->vdin1_fmt_poll_counter % VDIN1_FMT_POLL_INTERVAL != 0)
        return FALSE;

    guint w = 0, h = 0, np = 0;
    guint32 pf = 0;
    if (!vdin1_get_format(self, &w, &h, &pf, &np))
        return FALSE;

    if (w != self->vdin1_prev_width || h != self->vdin1_prev_height ||
        pf != self->vdin1_prev_pixfmt) {
        GST_INFO_OBJECT(self, "vdin1 format changed: %ux%u 0x%08x -> %ux%u 0x%08x",
                         self->vdin1_prev_width, self->vdin1_prev_height,
                         self->vdin1_prev_pixfmt, w, h, pf);
        return TRUE;
    }

    return FALSE;
}

/*
 * Context passed to the GstBuffer dispose callback so we can re-QBUF
 * the V4L2 buffer when downstream is done with it.
 */
typedef struct {
    GstStreamboxSrc *self;
    guint            buf_index;
} PathBBufContext;

static void
path_b_buf_release(gpointer data)
{
    PathBBufContext *ctx = (PathBBufContext *)data;
    GstStreamboxSrc *self = ctx->self;
    guint idx = ctx->buf_index;

    if (self->vdin1_fd >= 0 && self->streaming && idx < self->vdin1_n_bufs) {
        struct v4l2_buffer qbuf;
        struct v4l2_plane qplanes[1];
        memset(&qbuf, 0, sizeof(qbuf));
        memset(&qplanes, 0, sizeof(qplanes));
        qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        qbuf.memory = V4L2_MEMORY_DMABUF;
        qbuf.index = idx;
        qbuf.m.planes = qplanes;
        qbuf.length = 1;
        qplanes[0].m.fd = self->vdin1_bufs[idx].dma_fd;
        qplanes[0].length = self->vdin1_bufs[idx].size;

        if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qbuf) < 0) {
            GST_WARNING_OBJECT(self, "re-QBUF(%u) failed: %s",
                               idx, strerror(errno));
        } else {
            self->vdin1_bufs[idx].queued = TRUE;
        }
    }

    gst_object_unref(self);
    g_free(ctx);
}

/*
 * P010 output buffer pool: pre-allocated DMA-bufs with Vulkan imports
 * to avoid per-frame allocation and Vulkan object creation overhead.
 */

typedef struct {
    GstStreamboxSrc *self;
    guint            pool_index;   /* index into p010_out_fds[] */
} P010BufContext;

static void
p010_buf_release(gpointer data)
{
    P010BufContext *ctx = (P010BufContext *)data;
    GstStreamboxSrc *self = ctx->self;
    guint idx = ctx->pool_index;

    if (idx < self->p010_out_count) {
        g_mutex_lock(&self->p010_out_lock);
        self->p010_out_free[idx] = TRUE;
        g_mutex_unlock(&self->p010_out_lock);
    }

    gst_object_unref(self);
    g_free(ctx);
}

/* Allocate P010 output DMA-buf pool and pre-import into Vulkan */
static gboolean
p010_pool_init(GstStreamboxSrc *self, guint32 buf_size)
{
    self->p010_out_size = buf_size;
    self->p010_out_count = 0;

    for (guint i = 0; i < P010_OUT_POOL_SIZE; i++) {
        int fd = alloc_cma_dmabuf(self, buf_size);
        if (fd < 0) {
            GST_ERROR_OBJECT(self, "P010 pool: failed to allocate buf %u/%u",
                             i, P010_OUT_POOL_SIZE);
            return FALSE;
        }

        self->p010_out_fds[i] = fd;
        self->p010_out_free[i] = TRUE;
        self->p010_out_count = i + 1;

        /* Pre-import into Vulkan */
        int fd_dup = dup(fd);
        if (fd_dup < 0) {
            GST_ERROR_OBJECT(self, "P010 pool: dup(fd=%d) failed: %s",
                             fd, strerror(errno));
            return FALSE;
        }

        VkBuffer buffer;
        VkDeviceMemory memory;
        if (!vdin1_vk_import_dmabuf(self, fd_dup, buf_size, &buffer, &memory)) {
            GST_ERROR_OBJECT(self, "P010 pool: Vulkan import failed for buf %u", i);
            return FALSE;
        }

        self->vk_output_pool_cache[i].fd = fd;
        self->vk_output_pool_cache[i].fd_dup = fd_dup;
        self->vk_output_pool_cache[i].buffer = buffer;
        self->vk_output_pool_cache[i].memory = memory;
        self->vk_output_pool_cache[i].size = buf_size;
        self->vk_output_pool_cache[i].valid = 1;
    }

    GST_INFO_OBJECT(self, "P010 output pool: %u x %u bytes allocated + Vulkan imported",
                     P010_OUT_POOL_SIZE, buf_size);
    return TRUE;
}

static void
p010_pool_cleanup(GstStreamboxSrc *self)
{
    for (guint i = 0; i < self->p010_out_count; i++) {
        /* Destroy Vulkan objects */
        if (self->vk_output_pool_cache[i].valid) {
            vdin1_vk_cache_entry_destroy(self, &self->vk_output_pool_cache[i]);
        }
        /* Close DMA-buf fd */
        if (self->p010_out_fds[i] >= 0) {
            close(self->p010_out_fds[i]);
            self->p010_out_fds[i] = -1;
        }
        self->p010_out_free[i] = FALSE;
    }
    self->p010_out_count = 0;
    self->p010_out_size = 0;
}

/* Get a free P010 output buffer from the pool. Returns index or -1. */
static gint
p010_pool_acquire(GstStreamboxSrc *self)
{
    g_mutex_lock(&self->p010_out_lock);
    for (guint i = 0; i < self->p010_out_count; i++) {
        if (self->p010_out_free[i]) {
            self->p010_out_free[i] = FALSE;
            g_mutex_unlock(&self->p010_out_lock);
            return (gint)i;
        }
    }
    g_mutex_unlock(&self->p010_out_lock);
    return -1;
}

/*
 * Drain all pending V4L2 events from the signal monitor fd (vfm_cap).
 * Returns the most significant action needed:
 *   SIGNAL_EVENT_CHANGED — source parameters changed, need reconfigure
 *   SIGNAL_EVENT_LOST    — signal lost entirely
 *   SIGNAL_EVENT_NONE    — spurious / no actionable event
 *
 * Drains ALL pending events in a loop because the kernel queues up to 32
 * events, and we need to consume all of them to avoid stale events on the
 * next poll() cycle.
 */
static SignalEventAction
handle_signal_event(GstStreamboxSrc *self)
{
    SignalEventAction action = SIGNAL_EVENT_NONE;
    struct v4l2_event ev;
    int count = 0;

    while (1) {
        memset(&ev, 0, sizeof(ev));
        if (ioctl(self->signal_monitor_fd, VIDIOC_DQEVENT, &ev) < 0) {
            if (errno == ENOENT || errno == EAGAIN)
                break;  /* No more events */
            GST_WARNING_OBJECT(self, "VIDIOC_DQEVENT failed: %s", strerror(errno));
            break;
        }
        count++;

        if (ev.type == V4L2_EVENT_SOURCE_CHANGE) {
            guint32 changes = ev.u.src_change.changes;
            GST_INFO_OBJECT(self,
                "V4L2_EVENT_SOURCE_CHANGE: changes=0x%08x (event %d)",
                changes, count);

            if (changes & V4L2_EVENT_SRC_CH_RESOLUTION) {
                /* Resolution or signal parameters changed */
                action = SIGNAL_EVENT_CHANGED;
            } else if (changes == 0) {
                /*
                 * changes==0 from vfm_cap typically means signal lost.
                 * The vfm_cap driver fires SOURCE_CHANGE with changes=0
                 * when vdin0 transitions out of stable state.
                 */
                if (action != SIGNAL_EVENT_CHANGED)
                    action = SIGNAL_EVENT_LOST;
            }
        } else {
            GST_DEBUG_OBJECT(self, "Ignoring V4L2 event type %u", ev.type);
        }
    }

    if (count > 0) {
        GST_INFO_OBJECT(self,
            "Signal monitor: drained %d event(s), action=%d", count, action);
    }
    return action;
}

/*
 * Read current HDMI RX signal info and post a GST_MESSAGE_ELEMENT on the bus
 * with a GstStructure containing the signal parameters.  The GStreamer Manager
 * watches for messages named "hdmi-signal-change" and uses the fields to
 * decide whether and how to restart the pipeline.
 *
 * The `reason` string describes why we're exiting ("signal-lost",
 * "signal-changed", "format-changed", "timeout").
 *
 * Fields posted:
 *   reason        (string)  — why we're exiting
 *   width         (uint)    — Hactive from HDMI RX, 0 if no signal
 *   height        (uint)    — Vactive from HDMI RX, 0 if no signal
 *   frame-rate    (uint)    — raw value from HDMI RX (e.g. 5992 = 59.92 fps)
 *   color-space   (string)  — e.g. "0-RGB", "1-YUV422", etc.
 *   color-depth   (uint)    — 8, 10, or 12
 *   hdr-eotf      (string)  — e.g. "SMPTE_ST_2048", "SDR", "HLG"
 *   dolby-vision  (uint)    — 0 or 1
 *   interlace     (uint)    — 0 or 1
 */
static void
post_signal_change_message(GstStreamboxSrc *self, const gchar *reason)
{
    guint width = 0, height = 0, frame_rate = 0;
    guint color_depth = 0, dolby_vision = 0, interlace = 0;
    gchar color_space[64] = "";
    gchar hdr_eotf[64] = "";

    /* Parse HDMI RX info sysfs */
    FILE *f = fopen(HDMIRX_INFO_PATH, "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            guint val;
            if (sscanf(line, "Hactive: %u", &val) == 1) {
                width = val;
            } else if (sscanf(line, "Vactive: %u", &val) == 1) {
                height = val;
            } else if (sscanf(line, "Frame Rate: %u", &val) == 1) {
                frame_rate = val;
            } else if (sscanf(line, "Color Depth: %u", &val) == 1) {
                color_depth = val;
            } else if (sscanf(line, "Dolby Vision: %u", &val) == 1) {
                dolby_vision = val;
            } else if (sscanf(line, "Interlace: %u", &val) == 1) {
                interlace = val;
            } else if (sscanf(line, " Color Space: %63[^\n]", color_space) == 1) {
                /* already captured */
            } else if (sscanf(line, " HDR EOTF: %63[^\n]", hdr_eotf) == 1) {
                /* already captured */
            }
        }
        fclose(f);
    } else {
        GST_WARNING_OBJECT(self, "Cannot read %s for signal change message: %s",
                           HDMIRX_INFO_PATH, strerror(errno));
    }

    GST_INFO_OBJECT(self,
        "Posting hdmi-signal-change: reason=%s %ux%u@%u depth=%u cs=%s eotf=%s dv=%u",
        reason, width, height, frame_rate, color_depth,
        color_space, hdr_eotf, dolby_vision);

    GstStructure *s = gst_structure_new("hdmi-signal-change",
        "reason",       G_TYPE_STRING, reason,
        "width",        G_TYPE_UINT,   width,
        "height",       G_TYPE_UINT,   height,
        "frame-rate",   G_TYPE_UINT,   frame_rate,
        "color-space",  G_TYPE_STRING, color_space,
        "color-depth",  G_TYPE_UINT,   color_depth,
        "hdr-eotf",     G_TYPE_STRING, hdr_eotf,
        "dolby-vision", G_TYPE_UINT,   dolby_vision,
        "interlace",    G_TYPE_UINT,   interlace,
        NULL);

    gst_element_post_message(GST_ELEMENT(self),
        gst_message_new_element(GST_OBJECT(self), s));
}

static GstFlowReturn
create_path_b(GstStreamboxSrc *self, GstBuffer **buf)
{
    if (self->vdin1_fd < 0)
        return GST_FLOW_ERROR;

    const gchar *clean_exit_reason = NULL;

    /* Push caps on first frame */
    if (!self->caps_set)
        push_current_caps(self);

    /* Periodically check for format change (G_FMT polling, backup to event-driven) */
    if (vdin1_check_format_change(self)) {
        GST_INFO_OBJECT(self, "Format change detected via G_FMT polling — exiting for restart");
        clean_exit_reason = "format-changed";
        goto clean_exit;
    }

    /*
     * Poll + DQBUF loop.
     * GstPushSrc::create() MUST return either a valid buffer or a non-OK flow.
     * We loop internally on EINTR / EAGAIN / poll timeouts instead of returning
     * GST_FLOW_OK without a buffer (which would confuse the base class).
     * After 5 consecutive poll timeouts (5 seconds) with no frame, attempt
     * signal recovery (event-driven or timeout-based).
     */
    guint timeout_count = 0;
    const guint max_timeouts = 5;

retry:
    if (self->flushing)
        return GST_FLOW_FLUSHING;

    {
        /*
         * Poll up to 3 fds:
         *   [0] vdin1 capture fd — POLLIN for new frames
         *   [1] flush pipe        — POLLIN for EOS/flush
         *   [2] signal monitor    — POLLPRI for V4L2_EVENT_SOURCE_CHANGE
         */
        struct pollfd pfds[3];
        nfds_t nfds = 2;

        pfds[0].fd = self->vdin1_fd;
        pfds[0].events = POLLIN;
        pfds[1].fd = self->flush_pipefd[0];
        pfds[1].events = POLLIN;

        if (self->signal_monitor_fd >= 0) {
            pfds[2].fd = self->signal_monitor_fd;
            pfds[2].events = POLLPRI;
            nfds = 3;
        }

        int pret = poll(pfds, nfds, 1000);
        if (pret < 0) {
            if (errno == EINTR)
                goto retry;
            GST_ERROR_OBJECT(self, "poll failed: %s", strerror(errno));
            return GST_FLOW_ERROR;
        }

        if (self->flushing || (pfds[1].revents & POLLIN))
            return GST_FLOW_FLUSHING;

        /*
         * Check signal monitor for SOURCE_CHANGE events.
         * This fires BEFORE we lose frames — the event arrives as soon as the
         * HDMI RX detects a signal parameter change, while vdin1 might still
         * have one or two frames in its queue. Handle it immediately.
         */
        if (nfds == 3 && (pfds[2].revents & POLLPRI)) {
            SignalEventAction act = handle_signal_event(self);
            if (act == SIGNAL_EVENT_CHANGED || act == SIGNAL_EVENT_LOST) {
                GST_INFO_OBJECT(self,
                    "Signal event during capture: action=%d — exiting for restart",
                    act);
                clean_exit_reason = (act == SIGNAL_EVENT_LOST)
                    ? "signal-lost" : "signal-changed";
                goto clean_exit;
            }
        }

        if (pret == 0) {
            /* Timeout -- likely no signal or vdin1 not producing frames */
            timeout_count++;
            GST_WARNING_OBJECT(self, "vdin1 poll timeout (%u/%u)",
                               timeout_count, max_timeouts);
            if (timeout_count >= max_timeouts) {
                GST_WARNING_OBJECT(self,
                    "vdin1: no frames after %u seconds — signal may be lost, exiting",
                    max_timeouts);
                clean_exit_reason = "signal-timeout";
                goto clean_exit;
            }
            goto retry;
        }
    }

    /* Normal frame delivery path — fall through to DQBUF below */
    goto do_dqbuf;

clean_exit:
    /*
     * Clean exit on signal change / loss / timeout.
     *
     * Drain any in-flight async GPU work for the 10-bit path so we don't
     * leave dangling fences or hold vdin1 buffers that stop() will free.
     * Then post hdmi-signal-change message and return EOS.
     */
    if (self->vdin1_10bit && self->vk_async_pending && self->vk_initialized) {
        guint prev_slot = 1 - self->vk_slot;
        GST_INFO_OBJECT(self, "Draining async GPU before clean exit (slot %u)", prev_slot);
        vdin1_vk_wait_async(self, prev_slot, self->vk_async_in_fd);

        /* Re-QBUF the vdin1 buffer held by async pipeline */
        guint pidx = self->vk_async_vdin1_idx;
        if (pidx < self->vdin1_n_bufs && self->vdin1_fd >= 0) {
            struct v4l2_buffer qb;
            struct v4l2_plane qp[1];
            memset(&qb, 0, sizeof(qb));
            memset(&qp, 0, sizeof(qp));
            qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            qb.memory = V4L2_MEMORY_DMABUF;
            qb.index = pidx;
            qb.m.planes = qp;
            qb.length = 1;
            qp[0].m.fd = self->vdin1_bufs[pidx].dma_fd;
            qp[0].length = self->vdin1_bufs[pidx].size;
            xioctl(self->vdin1_fd, VIDIOC_QBUF, &qb);
            self->vdin1_bufs[pidx].queued = TRUE;
        }

        /* Return output buffer to pool */
        if (self->vk_async_out_pool_idx >= 0 &&
            (guint)self->vk_async_out_pool_idx < self->p010_out_count) {
            g_mutex_lock(&self->p010_out_lock);
            self->p010_out_free[self->vk_async_out_pool_idx] = TRUE;
            g_mutex_unlock(&self->p010_out_lock);
        }

        self->vk_async_pending = FALSE;
    }

    post_signal_change_message(self, clean_exit_reason ? clean_exit_reason : "unknown");

    /*
     * Force STREAMOFF before returning EOS.
     *
     * When HDMI RX stops supplying frames, VPP has no new data and vdin1
     * enters an internal wait loop.  If we return GST_FLOW_EOS without
     * stopping vdin1 first, the subsequent stop() call from GStreamer's
     * state change may stall in STREAMOFF because the driver is stuck.
     * By issuing STREAMOFF now (while we're still on the streaming thread),
     * we ensure the driver is in a clean state before pipeline teardown.
     *
     * STREAMOFF is idempotent — stop_path_b() calling it again is harmless.
     */
    if (self->vdin1_fd >= 0) {
        GST_INFO_OBJECT(self, "clean_exit: forcing STREAMOFF on vdin1");
        vdin1_streamoff(self);
    }

    return GST_FLOW_EOS;

do_dqbuf:
    /* DQBUF (DMABUF mode) */
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[1];
    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(&planes, 0, sizeof(planes));
    v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    v4l2_buf.memory = V4L2_MEMORY_DMABUF;
    v4l2_buf.m.planes = planes;
    v4l2_buf.length = 1;

    if (xioctl(self->vdin1_fd, VIDIOC_DQBUF, &v4l2_buf) < 0) {
        if (errno == EAGAIN)
            goto retry;
        GST_ERROR_OBJECT(self, "VIDIOC_DQBUF failed: %s", strerror(errno));
        return GST_FLOW_ERROR;
    }

    guint idx = v4l2_buf.index;

    if (idx >= self->vdin1_n_bufs || self->vdin1_bufs[idx].dma_fd < 0) {
        GST_ERROR_OBJECT(self, "DQBUF returned invalid index %u", idx);
        return GST_FLOW_ERROR;
    }

    self->vdin1_bufs[idx].queued = FALSE;

    if (self->vdin1_10bit) {
        /*
         * 10-bit path: AMLY -> P010 via double-buffered async Vulkan GPU.
         *
         * Double-buffered pipeline with 1-frame lookahead:
         *
         * Frame 0 (priming, !vk_async_pending):
         *   1. Sync convert current frame -> wrap as GstBuffer (to return)
         *   2. Re-QBUF current vdin1 buf (GPU done reading)
         *   3. Poll + DQBUF a SECOND frame (may block ~16ms)
         *   4. Acquire pool buf, submit async GPU for second frame
         *   5. Save async state, set vk_async_pending = TRUE
         *   6. Return frame 0's GstBuffer
         *
         * Frame N (N>=1, vk_async_pending):
         *   1. Wait prev GPU fence (should be ~instant: GPU ran during encode)
         *   2. Re-QBUF prev vdin1 buf (GPU done reading)
         *   3. Wrap prev GPU output as GstBuffer (this is what we return)
         *   4. Current DQBUF'd frame -> acquire pool -> submit async GPU
         *   5. Save async state, return prev frame's GstBuffer
         *
         * Overlap: GPU converts frame N while encoder encodes frame N-1.
         * Throughput: max(GPU, encode) ~13ms instead of sum ~24ms -> 60fps.
         */

        if (!self->vk_async_pending) {
            /* === FRAME 0: Prime the pipeline === */
            guint64 t_prime_start = _get_time_us();

            /* --- Step 1: Sync convert current frame --- */
            gint out_idx = p010_pool_acquire(self);
            if (out_idx < 0) {
                GST_WARNING_OBJECT(self, "P010 pool exhausted (priming)");
                goto qbuf_return;
            }

            if (!vdin1_vk_convert_sync(self, self->vdin1_bufs[idx].dma_fd,
                                        (guint)out_idx, self->width, self->height)) {
                GST_ERROR_OBJECT(self, "Vulkan sync convert failed (priming)");
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[out_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                goto qbuf_return;
            }

            guint64 t_sync_done = _get_time_us();

            /* --- Step 2: Re-QBUF current vdin1 buf (GPU is done reading) --- */
            {
                struct v4l2_buffer qbuf;
                struct v4l2_plane qplanes[1];
                memset(&qbuf, 0, sizeof(qbuf));
                memset(&qplanes, 0, sizeof(qplanes));
                qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qbuf.memory = V4L2_MEMORY_DMABUF;
                qbuf.index = idx;
                qbuf.m.planes = qplanes;
                qbuf.length = 1;
                qplanes[0].m.fd = self->vdin1_bufs[idx].dma_fd;
                qplanes[0].length = self->vdin1_bufs[idx].size;

                if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qbuf) < 0) {
                    GST_WARNING_OBJECT(self, "re-QBUF(%u) after priming: %s",
                                       idx, strerror(errno));
                } else {
                    self->vdin1_bufs[idx].queued = TRUE;
                }
            }

            /* --- Step 3: Wrap sync result as GstBuffer (we return this) --- */
            int prime_out_fd = self->p010_out_fds[out_idx];
            int prime_dup_fd = dup(prime_out_fd);
            if (prime_dup_fd < 0) {
                GST_ERROR_OBJECT(self, "dup(out_fd=%d) failed: %s",
                                 prime_out_fd, strerror(errno));
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[out_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                return GST_FLOW_ERROR;
            }

            GstAllocator *dmabuf_alloc_p = gst_dmabuf_allocator_new();
            GstMemory *mem_p = gst_dmabuf_allocator_alloc(dmabuf_alloc_p, prime_dup_fd,
                                                           self->p010_out_size);
            gst_object_unref(dmabuf_alloc_p);

            if (!mem_p) {
                GST_ERROR_OBJECT(self, "dmabuf alloc failed (priming)");
                close(prime_dup_fd);
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[out_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                return GST_FLOW_ERROR;
            }

            GstBuffer *prime_buffer = gst_buffer_new();
            gst_buffer_append_memory(prime_buffer, mem_p);

            GST_BUFFER_PTS(prime_buffer) = GST_CLOCK_TIME_NONE;
            GST_BUFFER_DURATION(prime_buffer) = gst_util_uint64_scale_int(GST_SECOND,
                                                                           self->fps_d,
                                                                           self->fps_n);

            {
                gsize p_offsets[GST_VIDEO_MAX_PLANES] = { 0, };
                gint p_strides[GST_VIDEO_MAX_PLANES] = { 0, };
                p_strides[0] = self->width * 2;
                p_strides[1] = self->width * 2;
                p_offsets[0] = 0;
                p_offsets[1] = 0;

                gst_buffer_add_video_meta_full(prime_buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                                GST_VIDEO_FORMAT_P010_10LE,
                                                self->width, self->height,
                                                2, p_offsets, p_strides);
            }

            P010BufContext *pctx_p = g_new(P010BufContext, 1);
            pctx_p->self = GST_STREAMBOX_SRC(gst_object_ref(self));
            pctx_p->pool_index = (guint)out_idx;
            gst_mini_object_set_qdata(GST_MINI_OBJECT(prime_buffer),
                                       g_quark_from_static_string("p010-pool-ctx"),
                                       pctx_p, p010_buf_release);

            self->frame_count++;
            self->vk_frame_count++;

            guint64 t_wrap_done = _get_time_us();

            /* --- Step 4: DQBUF a SECOND frame to bootstrap async pipeline --- */
            {
                guint dq2_timeout_count = 0;
            prime_dqbuf_retry:
                if (self->flushing) {
                    *buf = prime_buffer;
                    return GST_FLOW_OK;  /* Return what we have */
                }

                struct pollfd pfds2[3];
                nfds_t nfds2 = 2;
                pfds2[0].fd = self->vdin1_fd;
                pfds2[0].events = POLLIN;
                pfds2[1].fd = self->flush_pipefd[0];
                pfds2[1].events = POLLIN;
                if (self->signal_monitor_fd >= 0) {
                    pfds2[2].fd = self->signal_monitor_fd;
                    pfds2[2].events = POLLPRI;
                    nfds2 = 3;
                }

                int pret2 = poll(pfds2, nfds2, 1000);
                if (pret2 < 0) {
                    if (errno == EINTR) goto prime_dqbuf_retry;
                    GST_WARNING_OBJECT(self, "priming: poll for frame 1 failed: %s",
                                       strerror(errno));
                    /* Return frame 0 without async bootstrap */
                    *buf = prime_buffer;
                    return GST_FLOW_OK;
                }
                if (self->flushing || (pfds2[1].revents & POLLIN)) {
                    *buf = prime_buffer;
                    return GST_FLOW_OK;
                }
                /* Signal event during priming — return frame 0 and let next
                 * create() call handle signal change via clean_exit */
                if (nfds2 == 3 && (pfds2[2].revents & POLLPRI)) {
                    SignalEventAction act2 = handle_signal_event(self);
                    if (act2 == SIGNAL_EVENT_CHANGED || act2 == SIGNAL_EVENT_LOST) {
                        GST_WARNING_OBJECT(self,
                            "priming: signal event during frame 1 poll, returning frame 0");
                        *buf = prime_buffer;
                        return GST_FLOW_OK;
                    }
                }
                if (pret2 == 0) {
                    dq2_timeout_count++;
                    if (dq2_timeout_count >= 3) {
                        GST_WARNING_OBJECT(self,
                            "priming: no second frame after 3s, running sync");
                        *buf = prime_buffer;
                        return GST_FLOW_OK;
                    }
                    goto prime_dqbuf_retry;
                }

                struct v4l2_buffer v4l2_buf2;
                struct v4l2_plane planes2[1];
                memset(&v4l2_buf2, 0, sizeof(v4l2_buf2));
                memset(&planes2, 0, sizeof(planes2));
                v4l2_buf2.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                v4l2_buf2.memory = V4L2_MEMORY_DMABUF;
                v4l2_buf2.m.planes = planes2;
                v4l2_buf2.length = 1;

                if (xioctl(self->vdin1_fd, VIDIOC_DQBUF, &v4l2_buf2) < 0) {
                    if (errno == EAGAIN) goto prime_dqbuf_retry;
                    GST_WARNING_OBJECT(self,
                        "priming: DQBUF frame 1 failed: %s", strerror(errno));
                    *buf = prime_buffer;
                    return GST_FLOW_OK;
                }

                guint idx2 = v4l2_buf2.index;
                if (idx2 >= self->vdin1_n_bufs || self->vdin1_bufs[idx2].dma_fd < 0) {
                    GST_WARNING_OBJECT(self, "priming: DQBUF2 invalid index %u", idx2);
                    *buf = prime_buffer;
                    return GST_FLOW_OK;
                }
                self->vdin1_bufs[idx2].queued = FALSE;

                /* --- Step 5: Acquire pool buf + submit async for frame 1 --- */
                gint out_idx2 = p010_pool_acquire(self);
                if (out_idx2 < 0) {
                    GST_WARNING_OBJECT(self,
                        "priming: P010 pool exhausted for frame 1, re-QBUF + sync fallback");
                    /* Re-QBUF frame 1 and fall back to sync mode */
                    struct v4l2_buffer qb2;
                    struct v4l2_plane qp2[1];
                    memset(&qb2, 0, sizeof(qb2));
                    memset(&qp2, 0, sizeof(qp2));
                    qb2.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                    qb2.memory = V4L2_MEMORY_DMABUF;
                    qb2.index = idx2;
                    qb2.m.planes = qp2;
                    qb2.length = 1;
                    qp2[0].m.fd = self->vdin1_bufs[idx2].dma_fd;
                    qp2[0].length = self->vdin1_bufs[idx2].size;
                    xioctl(self->vdin1_fd, VIDIOC_QBUF, &qb2);
                    self->vdin1_bufs[idx2].queued = TRUE;
                    *buf = prime_buffer;
                    return GST_FLOW_OK;
                }

                guint async_slot = self->vk_slot;
                if (!vdin1_vk_submit_async(self, self->vdin1_bufs[idx2].dma_fd,
                                            (guint)out_idx2,
                                            self->width, self->height, async_slot)) {
                    GST_WARNING_OBJECT(self,
                        "priming: async submit for frame 1 failed, sync fallback");
                    g_mutex_lock(&self->p010_out_lock);
                    self->p010_out_free[out_idx2] = TRUE;
                    g_mutex_unlock(&self->p010_out_lock);
                    /* Re-QBUF frame 1 */
                    struct v4l2_buffer qb2;
                    struct v4l2_plane qp2[1];
                    memset(&qb2, 0, sizeof(qb2));
                    memset(&qp2, 0, sizeof(qp2));
                    qb2.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                    qb2.memory = V4L2_MEMORY_DMABUF;
                    qb2.index = idx2;
                    qb2.m.planes = qp2;
                    qb2.length = 1;
                    qp2[0].m.fd = self->vdin1_bufs[idx2].dma_fd;
                    qp2[0].length = self->vdin1_bufs[idx2].size;
                    xioctl(self->vdin1_fd, VIDIOC_QBUF, &qb2);
                    self->vdin1_bufs[idx2].queued = TRUE;
                    *buf = prime_buffer;
                    return GST_FLOW_OK;
                }

                /* Async submitted! Save state for next create() call */
                self->vk_async_pending = TRUE;
                self->vk_async_out_pool_idx = out_idx2;
                self->vk_async_vdin1_idx = idx2;
                self->vk_async_in_fd = self->vdin1_bufs[idx2].dma_fd;
                self->vk_async_pts = GST_CLOCK_TIME_NONE;
                self->vk_async_duration = gst_util_uint64_scale_int(GST_SECOND,
                                                                     self->fps_d,
                                                                     self->fps_n);
                self->vk_slot = 1 - self->vk_slot;  /* Flip slot */
            }

            guint64 t_prime_end = _get_time_us();
            GST_LOG_OBJECT(self,
                "PATH_B ASYNC frame %lu (priming): sync_gpu=%luus wrap=%luus "
                "dqbuf2+submit=%luus TOTAL=%luus async_pending=%d",
                (unsigned long)self->frame_count,
                (unsigned long)(t_sync_done - t_prime_start),
                (unsigned long)(t_wrap_done - t_sync_done),
                (unsigned long)(t_prime_end - t_wrap_done),
                (unsigned long)(t_prime_end - t_prime_start),
                self->vk_async_pending);

            *buf = prime_buffer;
            return GST_FLOW_OK;

        } else {
            /* === FRAME N (N>=1): Double-buffered async pipeline === */
            guint64 t0 = _get_time_us();

            /*
             * Step 1: Wait for PREVIOUS frame's GPU to finish.
             * The encoder was processing frame N-2's output while the GPU
             * converted frame N-1. By now (~13ms later) the GPU should be
             * done (~11ms), so this wait should be near-instant.
             */
            guint prev_slot = 1 - self->vk_slot;
            if (!vdin1_vk_wait_async(self, prev_slot, self->vk_async_in_fd)) {
                GST_ERROR_OBJECT(self, "Async GPU wait failed for slot %u", prev_slot);
                /* Recovery: return pending output to pool, re-QBUF held vdin1 buf */
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[self->vk_async_out_pool_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                {
                    guint pidx = self->vk_async_vdin1_idx;
                    struct v4l2_buffer qb;
                    struct v4l2_plane qp[1];
                    memset(&qb, 0, sizeof(qb));
                    memset(&qp, 0, sizeof(qp));
                    qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                    qb.memory = V4L2_MEMORY_DMABUF;
                    qb.index = pidx;
                    qb.m.planes = qp;
                    qb.length = 1;
                    qp[0].m.fd = self->vdin1_bufs[pidx].dma_fd;
                    qp[0].length = self->vdin1_bufs[pidx].size;
                    xioctl(self->vdin1_fd, VIDIOC_QBUF, &qb);
                    self->vdin1_bufs[pidx].queued = TRUE;
                }
                self->vk_async_pending = FALSE;
                goto qbuf_return;
            }

            guint64 t1 = _get_time_us();

            /*
             * Step 2: Re-QBUF the PREVIOUS vdin1 buffer (GPU done reading it).
             */
            {
                guint pidx = self->vk_async_vdin1_idx;
                struct v4l2_buffer qbuf;
                struct v4l2_plane qplanes[1];
                memset(&qbuf, 0, sizeof(qbuf));
                memset(&qplanes, 0, sizeof(qplanes));
                qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qbuf.memory = V4L2_MEMORY_DMABUF;
                qbuf.index = pidx;
                qbuf.m.planes = qplanes;
                qbuf.length = 1;
                qplanes[0].m.fd = self->vdin1_bufs[pidx].dma_fd;
                qplanes[0].length = self->vdin1_bufs[pidx].size;

                if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qbuf) < 0) {
                    GST_WARNING_OBJECT(self, "re-QBUF(%u) after async wait: %s",
                                       pidx, strerror(errno));
                } else {
                    self->vdin1_bufs[pidx].queued = TRUE;
                }
            }

            /*
             * Step 3: Wrap PREVIOUS frame's GPU output as GstBuffer.
             * This is the buffer we will return to downstream.
             */
            gint prev_out_idx = self->vk_async_out_pool_idx;
            int prev_out_fd = self->p010_out_fds[prev_out_idx];

            int prev_dup_fd = dup(prev_out_fd);
            if (prev_dup_fd < 0) {
                GST_ERROR_OBJECT(self, "dup(prev_out_fd=%d) failed: %s",
                                 prev_out_fd, strerror(errno));
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[prev_out_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                self->vk_async_pending = FALSE;
                goto qbuf_return;
            }

            GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new();
            GstMemory *prev_mem = gst_dmabuf_allocator_alloc(dmabuf_alloc,
                                                              prev_dup_fd,
                                                              self->p010_out_size);
            gst_object_unref(dmabuf_alloc);

            if (!prev_mem) {
                GST_ERROR_OBJECT(self, "dmabuf alloc failed (async wrap)");
                close(prev_dup_fd);
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[prev_out_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                self->vk_async_pending = FALSE;
                goto qbuf_return;
            }

            GstBuffer *prev_buffer = gst_buffer_new();
            gst_buffer_append_memory(prev_buffer, prev_mem);

            GST_BUFFER_PTS(prev_buffer) = GST_CLOCK_TIME_NONE;
            GST_BUFFER_DURATION(prev_buffer) = self->vk_async_duration;

            {
                gsize a_offsets[GST_VIDEO_MAX_PLANES] = { 0, };
                gint a_strides[GST_VIDEO_MAX_PLANES] = { 0, };
                a_strides[0] = self->width * 2;
                a_strides[1] = self->width * 2;
                a_offsets[0] = 0;
                a_offsets[1] = 0;

                gst_buffer_add_video_meta_full(prev_buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                                GST_VIDEO_FORMAT_P010_10LE,
                                                self->width, self->height,
                                                2, a_offsets, a_strides);
            }

            P010BufContext *pctx = g_new(P010BufContext, 1);
            pctx->self = GST_STREAMBOX_SRC(gst_object_ref(self));
            pctx->pool_index = (guint)prev_out_idx;
            gst_mini_object_set_qdata(GST_MINI_OBJECT(prev_buffer),
                                       g_quark_from_static_string("p010-pool-ctx"),
                                       pctx, p010_buf_release);

            self->frame_count++;
            self->vk_frame_count++;

            guint64 t2 = _get_time_us();

            /*
             * Step 4: Acquire output pool buffer for the NEW (current) frame.
             */
            gint new_out_idx = p010_pool_acquire(self);
            if (new_out_idx < 0) {
                GST_WARNING_OBJECT(self,
                    "P010 pool exhausted (async), returning prev frame without overlap");
                /* Can't pipeline this time. Re-QBUF current vdin1 buf. */
                self->vk_async_pending = FALSE;
                struct v4l2_buffer qb_cur;
                struct v4l2_plane qp_cur[1];
                memset(&qb_cur, 0, sizeof(qb_cur));
                memset(&qp_cur, 0, sizeof(qp_cur));
                qb_cur.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qb_cur.memory = V4L2_MEMORY_DMABUF;
                qb_cur.index = idx;
                qb_cur.m.planes = qp_cur;
                qb_cur.length = 1;
                qp_cur[0].m.fd = self->vdin1_bufs[idx].dma_fd;
                qp_cur[0].length = self->vdin1_bufs[idx].size;
                if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qb_cur) < 0) {
                    GST_WARNING_OBJECT(self, "re-QBUF(%u) pool-exhausted: %s",
                                       idx, strerror(errno));
                } else {
                    self->vdin1_bufs[idx].queued = TRUE;
                }
                *buf = prev_buffer;
                return GST_FLOW_OK;
            }

            /*
             * Step 5: Submit async GPU for the NEW (current) frame.
             * Non-blocking: GPU will run while downstream encodes prev_buffer.
             */
            guint cur_slot = self->vk_slot;
            if (!vdin1_vk_submit_async(self, self->vdin1_bufs[idx].dma_fd,
                                        (guint)new_out_idx,
                                        self->width, self->height, cur_slot)) {
                GST_ERROR_OBJECT(self, "Vulkan async submit failed");
                g_mutex_lock(&self->p010_out_lock);
                self->p010_out_free[new_out_idx] = TRUE;
                g_mutex_unlock(&self->p010_out_lock);
                self->vk_async_pending = FALSE;
                /* Re-QBUF current vdin1 buf, still return prev frame */
                struct v4l2_buffer qb_cur;
                struct v4l2_plane qp_cur[1];
                memset(&qb_cur, 0, sizeof(qb_cur));
                memset(&qp_cur, 0, sizeof(qp_cur));
                qb_cur.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qb_cur.memory = V4L2_MEMORY_DMABUF;
                qb_cur.index = idx;
                qb_cur.m.planes = qp_cur;
                qb_cur.length = 1;
                qp_cur[0].m.fd = self->vdin1_bufs[idx].dma_fd;
                qp_cur[0].length = self->vdin1_bufs[idx].size;
                if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qb_cur) < 0) {
                    GST_WARNING_OBJECT(self, "re-QBUF(%u) submit-fail: %s",
                                       idx, strerror(errno));
                } else {
                    self->vdin1_bufs[idx].queued = TRUE;
                }
                *buf = prev_buffer;
                return GST_FLOW_OK;
            }

            guint64 t3 = _get_time_us();

            /* Save async state for next create() call */
            self->vk_async_pending = TRUE;
            self->vk_async_out_pool_idx = new_out_idx;
            self->vk_async_vdin1_idx = idx;
            self->vk_async_in_fd = self->vdin1_bufs[idx].dma_fd;
            self->vk_async_pts = GST_CLOCK_TIME_NONE;
            self->vk_async_duration = gst_util_uint64_scale_int(GST_SECOND,
                                                                 self->fps_d,
                                                                 self->fps_n);
            self->vk_slot = 1 - self->vk_slot;  /* Flip slot */

            /* Timing report */
            if (self->frame_count <= 10 || self->frame_count % 100 == 0) {
                GST_LOG_OBJECT(self,
                    "PATH_B ASYNC frame %lu: gpu_wait=%luus wrap=%luus "
                    "pool+submit=%luus TOTAL=%luus",
                    (unsigned long)self->frame_count,
                    (unsigned long)(t1 - t0),
                    (unsigned long)(t2 - t1),
                    (unsigned long)(t3 - t2),
                    (unsigned long)(t3 - t0));
            }

            if (self->frame_count == 1 || self->frame_count % 300 == 0) {
                GST_INFO_OBJECT(self,
                    "Path B frame %lu: %ux%u P010 async pool[%d] (submitted pool[%d])",
                    (unsigned long)self->frame_count,
                    self->width, self->height, prev_out_idx, new_out_idx);
            }

            *buf = prev_buffer;
            return GST_FLOW_OK;
        }

    } else {
        /*
         * 8-bit path: NV21 zero-copy passthrough.
         *
         * vdin1 wrote directly into our pre-allocated CMA DMA-buf via DMABUF mode.
         * We dup() the fd and wrap it with gst_dmabuf_allocator_alloc().
         * The original fd stays owned by us for re-QBUF.
         * When the GstBuffer is freed by downstream, the dispose callback
         * re-QBUFs the buffer back to vdin1.
         */
        guint32 actual_size = self->width * self->height * 3 / 2;

        int dup_fd = dup(self->vdin1_bufs[idx].dma_fd);
        if (dup_fd < 0) {
            GST_ERROR_OBJECT(self, "dup(fd=%d) failed: %s",
                             self->vdin1_bufs[idx].dma_fd, strerror(errno));
            goto qbuf_return;
        }

        GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new();
        GstMemory *mem = gst_dmabuf_allocator_alloc(dmabuf_alloc, dup_fd, actual_size);
        gst_object_unref(dmabuf_alloc);

        if (!mem) {
            GST_ERROR_OBJECT(self, "gst_dmabuf_allocator_alloc(fd=%d) failed", dup_fd);
            close(dup_fd);
            goto qbuf_return;
        }

        GstBuffer *buffer = gst_buffer_new();
        gst_buffer_append_memory(buffer, mem);

        /* Set timestamps — let GstBaseSrc do-timestamp assign running-time PTS */
        GST_BUFFER_PTS(buffer) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(GST_SECOND,
                                                                 self->fps_d,
                                                                 self->fps_n);

        /* Add video meta for NV21 */
        gsize offsets[GST_VIDEO_MAX_PLANES] = { 0, };
        gint strides[GST_VIDEO_MAX_PLANES] = { 0, };
        strides[0] = self->width;
        strides[1] = self->width;
        offsets[0] = 0;
        offsets[1] = (gsize)self->width * self->height;

        gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                        GST_VIDEO_FORMAT_NV21,
                                        self->width, self->height,
                                        2, offsets, strides);

        /*
         * Attach dispose callback: when downstream unrefs this GstBuffer,
         * we re-QBUF the V4L2 buffer so vdin1 can reuse it.
         */
        PathBBufContext *ctx = g_new(PathBBufContext, 1);
        ctx->self = GST_STREAMBOX_SRC(gst_object_ref(self));
        ctx->buf_index = idx;
        gst_mini_object_set_qdata(GST_MINI_OBJECT(buffer),
                                   g_quark_from_static_string("pathb-ctx"),
                                   ctx, path_b_buf_release);

        self->frame_count++;

        if (self->frame_count == 1 || self->frame_count % 300 == 0) {
            GST_INFO_OBJECT(self, "Path B frame %lu: %ux%u NV21 zero-copy fd=%d",
                             (unsigned long)self->frame_count,
                             self->width, self->height,
                             self->vdin1_bufs[idx].dma_fd);
        }

        *buf = buffer;
        return GST_FLOW_OK;
    }

qbuf_return:
    /* Error path: re-QBUF immediately and return error */
    {
        struct v4l2_buffer qbuf;
        struct v4l2_plane qplanes[1];
        memset(&qbuf, 0, sizeof(qbuf));
        memset(&qplanes, 0, sizeof(qplanes));
        qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        qbuf.memory = V4L2_MEMORY_DMABUF;
        qbuf.index = idx;
        qbuf.m.planes = qplanes;
        qbuf.length = 1;
        qplanes[0].m.fd = self->vdin1_bufs[idx].dma_fd;
        qplanes[0].length = self->vdin1_bufs[idx].size;

        if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qbuf) < 0) {
            GST_ERROR_OBJECT(self, "VIDIOC_QBUF(%u) failed: %s",
                             idx, strerror(errno));
        } else {
            self->vdin1_bufs[idx].queued = TRUE;
        }
    }
    return GST_FLOW_ERROR;
}

/* ====================================================================
 * Main create dispatch
 * ==================================================================== */

static GstFlowReturn
gst_streambox_src_create(GstPushSrc *pushsrc, GstBuffer **buf)
{
    GstStreamboxSrc *self = GST_STREAMBOX_SRC(pushsrc);

    if (!self->streaming)
        return GST_FLOW_ERROR;

    if (self->flushing)
        return GST_FLOW_FLUSHING;

    if (self->source_mode == GST_STREAMBOX_SOURCE_VFMCAP)
        return create_path_a(self, buf);
    else
        return create_path_b(self, buf);
}

/* ---------- Plugin registration ---------- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_streambox_src_debug, "streamboxsrc", 0,
                            "StreamBox HDMI capture source");

    return gst_element_register(plugin, "streamboxsrc", GST_RANK_PRIMARY,
                                GST_TYPE_STREAMBOX_SRC);
}

#ifndef VERSION
#define VERSION "1.0.0"
#endif

#ifndef PACKAGE
#define PACKAGE "gst-plugin-streambox"
#endif

#ifndef PACKAGE_NAME
#define PACKAGE_NAME "gst-plugin-streambox"
#endif

#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "https://streambox.ai/"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    streamboxsrc,
    "Dual-path HDMI capture: raw (vfm_cap) or color-processed (vdin1)",
    plugin_init,
    VERSION,
    "LGPL",
    PACKAGE_NAME,
    GST_PACKAGE_ORIGIN
)
