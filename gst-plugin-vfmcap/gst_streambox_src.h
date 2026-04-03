/*
 * gst_streambox_src.h - Unified GStreamer HDMI capture source element
 *
 * Dual-path capture from Amlogic A311D2 (T7):
 *   Path A (vfmcap): Ultra low latency raw capture via libvfmcap SDK.
 *     Captures from /dev/video_cap (vfm_cap kernel module), converts
 *     AMLY 10-bit to P010/NV12 via Vulkan GPU. No color processing.
 *   Path B (vdin1): Color-processed capture via VPP loopback.
 *     Direct V4L2 capture from /dev/video71 (vdin1). Full HDR->SDR
 *     tone mapping and gamut conversion done by VPP hardware.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_STREAMBOX_SRC_H__
#define __GST_STREAMBOX_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <vulkan/vulkan.h>
#include <vfmcap.h>

G_BEGIN_DECLS

#define GST_TYPE_STREAMBOX_SRC            (gst_streambox_src_get_type())
#define GST_STREAMBOX_SRC(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_STREAMBOX_SRC, GstStreamboxSrc))
#define GST_STREAMBOX_SRC_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_STREAMBOX_SRC, GstStreamboxSrcClass))
#define GST_IS_STREAMBOX_SRC(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_STREAMBOX_SRC))
#define GST_IS_STREAMBOX_SRC_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_STREAMBOX_SRC))

typedef struct _GstStreamboxSrc      GstStreamboxSrc;
typedef struct _GstStreamboxSrcClass GstStreamboxSrcClass;

/* ---------- Enumerations ---------- */

typedef enum {
    GST_STREAMBOX_SOURCE_VFMCAP = 0,  /* Path A: raw capture via libvfmcap */
    GST_STREAMBOX_SOURCE_VDIN1  = 1,  /* Path B: color-processed via vdin1 */
} GstStreamboxSourceMode;

typedef enum {
    GST_STREAMBOX_OUTPUT_NV12 = 0,
    GST_STREAMBOX_OUTPUT_P010 = 1,
} GstStreamboxOutputFormat;

/* Signal handling state machine */
typedef enum {
    GST_STREAMBOX_STATE_IDLE         = 0,  /* Not started */
    GST_STREAMBOX_STATE_WAITING      = 1,  /* Waiting for signal */
    GST_STREAMBOX_STATE_CONFIGURE    = 2,  /* Configuring after signal lock */
    GST_STREAMBOX_STATE_STREAMING    = 3,  /* Active frame delivery */
    GST_STREAMBOX_STATE_DRAINING     = 4,  /* Signal lost, draining */
    GST_STREAMBOX_STATE_RECONFIGURE  = 5,  /* Resolution changed, reconfiguring */
} GstStreamboxState;

/* ---------- Path B (vdin1) V4L2 buffer ---------- */

#define STREAMBOX_VDIN1_MAX_BUFFERS 8

typedef struct {
    int      dma_fd;    /* DMA-buf fd (owned by us, passed to vdin1 via QBUF) */
    size_t   size;      /* allocation size */
    gboolean queued;    /* TRUE if currently queued to vdin1 */
} StreamboxVdin1Buffer;

/* ---------- Vulkan DMA-buf cache (for Path B 10-bit) ---------- */

#define VDIN1_VK_DMABUF_CACHE_SIZE 8

typedef struct {
    int             fd;         /* original DMA-buf fd (key) */
    int             fd_dup;     /* dup'd fd consumed by Vulkan */
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceSize    size;
    int             valid;
    guint64         last_used;
} Vdin1VkCacheEntry;

/* ---------- Instance structure ---------- */

struct _GstStreamboxSrc
{
    GstPushSrc parent;

    /* ---- Properties ---- */
    GstStreamboxSourceMode source_mode;    /* vfmcap or vdin1 */
    gchar    *device;                       /* device path (auto-detected or manual) */
    guint     num_buffers;                  /* V4L2 buffer count */
    GstStreamboxOutputFormat output_fmt;    /* NV12 or P010 (Path A only) */
    guint     vdin1_input;                  /* V4L2 input index for vdin1 (default 6 = VPP post-blend) */

    /* ---- Signal state machine ---- */
    GstStreamboxState sig_state;
    GMutex    state_lock;

    /* ---- Shared state ---- */
    gboolean  streaming;           /* TRUE while streaming */
    guint     width;
    guint     height;
    guint     fps_n;               /* framerate numerator */
    guint     fps_d;               /* framerate denominator */
    gboolean  caps_set;            /* Whether caps have been pushed */
    guint64   frame_count;

    /* ---- Colorimetry (from HDMI RX signal info) ---- */
    gchar     colorimetry[64];     /* GStreamer colorimetry string for caps */

    /* ---- Path A (vfmcap) state ---- */
    vfmcap_ctx_t *cap_ctx;        /* libvfmcap context */
    int       heap_fd;             /* /dev/dma_heap/system-uncached fd */
    guint32   out_buf_size;        /* Output buffer size for current fmt */

    /* Path A output CMA buffer pool — acquire/release lifecycle.
     * Each buffer is marked "in use" when acquired for GPU conversion
     * and marked "free" when the downstream element (encoder) unrefs
     * the GstBuffer.  This prevents the GPU from overwriting a buffer
     * that the encoder hardware is still DMA-reading. */
#define PATHA_OUT_POOL_SIZE 12
    int       patha_out_fds[PATHA_OUT_POOL_SIZE]; /* pre-allocated CMA output DMA-buf fds */
    gboolean  patha_out_free[PATHA_OUT_POOL_SIZE]; /* TRUE = available for use */
    GMutex    patha_out_lock;      /* protects patha_out_free[] */
    guint     patha_out_count;     /* number allocated (0 if pool not init'd) */

    /* ---- Signal monitor (vfm_cap event fd, used by Path B) ---- */
    int       signal_monitor_fd;   /* /dev/video_cap fd for V4L2_EVENT_SOURCE_CHANGE */

    /* ---- Path B (vdin1) state ---- */
    int       vdin1_fd;            /* /dev/video71 fd */
    int       heap_cma_fd;         /* /dev/dma_heap/heap-codecmm fd (CMA for vdin1) */
    StreamboxVdin1Buffer vdin1_bufs[STREAMBOX_VDIN1_MAX_BUFFERS];
    guint     vdin1_n_bufs;        /* Actual number of allocated buffers */
    guint32   vdin1_pixfmt;        /* Current pixel format from vdin1 */
    guint     vdin1_num_planes;    /* Number of planes (from G_FMT) */
    guint32   vdin1_sizeimage;     /* Driver's sizeimage (includes alignment padding) */
    guint     vdin1_prev_width;    /* For format-change detection via G_FMT polling */
    guint     vdin1_prev_height;
    guint32   vdin1_prev_pixfmt;
    guint64   vdin1_fmt_poll_counter; /* Check G_FMT every N frames */

    /* ---- Path B Vulkan (AMLY -> P010, for 10-bit mode) ---- */
    gboolean  vdin1_10bit;             /* TRUE when output_fmt=P010 on Path B */
    guint32   vdin1_amly_sizeimage;    /* Driver's sizeimage for AMLY format */

    VkInstance              vk_instance;
    VkPhysicalDevice        vk_physical_device;
    VkDevice                vk_device;
    VkQueue                 vk_compute_queue;
    guint32                 vk_queue_family;
    VkCommandPool           vk_command_pool;
    VkCommandBuffer         vk_cmd[2];             /* double-buffered command buffers */
    VkFence                 vk_fences[2];           /* double-buffered fences */
    guint                   vk_slot;                /* current slot (0 or 1) */
    VkDescriptorPool        vk_descriptor_pool;
    VkDescriptorSetLayout   vk_descriptor_set_layout;
    VkDescriptorSet         vk_descriptor_sets[2];  /* one per slot for double-buffering */
    VkPipelineLayout        vk_pipeline_layout;
    VkPipeline              vk_pipeline_p010;
    VkShaderModule          vk_shader_p010;
    VkPhysicalDeviceMemoryProperties vk_memory_props;
    gboolean                vk_initialized;
    guint64                 vk_frame_count;

    /* Async GPU pipeline state (double-buffered, 1-frame delay) */
    gboolean  vk_async_pending;        /* TRUE if GPU work is in-flight on prev slot */
    gint      vk_async_out_pool_idx;   /* P010 pool index of async GPU output */
    guint     vk_async_vdin1_idx;      /* vdin1 buf index being read by async GPU */
    int       vk_async_in_fd;          /* DMA-buf fd of vdin1 buf (for DMA_BUF_SYNC end) */
    guint64   vk_async_pts;            /* PTS of the async frame (to attach to output) */
    guint64   vk_async_duration;       /* duration of the async frame */

    /* DMA-buf caches for Vulkan */
    Vdin1VkCacheEntry       vk_input_cache[VDIN1_VK_DMABUF_CACHE_SIZE];
    gint                    vk_input_cache_count;
    Vdin1VkCacheEntry       vk_output_cache;  /* single output cache entry */

    int                     vk_pending_in_fd;
    gboolean                vk_has_pending;

    /* Output DMA-buf pool for 10-bit path (avoids per-frame alloc) */
#define P010_OUT_POOL_SIZE  4
    int       p010_out_fds[4];         /* pre-allocated output DMA-buf fds */
    gboolean  p010_out_free[4];        /* TRUE = available for use */
    guint32   p010_out_size;           /* size of each buffer */
    guint     p010_out_count;          /* number allocated (0 if pool not init'd) */
    GMutex    p010_out_lock;           /* protects free[] */

    /* Expanded Vulkan output cache (one entry per pool slot) */
    Vdin1VkCacheEntry       vk_output_pool_cache[4];

    /* ---- Flushing / unlock ---- */
    gboolean  flushing;            /* Set by unlock() to break out of blocking */
    int       flush_pipefd[2];     /* pipe for signaling flush */
};

struct _GstStreamboxSrcClass
{
    GstPushSrcClass parent_class;
};

GType gst_streambox_src_get_type(void);

G_END_DECLS

#endif /* __GST_STREAMBOX_SRC_H__ */
