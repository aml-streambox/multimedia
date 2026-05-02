/*
 * vfmcap.h - VFM Capture SDK Public API
 *
 * Zero-copy HDMI capture library for Amlogic A311D2 (T7) SoC.
 * Wraps /dev/video_cap (vfm_cap kernel module) with V4L2 streaming
 * and optional Vulkan-based format conversion on the Mali-G52 GPU.
 *
 * Default behavior is raw passthrough (zero GPU involvement).
 * GPU conversion is only performed when an output format is explicitly
 * requested via vfmcap_config_t.
 *
 * Data flow (CPU never touches pixel data when converting):
 *   vdin0 CMA buffer -> DMA-buf fd -> Vulkan GPU import ->
 *   graphics/compute shaders -> output DMA-buf -> downstream (encoder, display)
 *
 * Usage:
 *   vfmcap_config_t cfg = {0};
 *   cfg.output_format = VFMCAP_FMT_NV12;
 *   cfg.target_width = 1920;
 *   cfg.target_height = 1080;
 *   vfmcap_ctx_t *ctx = vfmcap_open("/dev/video_cap", &cfg);
 *   vfmcap_start(ctx, 6);
 *   while (running) {
 *       vfmcap_frame_t frame;
 *       int rc = vfmcap_acquire_frame(ctx, &frame, 1000);
 *       if (rc == VFMCAP_RECONFIGURED) {
 *           // signal changed, handle new width/height/format
 *       }
 *       // pass frame.dmabuf_fd to consumer
 *       vfmcap_release_frame(ctx, &frame);
 *   }
 *   vfmcap_stop(ctx);
 *   vfmcap_close(ctx);
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#ifndef VFMCAP_H
#define VFMCAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Opaque context ---------- */

typedef struct vfmcap_ctx vfmcap_ctx_t;

/* ---------- Output format enum ---------- */

typedef enum {
    VFMCAP_FMT_RAW = 0,           /* Default: raw passthrough, zero GPU */
    VFMCAP_FMT_NV12 = 1,          /* 8-bit semi-planar (Y 8-bit + UV interleaved 8-bit) */
    VFMCAP_FMT_NV21 = 2,          /* 8-bit semi-planar (Y 8-bit + VU interleaved 8-bit) */
    VFMCAP_FMT_P010 = 3,          /* 10-bit semi-planar (Y 16-bit LE + UV interleaved 16-bit LE) */
    VFMCAP_FMT_NV12_AFBC = 4,     /* 8-bit AFBC compressed NV12 */
    VFMCAP_FMT_A2B10G10R10_AFBC = 5, /* 10-bit AFBC compressed A2B10G10R10 */
    VFMCAP_FMT_VYUY_10BIT = 6,    /* Amlogic private VYUY 10-bit passthrough (no conversion) */
} vfmcap_output_fmt_t;

/* ---------- Color conversion mode ---------- */

typedef enum {
    VFMCAP_COLOR_PASSTHROUGH = 0,  /* No color space conversion (default) */
    VFMCAP_COLOR_HDR10_TO_SDR = 1, /* BT.2020+PQ (HDR10) -> BT.709 SDR (Reinhard tone map) */
    VFMCAP_COLOR_HLG_TO_SDR = 2,   /* BT.2020+HLG -> BT.709 SDR (Reinhard tone map) */
} vfmcap_color_mode_t;

/* ---------- Linear YUV plane layout ---------- */

typedef enum {
    VFMCAP_OUTPUT_LAYOUT_AUTO = 0,          /* Prefer single DMA-buf, fall back when needed */
    VFMCAP_OUTPUT_LAYOUT_SINGLE_DMABUF = 1, /* Y and UV in one DMA-buf; dmabuf_fd2 = -1 */
    VFMCAP_OUTPUT_LAYOUT_TWO_DMABUF = 2,    /* Y in dmabuf_fd, UV in dmabuf_fd2 */
} vfmcap_output_layout_t;

/* ---------- Frame descriptor ---------- */

/**
 * struct vfmcap_frame_t - Describes one captured frame
 *
 * Returned by vfmcap_acquire_frame(). The @dmabuf_fd points directly
 * to vdin0's CMA buffer when output_format is RAW or VYUY_10BIT.
 * When GPU conversion is requested, it points to the converted output
 * buffer managed by libvfmcap.
 *
 * The fd is valid until vfmcap_release_frame() is called.
 */
typedef struct {
    int      dmabuf_fd;       /* DMA-buf fd for the frame (Y plane for multi-planar) */
    int      dmabuf_fd2;      /* DMA-buf fd for UV plane (NV12/P010 linear) or -1 */
    uint32_t index;           /* V4L2 buffer index (internal use) */
    uint32_t width;           /* Frame width in pixels */
    uint32_t height;          /* Frame height in pixels */
    uint32_t bytesperline;    /* Bytes per line */
    uint32_t size;            /* Total buffer size in bytes */
    uint32_t pixelformat;     /* V4L2_PIX_FMT (e.g. 'AMLY', 'NV12') */
    uint32_t bitdepth;        /* 8, 10, or 12 */
    uint32_t sequence;        /* Frame sequence number */
    uint64_t timestamp_us;    /* Frame timestamp in microseconds */
    uint32_t signal_type;     /* HDR/colorimetry from vframe.signal_type */

    /* New fields for architecture redesign */
    uint64_t drm_modifier;    /* DRM format modifier (0 for linear) */
    uint32_t is_repeated;     /* 1 if this frame was repeated for framerate conversion */
    void    *priv;            /* Internal state (pool entries, etc.) */
} vfmcap_frame_t;

/* ---------- Signal info ---------- */

/**
 * struct vfmcap_signal_info_t - Current signal parameters
 *
 * Populated by vfmcap_get_signal_info() or delivered via
 * V4L2_EVENT_SOURCE_CHANGE.
 */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t fps;             /* fps * 1000 (e.g. 59940 = 59.94) */
    uint32_t pixelformat;     /* V4L2_PIX_FMT_* */
    uint32_t signal_type;     /* HDR/DV/colorimetry */
    uint32_t hdr_status;      /* 0=SDR, 1=HDR10, 2=HLG, 3=HDR10+, 4=DV */
    uint32_t is_interlaced;
    uint32_t status;          /* 0=STABLE, 1=NOSIG, 2=NOTSUP */
    uint32_t bitdepth;        /* 8, 10, or 12 */
} vfmcap_signal_info_t;

/* Signal status values */
#define VFMCAP_SIG_STABLE   0
#define VFMCAP_SIG_NOSIG    1
#define VFMCAP_SIG_NOTSUP   2

/* ---------- Event types ---------- */

#define VFMCAP_EVENT_SOURCE_CHANGE  1  /* Resolution/format changed */
#define VFMCAP_EVENT_NOSIG          2  /* Signal lost */
#define VFMCAP_EVENT_TIMEOUT        0  /* poll() timed out */
#define VFMCAP_EVENT_ERROR         -1  /* Error occurred */

/* ---------- Error codes ---------- */

#define VFMCAP_OK                 0
#define VFMCAP_ERR_OPEN          -1   /* Failed to open device */
#define VFMCAP_ERR_IOCTL         -2   /* V4L2 ioctl failed */
#define VFMCAP_ERR_TIMEOUT       -3   /* DQBUF / poll timed out */
#define VFMCAP_ERR_NOSIG         -4   /* No signal detected */
#define VFMCAP_ERR_VULKAN        -5   /* Vulkan initialization/conversion failed */
#define VFMCAP_ERR_NOMEM         -6   /* Out of memory */
#define VFMCAP_ERR_INVAL         -7   /* Invalid argument */
#define VFMCAP_ERR_STATE         -8   /* Wrong state (e.g. not started) */
#define VFMCAP_RECONFIGURED       1   /* Frame returned after dynamic reconfiguration */

/* ---------- Configuration ---------- */

typedef struct {
    vfmcap_output_fmt_t output_format;  /* Requested output format (default: RAW) */
    uint32_t target_width;              /* Target output width (0 = match source) */
    uint32_t target_height;             /* Target output height (0 = match source) */
    float    target_fps;                /* Target framerate (0 = match source) */
    vfmcap_color_mode_t color_mode;     /* HDR/color conversion mode (default: PASSTHROUGH) */
} vfmcap_config_t;

/* ---------- Lifecycle ---------- */

/**
 * vfmcap_open - Open the capture device
 * @device: Device path (NULL for default "/dev/video_cap")
 * @config: Capture configuration (NULL for default raw passthrough)
 *
 * Returns context pointer on success, NULL on failure.
 * Call vfmcap_last_error() for error details.
 */
vfmcap_ctx_t *vfmcap_open(const char *device, const vfmcap_config_t *config);

/**
 * vfmcap_set_output_layout - Select linear NV12/P010 DMA-buf plane layout
 * @ctx: Context from vfmcap_open()
 * @layout: AUTO, SINGLE_DMABUF, or TWO_DMABUF
 *
 * Must be called before vfmcap_start(). This does not affect RAW/VYUY passthrough
 * or AFBC outputs. Environment overrides are also supported:
 * VFMCAP_OUTPUT_LAYOUT=auto|single|two, VFMCAP_P010_SEPARATE=1,
 * VFMCAP_NV12_SEPARATE=1.
 */
int vfmcap_set_output_layout(vfmcap_ctx_t *ctx, vfmcap_output_layout_t layout);

/**
 * vfmcap_start - Start V4L2 streaming
 * @ctx: Context from vfmcap_open()
 * @num_buffers: Number of V4L2 MMAP buffers (4-16, recommend 6)
 *
 * Calls REQBUFS, STREAMON, and subscribes to SOURCE_CHANGE events.
 * Also initializes the Vulkan pipeline if conversion is configured.
 *
 * Returns VFMCAP_OK on success, negative error code on failure.
 */
int vfmcap_start(vfmcap_ctx_t *ctx, unsigned int num_buffers);

/**
 * vfmcap_stop - Stop V4L2 streaming
 * @ctx: Context
 *
 * Calls STREAMOFF and releases V4L2 buffers.
 * Vulkan resources remain allocated (for quick restart).
 */
void vfmcap_stop(vfmcap_ctx_t *ctx);

/**
 * vfmcap_close - Close device and free all resources
 * @ctx: Context (may be NULL)
 *
 * Stops streaming if active, destroys Vulkan resources, closes device.
 */
void vfmcap_close(vfmcap_ctx_t *ctx);

/* ---------- Frame acquisition ---------- */

/**
 * vfmcap_acquire_frame - Dequeue one frame (zero-copy)
 * @ctx: Context
 * @frame: Output frame descriptor
 * @timeout_ms: Timeout in milliseconds (-1 = block forever, 0 = non-blocking)
 *
 * Calls DQBUF + VFM_CAP_IOC_GET_DMABUF to get a DMA-buf fd pointing
 * directly to vdin0's CMA frame buffer. The fd is valid until
 * vfmcap_release_frame() is called.
 *
 * Returns VFMCAP_OK, VFMCAP_RECONFIGURED, VFMCAP_ERR_TIMEOUT, or negative error.
 */
int vfmcap_acquire_frame(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int timeout_ms);

/**
 * vfmcap_release_frame - Release frame back to capture device
 * @ctx: Context
 * @frame: Frame previously acquired via vfmcap_acquire_frame()
 *
 * Closes the DMA-buf fd and calls QBUF to recycle the buffer back
 * to vdin0. Must be called for every acquired frame.
 */
void vfmcap_release_frame(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame);

/* ---------- GPU format conversion (legacy API - deprecated) ---------- */

/**
 * vfmcap_convert_p010 - Convert AMLY frame to P010 via Vulkan GPU
 * @ctx: Context
 * @frame: Acquired frame (source AMLY DMA-buf)
 * @out_dmabuf_fd: Output DMA-buf fd (caller-allocated, must be large enough)
 *
 * DEPRECATED: Use vfmcap_open() with output_format=VFMCAP_FMT_P010 instead.
 */
int vfmcap_convert_p010(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int out_dmabuf_fd);

/**
 * vfmcap_convert_nv12 - Convert AMLY frame to NV12 via Vulkan GPU
 * @ctx: Context
 * @frame: Acquired frame (source AMLY DMA-buf)
 * @out_dmabuf_fd: Output DMA-buf fd (caller-allocated, must be large enough)
 *
 * DEPRECATED: Use vfmcap_open() with output_format=VFMCAP_FMT_NV12 instead.
 */
int vfmcap_convert_nv12(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int out_dmabuf_fd);

/**
 * vfmcap_convert_submit - Async submit GPU conversion
 * @ctx: Context
 * @frame: Acquired frame
 * @out_dmabuf_fd: Output DMA-buf fd
 * @fmt: Output format (VFMCAP_FMT_P010 or VFMCAP_FMT_NV12)
 *
 * DEPRECATED: Use vfmcap_open() with configured output format instead.
 */
int vfmcap_convert_submit(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame,
                          int out_dmabuf_fd, vfmcap_output_fmt_t fmt);

/**
 * vfmcap_convert_wait - Wait for async GPU conversion to complete
 * @ctx: Context
 *
 * DEPRECATED: Use vfmcap_open() with configured output format instead.
 */
int vfmcap_convert_wait(vfmcap_ctx_t *ctx);

/* ---------- Signal event handling ---------- */

/**
 * vfmcap_poll_event - Poll for V4L2 signal events
 * @ctx: Context
 * @timeout_ms: Timeout in milliseconds (-1 = block, 0 = non-blocking)
 *
 * Returns VFMCAP_EVENT_SOURCE_CHANGE, VFMCAP_EVENT_NOSIG,
 * VFMCAP_EVENT_TIMEOUT, or VFMCAP_EVENT_ERROR.
 */
int vfmcap_poll_event(vfmcap_ctx_t *ctx, int timeout_ms);

/**
 * vfmcap_get_signal_info - Get current signal parameters
 * @ctx: Context
 * @info: Output signal info
 *
 * Returns VFMCAP_OK or negative error code.
 */
int vfmcap_get_signal_info(vfmcap_ctx_t *ctx, vfmcap_signal_info_t *info);

/* ---------- Utility ---------- */

/**
 * vfmcap_last_error - Get last error message
 * @ctx: Context (may be NULL for open() errors)
 *
 * Returns a human-readable error string. The pointer is valid until
 * the next API call on this context.
 */
const char *vfmcap_last_error(vfmcap_ctx_t *ctx);

/**
 * vfmcap_output_size - Calculate output buffer size for a given format
 * @width: Frame width
 * @height: Frame height
 * @fmt: Output format
 *
 * Returns required DMA-buf size in bytes.
 */
uint32_t vfmcap_output_size(uint32_t width, uint32_t height, vfmcap_output_fmt_t fmt);

#ifdef __cplusplus
}
#endif

#endif /* VFMCAP_H */
