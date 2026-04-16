/*
 * vfmcap.c - VFM Capture SDK - V4L2 capture core
 *
 * Wraps /dev/video_cap V4L2 device with zero-copy DMA-buf streaming.
 * CPU never touches frame data — only control plane (ioctls, poll).
 * Optional GPU format conversion via Vulkan graphics/compute pipeline.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include "../include/vfmcap.h"
#include "vfmcap_vulkan.h"

/* ---------- Custom ioctl from vfm_cap kernel module ---------- */

struct vfm_cap_dmabuf_req {
    __u32 index;
    __s32 fd;
    __u32 size;
    __u32 reserved;
};

#define VFM_CAP_IOC_GET_DMABUF _IOWR('V', 192, struct vfm_cap_dmabuf_req)

/* ---------- Private signal info struct (matches kernel header) ---------- */

struct vfm_cap_signal_info_kern {
    __u32 width;
    __u32 height;
    __u32 fps;
    __u32 color_format;
    __u32 signal_type;
    __u32 hdr_status;
    __u32 is_interlaced;
    __u32 status;
    __u32 bitdepth;
};

/* ---------- Internal constants ---------- */

#define VFMCAP_DEFAULT_DEVICE   "/dev/video_cap"
#define VFMCAP_MAX_BUFFERS      16
#define VFMCAP_MIN_BUFFERS      4
#define VFMCAP_ERROR_SIZE       256

/* ---------- Internal frame priv (for output pool tracking) ---------- */

typedef struct {
    int             vdin_fd;
    int             out_y_fd;
    int             out_uv_fd;
    vfmcap_vk_fmt_t vk_fmt;
} vfmcap_frame_priv_t;

/* ---------- Internal context ---------- */

struct vfmcap_ctx {
    int                  fd;           /* V4L2 device fd */
    char                 device[128];  /* Device path */
    int                  streaming;    /* 1 if STREAMON called */

    /* Configuration */
    vfmcap_config_t      config;

    /* Current format */
    uint32_t             width;
    uint32_t             height;
    uint32_t             pixelformat;
    uint32_t             bytesperline;
    uint32_t             sizeimage;
    uint32_t             bitdepth;

    /* V4L2 MMAP buffers (flow-control tokens only) */
    unsigned int         num_buffers;

    /* Vulkan pipeline (NULL when not needed) */
    VulkanCtx           *vk;

    /* Framerate conversion state */
    uint64_t             ts_accum_us;  /* Timestamp accumulator for target_fps */
    uint64_t             last_frame_ts_us;
    vfmcap_frame_t       last_frame;   /* Cached for frame repeat */
    int                  has_last_frame;

    /* Dynamic reconfiguration state */
    int                  reconfig_pending; /* 1 if reconfig occurred, next acquire returns RECONFIGURED */
    int                  signal_lost;      /* 1 if signal currently lost */
    uint32_t             prev_width;       /* Dimensions before reconfig (for comparison) */
    uint32_t             prev_height;

    /* Error message */
    char                 last_error[VFMCAP_ERROR_SIZE];
};

/* Static error for pre-context failures */
static char s_global_error[VFMCAP_ERROR_SIZE] = {0};

/* ---------- Helper: xioctl with EINTR retry ---------- */

static int xioctl(int fd, unsigned long request, void *arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

/* ---------- Helpers ---------- */

static int64_t now_ms_mono(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int needs_vulkan(const vfmcap_config_t *cfg)
{
    if (!cfg) return 0;
    switch (cfg->output_format) {
    case VFMCAP_FMT_RAW:
    case VFMCAP_FMT_VYUY_10BIT:
        return 0;
    default:
        return 1;
    }
}

static vfmcap_vk_fmt_t to_vk_fmt(vfmcap_output_fmt_t fmt)
{
    switch (fmt) {
    case VFMCAP_FMT_NV12: return VFMCAP_VK_FMT_NV12;
    case VFMCAP_FMT_NV21: return VFMCAP_VK_FMT_NV21;
    case VFMCAP_FMT_P010: return VFMCAP_VK_FMT_P010;
    case VFMCAP_FMT_NV12_AFBC: return VFMCAP_VK_FMT_NV12_AFBC;
    case VFMCAP_FMT_A2B10G10R10_AFBC: return VFMCAP_VK_FMT_A2B10G10R10_AFBC;
    default: return VFMCAP_VK_FMT_NV12;
    }
}

/* ---------- Dynamic reconfiguration ---------- */

static int vfmcap_do_reconfig(vfmcap_ctx_t *ctx)
{
    uint32_t old_w = ctx->prev_width;
    uint32_t old_h = ctx->prev_height;

    fprintf(stderr, "[vfmcap] Reconfiguring: %ux%u -> %ux%u\n",
            old_w, old_h, ctx->width, ctx->height);

    /* 12.3: V4L2 reconfiguration cycle */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    xioctl(ctx->fd, VIDIOC_STREAMOFF, &type);

    struct v4l2_requestbuffers reqbufs;
    memset(&reqbufs, 0, sizeof(reqbufs));
    reqbufs.count = 0;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbufs.memory = V4L2_MEMORY_MMAP;
    xioctl(ctx->fd, VIDIOC_REQBUFS, &reqbufs);

    {
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        xioctl(ctx->fd, VIDIOC_G_FMT, &fmt);
        ctx->width = fmt.fmt.pix_mp.width;
        ctx->height = fmt.fmt.pix_mp.height;
        ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
        ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
        ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
        if (ctx->pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y'))
            ctx->bitdepth = 10;
        else if (ctx->pixelformat == V4L2_PIX_FMT_NV12 ||
                 ctx->pixelformat == V4L2_PIX_FMT_NV21)
            ctx->bitdepth = 8;
        else if (ctx->pixelformat == v4l2_fourcc('P', '0', '1', '0'))
            ctx->bitdepth = 10;
        else
            ctx->bitdepth = 8;
    }

    memset(&reqbufs, 0, sizeof(reqbufs));
    reqbufs.count = ctx->num_buffers ? ctx->num_buffers : 6;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbufs.memory = V4L2_MEMORY_MMAP;
    if (xioctl(ctx->fd, VIDIOC_REQBUFS, &reqbufs) < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "REQBUFS after reconfig failed: %s", strerror(errno));
        return VFMCAP_ERR_IOCTL;
    }
    ctx->num_buffers = reqbufs.count;

    for (unsigned int i = 0; i < ctx->num_buffers; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane plane;
        memset(&buf, 0, sizeof(buf));
        memset(&plane, 0, sizeof(plane));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = 1;
        buf.m.planes = &plane;
        if (xioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                     "QBUF(%u) after reconfig failed: %s", i, strerror(errno));
            return VFMCAP_ERR_IOCTL;
        }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "STREAMON after reconfig failed: %s", strerror(errno));
        return VFMCAP_ERR_IOCTL;
    }

    /* 12.4+12.5+12.8: Reconfigure Vulkan pools if dimensions changed */
    if (needs_vulkan(&ctx->config) && ctx->vk) {
        uint32_t dst_w = ctx->config.target_width ? ctx->config.target_width : ctx->width;
        uint32_t dst_h = ctx->config.target_height ? ctx->config.target_height : ctx->height;
        if (vfmcap_vk_reconfig_pools(ctx->vk, ctx->width, ctx->height,
                                      dst_w, dst_h,
                                      to_vk_fmt(ctx->config.output_format)) != 0) {
            fprintf(stderr, "[vfmcap] WARNING: Vulkan pool reconfig failed: %s\n",
                    vfmcap_vk_last_error(ctx->vk));
        }
    }

    /* Reset framerate accumulator */
    ctx->ts_accum_us = 0;

    fprintf(stderr, "[vfmcap] Reconfigured: %ux%u -> %ux%u pixfmt=%.4s\n",
            old_w, old_h, ctx->width, ctx->height, (char *)&ctx->pixelformat);

    return VFMCAP_OK;
}

/* ---------- Lifecycle ---------- */

vfmcap_ctx_t *vfmcap_open(const char *device, const vfmcap_config_t *config)
{
    const char *dev = device ? device : VFMCAP_DEFAULT_DEVICE;

    int fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        snprintf(s_global_error, sizeof(s_global_error),
                 "Failed to open %s: %s", dev, strerror(errno));
        return NULL;
    }

    /* Verify it's a V4L2 capture device */
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        snprintf(s_global_error, sizeof(s_global_error),
                 "VIDIOC_QUERYCAP failed: %s", strerror(errno));
        close(fd);
        return NULL;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) &&
        !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        snprintf(s_global_error, sizeof(s_global_error),
                 "%s is not a video capture device", dev);
        close(fd);
        return NULL;
    }

    vfmcap_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        snprintf(s_global_error, sizeof(s_global_error), "Out of memory");
        close(fd);
        return NULL;
    }

    ctx->fd = fd;
    strncpy(ctx->device, dev, sizeof(ctx->device) - 1);
    ctx->last_frame.dmabuf_fd = -1;

    if (config) {
        memcpy(&ctx->config, config, sizeof(ctx->config));
    } else {
        ctx->config.output_format = VFMCAP_FMT_RAW;
    }

    /* Read current format */
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) == 0) {
        ctx->width = fmt.fmt.pix_mp.width;
        ctx->height = fmt.fmt.pix_mp.height;
        ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
        ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
        ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
        /* Infer bitdepth from pixel format */
        if (ctx->pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y'))
            ctx->bitdepth = 10;
        else if (ctx->pixelformat == V4L2_PIX_FMT_NV12 ||
                 ctx->pixelformat == V4L2_PIX_FMT_NV21 ||
                 ctx->pixelformat == V4L2_PIX_FMT_YUV420 ||
                 ctx->pixelformat == V4L2_PIX_FMT_YVU420)
            ctx->bitdepth = 8;
        else if (ctx->pixelformat == v4l2_fourcc('P', '0', '1', '0'))
            ctx->bitdepth = 10;
        else
            ctx->bitdepth = 8; /* default assumption */
    }

    fprintf(stderr, "[vfmcap] Opened %s: %ux%u pixfmt=%.4s format=%d\n",
            dev, ctx->width, ctx->height, (char *)&ctx->pixelformat,
            ctx->config.output_format);

    return ctx;
}

int vfmcap_start(vfmcap_ctx_t *ctx, unsigned int num_buffers)
{
    if (!ctx) return VFMCAP_ERR_INVAL;
    if (ctx->streaming) return VFMCAP_OK; /* already started */

    if (num_buffers < VFMCAP_MIN_BUFFERS)
        num_buffers = VFMCAP_MIN_BUFFERS;
    if (num_buffers > VFMCAP_MAX_BUFFERS)
        num_buffers = VFMCAP_MAX_BUFFERS;

    /* Validate format compatibility for VYUY passthrough */
    if (ctx->config.output_format == VFMCAP_FMT_VYUY_10BIT && ctx->bitdepth != 10) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "VYUY 10-bit passthrough requires 10-bit input");
        return VFMCAP_ERR_INVAL;
    }

    /* Re-read format (may have changed due to signal change) */
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) == 0) {
        ctx->width = fmt.fmt.pix_mp.width;
        ctx->height = fmt.fmt.pix_mp.height;
        ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
        ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
        ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
        if (ctx->pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y'))
            ctx->bitdepth = 10;
        else if (ctx->pixelformat == V4L2_PIX_FMT_NV12 ||
                 ctx->pixelformat == V4L2_PIX_FMT_NV21 ||
                 ctx->pixelformat == V4L2_PIX_FMT_YUV420 ||
                 ctx->pixelformat == V4L2_PIX_FMT_YVU420)
            ctx->bitdepth = 8;
        else if (ctx->pixelformat == v4l2_fourcc('P', '0', '1', '0'))
            ctx->bitdepth = 10;
        else
            ctx->bitdepth = 8;
    }

    /* Subscribe to SOURCE_CHANGE events */
    struct v4l2_event_subscription sub;
    memset(&sub, 0, sizeof(sub));
    sub.type = V4L2_EVENT_SOURCE_CHANGE;
    xioctl(ctx->fd, VIDIOC_SUBSCRIBE_EVENT, &sub);

    /* Drain stale events */
    {
        struct v4l2_event ev;
        int drained = 0;
        memset(&ev, 0, sizeof(ev));
        while (xioctl(ctx->fd, VIDIOC_DQEVENT, &ev) == 0) {
            drained++;
            memset(&ev, 0, sizeof(ev));
        }
        if (drained > 0)
            fprintf(stderr, "[vfmcap] Drained %d stale event(s) at start\n", drained);
    }

    /* Request buffers (MMAP - used as flow-control tokens only) */
    struct v4l2_requestbuffers reqbufs;
    memset(&reqbufs, 0, sizeof(reqbufs));
    reqbufs.count = num_buffers;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbufs.memory = V4L2_MEMORY_MMAP;

    if (xioctl(ctx->fd, VIDIOC_REQBUFS, &reqbufs) < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "VIDIOC_REQBUFS failed: %s", strerror(errno));
        return VFMCAP_ERR_IOCTL;
    }
    ctx->num_buffers = reqbufs.count;

    fprintf(stderr, "[vfmcap] Allocated %u buffers\n", ctx->num_buffers);

    /* Queue all buffers */
    for (unsigned int i = 0; i < ctx->num_buffers; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane plane;
        memset(&buf, 0, sizeof(buf));
        memset(&plane, 0, sizeof(plane));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = 1;
        buf.m.planes = &plane;

        if (xioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                     "VIDIOC_QBUF(%u) failed: %s", i, strerror(errno));
            return VFMCAP_ERR_IOCTL;
        }
    }

    /* Start streaming */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "VIDIOC_STREAMON failed: %s", strerror(errno));
        return VFMCAP_ERR_IOCTL;
    }
    ctx->streaming = 1;
    ctx->signal_lost = 0;

    /* Initialize Vulkan if conversion is requested */
    if (needs_vulkan(&ctx->config) && !ctx->vk && ctx->width > 0 && ctx->height > 0) {
        uint32_t dst_w = ctx->config.target_width ? ctx->config.target_width : ctx->width;
        uint32_t dst_h = ctx->config.target_height ? ctx->config.target_height : ctx->height;
        if (vfmcap_vk_init(&ctx->vk, ctx->width, ctx->height,
                            dst_w, dst_h,
                            to_vk_fmt(ctx->config.output_format),
                            (uint32_t)ctx->config.color_mode) == 0) {
            /* success */
        } else {
            fprintf(stderr, "[vfmcap] WARNING: Vulkan init failed: %s\n",
                    vfmcap_vk_last_error(ctx->vk));
            if (ctx->vk) {
                vfmcap_vk_cleanup(ctx->vk);
                ctx->vk = NULL;
            }
            /* Non-fatal for raw paths, fatal for conversion paths */
            if (needs_vulkan(&ctx->config)) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "Vulkan init failed: %s", vfmcap_vk_last_error(NULL));
                return VFMCAP_ERR_VULKAN;
            }
        }
    }

    fprintf(stderr, "[vfmcap] Streaming started: %ux%u pixfmt=%.4s vulkan=%s\n",
            ctx->width, ctx->height, (char *)&ctx->pixelformat,
            ctx->vk ? "yes" : "no");

    return VFMCAP_OK;
}

void vfmcap_stop(vfmcap_ctx_t *ctx)
{
    if (!ctx || !ctx->streaming) return;

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    xioctl(ctx->fd, VIDIOC_STREAMOFF, &type);

    /* Release buffers */
    struct v4l2_requestbuffers reqbufs;
    memset(&reqbufs, 0, sizeof(reqbufs));
    reqbufs.count = 0;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    reqbufs.memory = V4L2_MEMORY_MMAP;
    xioctl(ctx->fd, VIDIOC_REQBUFS, &reqbufs);

    ctx->streaming = 0;
    ctx->num_buffers = 0;

    fprintf(stderr, "[vfmcap] Streaming stopped\n");
}

void vfmcap_close(vfmcap_ctx_t *ctx)
{
    if (!ctx) return;

    /* Vulkan cleanup MUST happen BEFORE vfmcap_stop() to avoid UAF */
    if (ctx->vk) {
        vfmcap_vk_cleanup(ctx->vk);
        ctx->vk = NULL;
    }

    if (ctx->streaming)
        vfmcap_stop(ctx);

    if (ctx->fd >= 0) {
        close(ctx->fd);
        ctx->fd = -1;
    }

    if (ctx->last_frame.dmabuf_fd >= 0) {
        close(ctx->last_frame.dmabuf_fd);
        ctx->last_frame.dmabuf_fd = -1;
    }

    fprintf(stderr, "[vfmcap] Closed %s\n", ctx->device);
    free(ctx);
}

/* ---------- Frame acquisition ---------- */

int vfmcap_acquire_frame(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int timeout_ms)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->streaming) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Not streaming (call vfmcap_start first)");
        return VFMCAP_ERR_STATE;
    }

    memset(frame, 0, sizeof(*frame));
    frame->dmabuf_fd = -1;

again:
    /* Signal loss recovery: poll for signal restoration */
    if (ctx->signal_lost) {
        for (;;) {
            struct pollfd pfd;
            pfd.fd = ctx->fd;
            pfd.events = POLLPRI;
            pfd.revents = 0;
            int pret = poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : 1000);
            if (pret == 0)
                return VFMCAP_ERR_NOSIG;
            if (pret < 0) {
                if (errno == EINTR) continue;
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "poll() during signal recovery failed: %s", strerror(errno));
                return VFMCAP_ERR_IOCTL;
            }
            if (pfd.revents & POLLPRI) {
                struct v4l2_event event;
                memset(&event, 0, sizeof(event));
                while (xioctl(ctx->fd, VIDIOC_DQEVENT, &event) == 0) {
                    if (event.type == V4L2_EVENT_SOURCE_CHANGE) {
                        struct vfm_cap_signal_info_kern *sig =
                            (struct vfm_cap_signal_info_kern *)event.u.data;
                        if (sig->status == 1) {
                            memset(&event, 0, sizeof(event));
                            continue;
                        }
                        ctx->signal_lost = 0;
                        ctx->prev_width = ctx->width;
                        ctx->prev_height = ctx->height;
                        struct v4l2_format fmt;
                        memset(&fmt, 0, sizeof(fmt));
                        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                        if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) == 0) {
                            ctx->width = fmt.fmt.pix_mp.width;
                            ctx->height = fmt.fmt.pix_mp.height;
                            ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
                            ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
                            ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
                            if (ctx->pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y'))
                                ctx->bitdepth = 10;
                            else if (ctx->pixelformat == V4L2_PIX_FMT_NV12 ||
                                     ctx->pixelformat == V4L2_PIX_FMT_NV21)
                                ctx->bitdepth = 8;
                            else if (ctx->pixelformat == v4l2_fourcc('P', '0', '1', '0'))
                                ctx->bitdepth = 10;
                            else
                                ctx->bitdepth = 8;
                        }
                        vfmcap_do_reconfig(ctx);
                        ctx->reconfig_pending = 1;
                        fprintf(stderr, "[vfmcap] Signal restored: %ux%u\n",
                                ctx->width, ctx->height);
                        break;
                    }
                    memset(&event, 0, sizeof(event));
                }
                if (!ctx->signal_lost)
                    break;
            }
        }
        goto again;
    }

    /* Poll for buffer ready */
    if (timeout_ms != 0) {
        int64_t deadline = now_ms_mono() + timeout_ms;

        for (;;) {
            int remaining = (int)(deadline - now_ms_mono());
            if (remaining <= 0)
                return VFMCAP_ERR_TIMEOUT;

            struct pollfd pfd;
            pfd.fd = ctx->fd;
            pfd.events = POLLIN | POLLPRI;
            pfd.revents = 0;

            int ret = poll(&pfd, 1, remaining);
            if (ret == 0)
                return VFMCAP_ERR_TIMEOUT;
            if (ret < 0) {
                if (errno == EINTR)
                    continue;
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "poll() failed: %s", strerror(errno));
                return VFMCAP_ERR_IOCTL;
            }

            /* Handle events */
            if (pfd.revents & POLLPRI) {
                struct v4l2_event event;
                memset(&event, 0, sizeof(event));
                while (xioctl(ctx->fd, VIDIOC_DQEVENT, &event) == 0) {
                    if (event.type == V4L2_EVENT_SOURCE_CHANGE) {
                        struct vfm_cap_signal_info_kern *sig =
                            (struct vfm_cap_signal_info_kern *)event.u.data;
                        if (sig->status == 1) {
                            ctx->signal_lost = 1;
                            return VFMCAP_ERR_NOSIG;
                        }
                        /* Signal changed — save prev dims, read new format, reconfig */
                        ctx->prev_width = ctx->width;
                        ctx->prev_height = ctx->height;
                        struct v4l2_format fmt;
                        memset(&fmt, 0, sizeof(fmt));
                        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                        if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) == 0) {
                            ctx->width = fmt.fmt.pix_mp.width;
                            ctx->height = fmt.fmt.pix_mp.height;
                            ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
                            ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
                            ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
                            if (ctx->pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y'))
                                ctx->bitdepth = 10;
                            else if (ctx->pixelformat == V4L2_PIX_FMT_NV12 ||
                                     ctx->pixelformat == V4L2_PIX_FMT_NV21)
                                ctx->bitdepth = 8;
                            else if (ctx->pixelformat == v4l2_fourcc('P', '0', '1', '0'))
                                ctx->bitdepth = 10;
                            else
                                ctx->bitdepth = 8;
                        }
                        int rc = vfmcap_do_reconfig(ctx);
                        if (rc != VFMCAP_OK) {
                            ctx->reconfig_pending = 1;
                            return rc;
                        }
                        ctx->reconfig_pending = 1;
                    }
                    memset(&event, 0, sizeof(event));
                }
            }

            if (pfd.revents & POLLIN)
                break;
        }
    }

    /* DQBUF */
    struct v4l2_buffer buf;
    struct v4l2_plane plane;
    memset(&buf, 0, sizeof(buf));
    memset(&plane, 0, sizeof(plane));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = 1;
    buf.m.planes = &plane;

    if (xioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN)
            return VFMCAP_ERR_TIMEOUT;
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "VIDIOC_DQBUF failed: %s", strerror(errno));
        return VFMCAP_ERR_IOCTL;
    }

    /* Get DMA-buf fd for this frame */
    struct vfm_cap_dmabuf_req dmabuf_req;
    memset(&dmabuf_req, 0, sizeof(dmabuf_req));
    dmabuf_req.index = buf.index;

    if (xioctl(ctx->fd, VFM_CAP_IOC_GET_DMABUF, &dmabuf_req) < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "VFM_CAP_IOC_GET_DMABUF failed: %s", strerror(errno));
        xioctl(ctx->fd, VIDIOC_QBUF, &buf);
        return VFMCAP_ERR_IOCTL;
    }

    /* Fill frame descriptor */
    frame->dmabuf_fd = dmabuf_req.fd;
    frame->dmabuf_fd2 = -1;
    frame->index = buf.index;
    frame->width = ctx->width;
    frame->height = ctx->height;
    frame->bytesperline = ctx->bytesperline;
    frame->size = dmabuf_req.size;
    frame->pixelformat = ctx->pixelformat;
    frame->bitdepth = ctx->bitdepth;
    frame->sequence = buf.sequence;
    frame->timestamp_us = (uint64_t)buf.timestamp.tv_sec * 1000000ULL +
                          (uint64_t)buf.timestamp.tv_usec;
    frame->drm_modifier = 0;
    frame->is_repeated = 0;
    frame->priv = NULL;

    /* Framerate conversion: drop frames to achieve target_fps */
    if (ctx->config.target_fps > 0.0f && !ctx->reconfig_pending) {
        uint64_t frame_interval_us = (uint64_t)(1000000.0f / ctx->config.target_fps);

        if (ctx->ts_accum_us == 0) {
            ctx->ts_accum_us = frame->timestamp_us;
        }

        int64_t ahead = (int64_t)ctx->ts_accum_us - (int64_t)frame->timestamp_us;
        if (ahead > (int64_t)(frame_interval_us / 2)) {
            close(dmabuf_req.fd);
            xioctl(ctx->fd, VIDIOC_QBUF, &buf);
            goto again;
        }
        ctx->ts_accum_us += frame_interval_us;
    }

    /* GPU conversion path (integrated into acquire) */
    if (needs_vulkan(&ctx->config) && ctx->vk) {
        uint32_t dst_w = ctx->config.target_width ? ctx->config.target_width : ctx->width;
        uint32_t dst_h = ctx->config.target_height ? ctx->config.target_height : ctx->height;
        vfmcap_vk_fmt_t vk_fmt = to_vk_fmt(ctx->config.output_format);

        /* For now, only use graphics path for 8-bit input -> NV12/P010 */
        if (ctx->bitdepth == 8 &&
            (vk_fmt == VFMCAP_VK_FMT_NV12 || vk_fmt == VFMCAP_VK_FMT_P010)) {
            int out_y_fd = -1, out_uv_fd = -1;
            int ret = vfmcap_vk_render_and_wait(ctx->vk, dmabuf_req.fd,
                                                ctx->width, ctx->height,
                                                dst_w, dst_h, vk_fmt,
                                                &out_y_fd, &out_uv_fd);
            if (ret != 0) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "Vulkan render failed: %s", vfmcap_vk_last_error(ctx->vk));
                close(dmabuf_req.fd);
                xioctl(ctx->fd, VIDIOC_QBUF, &buf);
                return VFMCAP_ERR_VULKAN;
            }

            vfmcap_frame_priv_t *priv = calloc(1, sizeof(*priv));
            if (!priv) {
                vfmcap_vk_release_output(ctx->vk, out_y_fd, out_uv_fd, vk_fmt);
                close(dmabuf_req.fd);
                xioctl(ctx->fd, VIDIOC_QBUF, &buf);
                return VFMCAP_ERR_NOMEM;
            }
            priv->vdin_fd = dmabuf_req.fd;
            priv->out_y_fd = out_y_fd;
            priv->out_uv_fd = out_uv_fd;
            priv->vk_fmt = vk_fmt;

            frame->dmabuf_fd = out_y_fd;
            frame->dmabuf_fd2 = out_uv_fd;
            frame->width = dst_w;
            frame->height = dst_h;
            frame->size = vfmcap_output_size(dst_w, dst_h, ctx->config.output_format);
            frame->pixelformat = (vk_fmt == VFMCAP_VK_FMT_NV12) ?
                                 V4L2_PIX_FMT_NV12 :
                                 v4l2_fourcc('P', '0', '1', '0');
            frame->priv = priv;
        }
        /* 10-bit input -> NV12/P010 via compute-to-graphics chain */
        else if (ctx->bitdepth == 10 &&
                 (vk_fmt == VFMCAP_VK_FMT_NV12 || vk_fmt == VFMCAP_VK_FMT_P010)) {
            int out_y_fd = -1, out_uv_fd = -1;
            int ret = vfmcap_vk_render_10bit_and_wait(ctx->vk, dmabuf_req.fd,
                                                       ctx->width, ctx->height,
                                                       dst_w, dst_h, vk_fmt,
                                                       &out_y_fd, &out_uv_fd);
            if (ret != 0) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "Vulkan 10-bit render failed: %s", vfmcap_vk_last_error(ctx->vk));
                close(dmabuf_req.fd);
                xioctl(ctx->fd, VIDIOC_QBUF, &buf);
                return VFMCAP_ERR_VULKAN;
            }

            vfmcap_frame_priv_t *priv = calloc(1, sizeof(*priv));
            if (!priv) {
                vfmcap_vk_release_output(ctx->vk, out_y_fd, out_uv_fd, vk_fmt);
                close(dmabuf_req.fd);
                xioctl(ctx->fd, VIDIOC_QBUF, &buf);
                return VFMCAP_ERR_NOMEM;
            }
            priv->vdin_fd = dmabuf_req.fd;
            priv->out_y_fd = out_y_fd;
            priv->out_uv_fd = out_uv_fd;
            priv->vk_fmt = vk_fmt;

            frame->dmabuf_fd = out_y_fd;
            frame->dmabuf_fd2 = out_uv_fd;
            frame->width = dst_w;
            frame->height = dst_h;
            frame->size = vfmcap_output_size(dst_w, dst_h, ctx->config.output_format);
            frame->pixelformat = (vk_fmt == VFMCAP_VK_FMT_NV12) ?
                                 V4L2_PIX_FMT_NV12 :
                                 v4l2_fourcc('P', '0', '1', '0');
            frame->priv = priv;
        }
    }

    /* Handle reconfig notification */
    if (ctx->reconfig_pending) {
        ctx->reconfig_pending = 0;
        return VFMCAP_RECONFIGURED;
    }

    return VFMCAP_OK;
}

void vfmcap_release_frame(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame)
{
    if (!ctx || !frame) return;

    if (frame->priv) {
        vfmcap_frame_priv_t *priv = (vfmcap_frame_priv_t *)frame->priv;
        if (ctx->vk) {
            vfmcap_vk_release_output(ctx->vk, priv->out_y_fd, priv->out_uv_fd, priv->vk_fmt);
        }
        if (priv->vdin_fd >= 0) {
            close(priv->vdin_fd);
            priv->vdin_fd = -1;
        }
        free(priv);
        frame->priv = NULL;
        frame->dmabuf_fd = -1;
        frame->dmabuf_fd2 = -1;
    } else if (frame->dmabuf_fd >= 0) {
        close(frame->dmabuf_fd);
        frame->dmabuf_fd = -1;
    }

    /* QBUF to recycle the buffer back to vdin0 */
    struct v4l2_buffer buf;
    struct v4l2_plane plane;
    memset(&buf, 0, sizeof(buf));
    memset(&plane, 0, sizeof(plane));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = frame->index;
    buf.length = 1;
    buf.m.planes = &plane;

    if (xioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
        fprintf(stderr, "[vfmcap] WARNING: QBUF(%u) failed: %s\n",
                frame->index, strerror(errno));
    }
}

/* ---------- GPU format conversion (legacy API) ---------- */

int vfmcap_convert_p010(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int out_dmabuf_fd)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->vk) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan not initialized");
        return VFMCAP_ERR_VULKAN;
    }
    if (frame->dmabuf_fd < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Frame has no DMA-buf fd");
        return VFMCAP_ERR_INVAL;
    }

    int ret = vfmcap_vk_convert(ctx->vk, frame->dmabuf_fd, out_dmabuf_fd,
                                frame->width, frame->height,
                                VFMCAP_VK_FMT_P010);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan P010 conversion failed: %s", vfmcap_vk_last_error(ctx->vk));
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

int vfmcap_convert_nv12(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int out_dmabuf_fd)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->vk) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan not initialized");
        return VFMCAP_ERR_VULKAN;
    }
    if (frame->dmabuf_fd < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Frame has no DMA-buf fd");
        return VFMCAP_ERR_INVAL;
    }

    int ret = vfmcap_vk_convert(ctx->vk, frame->dmabuf_fd, out_dmabuf_fd,
                                frame->width, frame->height,
                                VFMCAP_VK_FMT_NV12);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan NV12 conversion failed: %s", vfmcap_vk_last_error(ctx->vk));
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

int vfmcap_convert_submit(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame,
                          int out_dmabuf_fd, vfmcap_output_fmt_t fmt)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->vk) return VFMCAP_ERR_VULKAN;
    if (frame->dmabuf_fd < 0) return VFMCAP_ERR_INVAL;

    vfmcap_vk_fmt_t vk_fmt = (fmt == VFMCAP_FMT_NV12) ?
                              VFMCAP_VK_FMT_NV12 : VFMCAP_VK_FMT_P010;

    int ret = vfmcap_vk_convert_submit(ctx->vk, frame->dmabuf_fd, out_dmabuf_fd,
                                       frame->width, frame->height, vk_fmt);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan submit failed: %s", vfmcap_vk_last_error(ctx->vk));
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

int vfmcap_convert_wait(vfmcap_ctx_t *ctx)
{
    if (!ctx) return VFMCAP_ERR_INVAL;
    if (!ctx->vk) return VFMCAP_ERR_VULKAN;

    int ret = vfmcap_vk_convert_wait(ctx->vk);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan wait failed: %s", vfmcap_vk_last_error(ctx->vk));
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

/* ---------- Signal event handling ---------- */

int vfmcap_poll_event(vfmcap_ctx_t *ctx, int timeout_ms)
{
    if (!ctx) return VFMCAP_EVENT_ERROR;

    struct pollfd pfd;
    pfd.fd = ctx->fd;
    pfd.events = POLLPRI;
    pfd.revents = 0;

    int ret = poll(&pfd, 1, timeout_ms);
    if (ret == 0) return VFMCAP_EVENT_TIMEOUT;
    if (ret < 0) return VFMCAP_EVENT_ERROR;

    if (pfd.revents & POLLPRI) {
        struct v4l2_event event;
        memset(&event, 0, sizeof(event));
        if (xioctl(ctx->fd, VIDIOC_DQEVENT, &event) == 0) {
            if (event.type == V4L2_EVENT_SOURCE_CHANGE) {
                struct vfm_cap_signal_info_kern *sig =
                    (struct vfm_cap_signal_info_kern *)event.u.data;

                struct v4l2_format fmt;
                memset(&fmt, 0, sizeof(fmt));
                fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) == 0) {
                    ctx->width = fmt.fmt.pix_mp.width;
                    ctx->height = fmt.fmt.pix_mp.height;
                    ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
                    ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
                    ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
                    if (ctx->pixelformat == v4l2_fourcc('A', 'M', 'L', 'Y'))
                        ctx->bitdepth = 10;
                    else if (ctx->pixelformat == V4L2_PIX_FMT_NV12 ||
                             ctx->pixelformat == V4L2_PIX_FMT_NV21)
                        ctx->bitdepth = 8;
                    else if (ctx->pixelformat == v4l2_fourcc('P', '0', '1', '0'))
                        ctx->bitdepth = 10;
                    else
                        ctx->bitdepth = 8;
                }

                if (sig->status == 1)
                    return VFMCAP_EVENT_NOSIG;
                return VFMCAP_EVENT_SOURCE_CHANGE;
            }
        }
    }

    return VFMCAP_EVENT_TIMEOUT;
}

int vfmcap_get_signal_info(vfmcap_ctx_t *ctx, vfmcap_signal_info_t *info)
{
    if (!ctx || !info) return VFMCAP_ERR_INVAL;

    memset(info, 0, sizeof(*info));

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "VIDIOC_G_FMT failed: %s", strerror(errno));
        return VFMCAP_ERR_IOCTL;
    }

    info->width = fmt.fmt.pix_mp.width;
    info->height = fmt.fmt.pix_mp.height;
    info->pixelformat = fmt.fmt.pix_mp.pixelformat;

    return VFMCAP_OK;
}

/* ---------- Utility ---------- */

const char *vfmcap_last_error(vfmcap_ctx_t *ctx)
{
    if (ctx)
        return ctx->last_error;
    return s_global_error;
}

uint32_t vfmcap_output_size(uint32_t width, uint32_t height, vfmcap_output_fmt_t fmt)
{
    switch (fmt) {
    case VFMCAP_FMT_P010:
        return width * height * 3;
    case VFMCAP_FMT_NV12:
    case VFMCAP_FMT_NV21:
        return width * height * 3 / 2;
    case VFMCAP_FMT_RAW:
    case VFMCAP_FMT_VYUY_10BIT:
        /* Caller should use frame.size from acquire_frame */
        return width * height * 3; /* rough upper bound */
    case VFMCAP_FMT_NV12_AFBC:
    case VFMCAP_FMT_A2B10G10R10_AFBC:
        /* AFBC size depends on driver layout; rough estimate */
        return width * height * 2;
    default:
        return 0;
    }
}
