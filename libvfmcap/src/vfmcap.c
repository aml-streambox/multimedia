/*
 * vfmcap.c - VFM Capture SDK - V4L2 capture core
 *
 * Wraps /dev/video_cap V4L2 device with zero-copy DMA-buf streaming.
 * CPU never touches frame data — only control plane (ioctls, poll).
 * Raw format conversion only (AMLY -> P010/NV12). No color space conversion.
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

#include "vfmcap.h"
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

/* ---------- Internal context ---------- */

struct vfmcap_ctx {
    int                  fd;           /* V4L2 device fd */
    char                 device[128];  /* Device path */
    int                  streaming;    /* 1 if STREAMON called */
    int                  vulkan_init;  /* 1 if Vulkan pipeline initialized */

    /* Current format */
    uint32_t             width;
    uint32_t             height;
    uint32_t             pixelformat;
    uint32_t             bytesperline;
    uint32_t             sizeimage;
    uint32_t             bitdepth;

    /* V4L2 MMAP buffers (flow-control tokens only) */
    unsigned int         num_buffers;

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

/* ---------- Lifecycle ---------- */

vfmcap_ctx_t *vfmcap_open(const char *device)
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
    }

    fprintf(stderr, "[vfmcap] Opened %s: %ux%u pixfmt=%.4s\n",
            dev, ctx->width, ctx->height, (char *)&ctx->pixelformat);

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
    }

    /* Subscribe to SOURCE_CHANGE events */
    struct v4l2_event_subscription sub;
    memset(&sub, 0, sizeof(sub));
    sub.type = V4L2_EVENT_SOURCE_CHANGE;
    xioctl(ctx->fd, VIDIOC_SUBSCRIBE_EVENT, &sub);

    /*
     * Drain any stale events queued before we subscribed.
     * The kernel may have queued SOURCE_CHANGE events from prior signal
     * transitions (e.g. HDMI source went to sleep and woke back up).
     * If we don't drain them, the first poll() in acquire_frame() will
     * return immediately with POLLPRI, potentially wasting time on
     * obsolete events.  Belt-and-suspenders: acquire_frame()'s poll loop
     * also handles this, but draining here keeps the initial acquire fast.
     */
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

    /* Initialize Vulkan if not already done */
    if (!ctx->vulkan_init && ctx->width > 0 && ctx->height > 0) {
        if (vfmcap_vk_init(ctx->width, ctx->height) == 0) {
            ctx->vulkan_init = 1;
        } else {
            fprintf(stderr, "[vfmcap] WARNING: Vulkan init failed: %s\n",
                    vfmcap_vk_last_error());
            /* Non-fatal — raw AMLY frames still available */
        }
    }

    fprintf(stderr, "[vfmcap] Streaming started: %ux%u pixfmt=%.4s vulkan=%s\n",
            ctx->width, ctx->height, (char *)&ctx->pixelformat,
            ctx->vulkan_init ? "yes" : "no");

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

    /*
     * Order matters!  Vulkan cleanup MUST happen BEFORE vfmcap_stop().
     *
     * vfmcap_stop() calls VIDIOC_REQBUFS(count=0) which tells the kernel
     * driver to free the CMA buffer allocations.  If the Vulkan pipeline
     * still holds VkDeviceMemory objects imported from those DMA-buf fds,
     * the subsequent vkFreeMemory() in vfmcap_vk_cleanup() will try to
     * unmap pages that have already been freed by the kernel, causing a
     * use-after-free oops (BUG: 00000000f2000800) and system crash.
     *
     * By cleaning up Vulkan first, all GPU-imported references to the
     * DMA-buf pages are released cleanly while the kernel buffers still
     * exist.  Then vfmcap_stop() can safely free them.
     */
    if (ctx->vulkan_init) {
        vfmcap_vk_cleanup();
        ctx->vulkan_init = 0;
    }

    if (ctx->streaming)
        vfmcap_stop(ctx);

    if (ctx->fd >= 0) {
        close(ctx->fd);
        ctx->fd = -1;
    }

    fprintf(stderr, "[vfmcap] Closed %s\n", ctx->device);
    free(ctx);
}

/* ---------- Helpers ---------- */

static int64_t now_ms_mono(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
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

    /*
     * Poll for buffer ready.
     *
     * Must loop because poll() can return early due to V4L2 events
     * (POLLPRI) without POLLIN being set — e.g. a SOURCE_CHANGE event
     * queued by the kernel when signal transitions occur.  Without the
     * loop, a stale event would consume the entire timeout in a single
     * poll() return, causing a false timeout even though vdin0 is
     * actively delivering frames.
     */
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
                    continue;  /* re-check deadline and retry */
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "poll() failed: %s", strerror(errno));
                return VFMCAP_ERR_IOCTL;
            }

            /* Handle events (if any) */
            if (pfd.revents & POLLPRI) {
                struct v4l2_event event;
                memset(&event, 0, sizeof(event));
                while (xioctl(ctx->fd, VIDIOC_DQEVENT, &event) == 0) {
                    if (event.type == V4L2_EVENT_SOURCE_CHANGE) {
                        /* Extract signal info from event data */
                        struct vfm_cap_signal_info_kern *sig =
                            (struct vfm_cap_signal_info_kern *)event.u.data;
                        if (sig->status == 1) { /* NOSIG */
                            return VFMCAP_ERR_NOSIG;
                        }
                        /* Signal changed — format may have updated */
                        struct v4l2_format fmt;
                        memset(&fmt, 0, sizeof(fmt));
                        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                        if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) == 0) {
                            ctx->width = fmt.fmt.pix_mp.width;
                            ctx->height = fmt.fmt.pix_mp.height;
                            ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
                            ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
                            ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
                        }
                        fprintf(stderr, "[vfmcap] Source changed: %ux%u\n",
                                ctx->width, ctx->height);
                    }
                    memset(&event, 0, sizeof(event));
                }
            }

            /* If data is available, break out to DQBUF */
            if (pfd.revents & POLLIN)
                break;

            /* Only event(s) — no data yet. Loop and re-poll with remaining time. */
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
        /* QBUF to recycle the buffer even on failure */
        xioctl(ctx->fd, VIDIOC_QBUF, &buf);
        return VFMCAP_ERR_IOCTL;
    }

    /* Fill frame descriptor */
    frame->dmabuf_fd = dmabuf_req.fd;
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

    return VFMCAP_OK;
}

void vfmcap_release_frame(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame)
{
    if (!ctx || !frame) return;

    /* Close the DMA-buf fd (releases kernel reference) */
    if (frame->dmabuf_fd >= 0) {
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

/* ---------- GPU format conversion ---------- */

int vfmcap_convert_p010(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int out_dmabuf_fd)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->vulkan_init) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan not initialized");
        return VFMCAP_ERR_VULKAN;
    }
    if (frame->dmabuf_fd < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Frame has no DMA-buf fd");
        return VFMCAP_ERR_INVAL;
    }

    int ret = vfmcap_vk_convert(frame->dmabuf_fd, out_dmabuf_fd,
                                frame->width, frame->height,
                                VFMCAP_VK_FMT_P010);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan P010 conversion failed: %s", vfmcap_vk_last_error());
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

int vfmcap_convert_nv12(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame, int out_dmabuf_fd)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->vulkan_init) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan not initialized");
        return VFMCAP_ERR_VULKAN;
    }
    if (frame->dmabuf_fd < 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Frame has no DMA-buf fd");
        return VFMCAP_ERR_INVAL;
    }

    int ret = vfmcap_vk_convert(frame->dmabuf_fd, out_dmabuf_fd,
                                frame->width, frame->height,
                                VFMCAP_VK_FMT_NV12);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan NV12 conversion failed: %s", vfmcap_vk_last_error());
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

int vfmcap_convert_submit(vfmcap_ctx_t *ctx, vfmcap_frame_t *frame,
                          int out_dmabuf_fd, vfmcap_output_fmt_t fmt)
{
    if (!ctx || !frame) return VFMCAP_ERR_INVAL;
    if (!ctx->vulkan_init) return VFMCAP_ERR_VULKAN;
    if (frame->dmabuf_fd < 0) return VFMCAP_ERR_INVAL;

    vfmcap_vk_fmt_t vk_fmt = (fmt == VFMCAP_FMT_NV12) ?
                              VFMCAP_VK_FMT_NV12 : VFMCAP_VK_FMT_P010;

    int ret = vfmcap_vk_convert_submit(frame->dmabuf_fd, out_dmabuf_fd,
                                       frame->width, frame->height, vk_fmt);
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan submit failed: %s", vfmcap_vk_last_error());
        return VFMCAP_ERR_VULKAN;
    }
    return VFMCAP_OK;
}

int vfmcap_convert_wait(vfmcap_ctx_t *ctx)
{
    if (!ctx) return VFMCAP_ERR_INVAL;
    if (!ctx->vulkan_init) return VFMCAP_ERR_VULKAN;

    int ret = vfmcap_vk_convert_wait();
    if (ret != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Vulkan wait failed: %s", vfmcap_vk_last_error());
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
    pfd.events = POLLPRI; /* events only, not data */
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

                /* Update cached format */
                struct v4l2_format fmt;
                memset(&fmt, 0, sizeof(fmt));
                fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                if (xioctl(ctx->fd, VIDIOC_G_FMT, &fmt) == 0) {
                    ctx->width = fmt.fmt.pix_mp.width;
                    ctx->height = fmt.fmt.pix_mp.height;
                    ctx->pixelformat = fmt.fmt.pix_mp.pixelformat;
                    ctx->bytesperline = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
                    ctx->sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
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

    /* Read format from device */
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
    /* fps, signal_type, hdr_status etc. come from sysfs or future G_CTRL */

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
        /* Y plane: width * height * 2 (16-bit per pixel)
         * UV plane: width * height (16-bit per UV pair, half height) */
        return width * height * 3;
    case VFMCAP_FMT_NV12:
        /* Y plane: width * height
         * UV plane: width * height / 2 */
        return width * height * 3 / 2;
    default:
        return 0;
    }
}
