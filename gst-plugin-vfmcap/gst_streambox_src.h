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
    void    *start;     /* mmap'd address */
    size_t   length;    /* mmap'd length */
} StreamboxVdin1Buffer;

/* ---------- Instance structure ---------- */

struct _GstStreamboxSrc
{
    GstPushSrc parent;

    /* ---- Properties ---- */
    GstStreamboxSourceMode source_mode;    /* vfmcap or vdin1 */
    gchar    *device;                       /* device path (auto-detected or manual) */
    guint     num_buffers;                  /* V4L2 buffer count */
    GstStreamboxOutputFormat output_fmt;    /* NV12 or P010 (Path A only) */
    gboolean  auto_restart;                 /* Auto-restart on signal recovery */
    guint     signal_timeout_ms;            /* Timeout before EOS on signal loss */
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

    /* ---- Path A (vfmcap) state ---- */
    vfmcap_ctx_t *cap_ctx;        /* libvfmcap context */
    int       heap_fd;             /* /dev/dma_heap/system-uncached fd */
    guint32   out_buf_size;        /* Output buffer size for current fmt */

    /* ---- Path B (vdin1) state ---- */
    int       vdin1_fd;            /* /dev/video71 fd */
    StreamboxVdin1Buffer vdin1_bufs[STREAMBOX_VDIN1_MAX_BUFFERS];
    guint     vdin1_n_bufs;        /* Actual number of allocated buffers */
    guint32   vdin1_pixfmt;        /* Current pixel format from vdin1 */
    guint     vdin1_num_planes;    /* Number of planes (from G_FMT) */
    guint     vdin1_prev_width;    /* For format-change detection via G_FMT polling */
    guint     vdin1_prev_height;
    guint32   vdin1_prev_pixfmt;
    guint64   vdin1_fmt_poll_counter; /* Check G_FMT every N frames */

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
