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
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>

#include "gst_streambox_src.h"

GST_DEBUG_CATEGORY_STATIC(gst_streambox_src_debug);
#define GST_CAT_DEFAULT gst_streambox_src_debug

/* ---------- Constants ---------- */

#define DEFAULT_DEVICE_VFMCAP  "/dev/video_cap"
#define DEFAULT_DEVICE_VDIN1   "/dev/video71"
#define DEFAULT_NUM_BUFFERS    6
#define DEFAULT_OUTPUT_FMT     GST_STREAMBOX_OUTPUT_NV12
#define DEFAULT_SOURCE_MODE    GST_STREAMBOX_SOURCE_VFMCAP
#define DEFAULT_AUTO_RESTART   TRUE
#define DEFAULT_SIGNAL_TIMEOUT 5000

#define DMA_HEAP_PATH          "/dev/dma_heap/system-uncached"

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

/* ---------- Properties ---------- */

enum
{
    PROP_0,
    PROP_SOURCE,
    PROP_DEVICE,
    PROP_NUM_BUFFERS,
    PROP_OUTPUT_FORMAT,
    PROP_AUTO_RESTART,
    PROP_SIGNAL_TIMEOUT,
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
        "framerate = (fraction) [ 0/1, 120/1 ]"
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

/* ---------- DMA-heap allocation (Path A) ---------- */

static int
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

/* Set format on vdin1 via VIDIOC_S_FMT (multi-plane). Request NV21. */
static gboolean
vdin1_set_format(GstStreamboxSrc *self, guint w, guint h)
{
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = w;
    fmt.fmt.pix_mp.height = h;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV21;
    fmt.fmt.pix_mp.num_planes = 1;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;

    if (xioctl(self->vdin1_fd, VIDIOC_S_FMT, &fmt) < 0) {
        GST_ERROR_OBJECT(self, "VIDIOC_S_FMT failed: %s", strerror(errno));
        return FALSE;
    }

    /* Re-read what driver actually set */
    guint np = 0;
    return vdin1_get_format(self, &self->width, &self->height,
                            &self->vdin1_pixfmt, &np);
}

static gboolean
vdin1_reqbufs(GstStreamboxSrc *self, guint count)
{
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(self->vdin1_fd, VIDIOC_REQBUFS, &req) < 0) {
        GST_ERROR_OBJECT(self, "VIDIOC_REQBUFS(%u) failed: %s",
                         count, strerror(errno));
        return FALSE;
    }

    if (req.count < 3) {
        GST_ERROR_OBJECT(self, "Insufficient buffer count: %u (need >= 3)",
                         req.count);
        return FALSE;
    }

    self->vdin1_n_bufs = req.count;
    GST_INFO_OBJECT(self, "vdin1: allocated %u buffers", req.count);
    return TRUE;
}

static gboolean
vdin1_mmap_buffers(GstStreamboxSrc *self)
{
    for (guint i = 0; i < self->vdin1_n_bufs; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        memset(&buf, 0, sizeof(buf));
        memset(&planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 1;  /* single plane for NV21 */

        if (xioctl(self->vdin1_fd, VIDIOC_QUERYBUF, &buf) < 0) {
            GST_ERROR_OBJECT(self, "VIDIOC_QUERYBUF(%u) failed: %s",
                             i, strerror(errno));
            return FALSE;
        }

        self->vdin1_bufs[i].length = planes[0].length;
        self->vdin1_bufs[i].start = mmap(NULL, planes[0].length,
                                         PROT_READ | PROT_WRITE,
                                         MAP_SHARED,
                                         self->vdin1_fd,
                                         planes[0].m.mem_offset);

        if (self->vdin1_bufs[i].start == MAP_FAILED) {
            GST_ERROR_OBJECT(self, "mmap buffer %u failed: %s",
                             i, strerror(errno));
            self->vdin1_bufs[i].start = NULL;
            return FALSE;
        }

        GST_DEBUG_OBJECT(self, "vdin1: buffer %u mapped at %p, length %zu",
                         i, self->vdin1_bufs[i].start,
                         self->vdin1_bufs[i].length);
    }
    return TRUE;
}

static void
vdin1_munmap_buffers(GstStreamboxSrc *self)
{
    for (guint i = 0; i < self->vdin1_n_bufs; i++) {
        if (self->vdin1_bufs[i].start && self->vdin1_bufs[i].start != MAP_FAILED) {
            munmap(self->vdin1_bufs[i].start, self->vdin1_bufs[i].length);
            self->vdin1_bufs[i].start = NULL;
        }
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
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 1;

        if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &buf) < 0) {
            GST_ERROR_OBJECT(self, "VIDIOC_QBUF(%u) failed: %s",
                             i, strerror(errno));
            return FALSE;
        }
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

    g_object_class_install_property(gobject_class, PROP_OUTPUT_FORMAT,
        g_param_spec_enum("output-format", "Output Format",
            "Output pixel format (Path A only: NV12 or P010)",
            GST_TYPE_STREAMBOX_OUTPUT_FORMAT,
            DEFAULT_OUTPUT_FMT,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_AUTO_RESTART,
        g_param_spec_boolean("auto-restart", "Auto Restart",
            "Auto-restart capture on signal recovery",
            DEFAULT_AUTO_RESTART,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_SIGNAL_TIMEOUT,
        g_param_spec_uint("signal-timeout", "Signal Timeout",
            "Milliseconds to wait for signal before EOS",
            500, 30000, DEFAULT_SIGNAL_TIMEOUT,
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

    pushsrc_class->create = GST_DEBUG_FUNCPTR(gst_streambox_src_create);
}

/* ---------- Instance init ---------- */

static void
gst_streambox_src_init(GstStreamboxSrc *self)
{
    self->source_mode = DEFAULT_SOURCE_MODE;
    self->device = NULL;  /* auto-detect */
    self->num_buffers = DEFAULT_NUM_BUFFERS;
    self->output_fmt = DEFAULT_OUTPUT_FMT;
    self->auto_restart = DEFAULT_AUTO_RESTART;
    self->signal_timeout_ms = DEFAULT_SIGNAL_TIMEOUT;
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

    /* Path A */
    self->cap_ctx = NULL;
    self->heap_fd = -1;
    self->out_buf_size = 0;

    /* Path B */
    self->vdin1_fd = -1;
    self->vdin1_n_bufs = 0;
    self->vdin1_pixfmt = 0;
    self->vdin1_num_planes = 0;
    self->vdin1_prev_width = 0;
    self->vdin1_prev_height = 0;
    self->vdin1_prev_pixfmt = 0;
    self->vdin1_fmt_poll_counter = 0;
    memset(self->vdin1_bufs, 0, sizeof(self->vdin1_bufs));

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
    case PROP_OUTPUT_FORMAT:
        self->output_fmt = g_value_get_enum(value);
        break;
    case PROP_AUTO_RESTART:
        self->auto_restart = g_value_get_boolean(value);
        break;
    case PROP_SIGNAL_TIMEOUT:
        self->signal_timeout_ms = g_value_get_uint(value);
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
    case PROP_OUTPUT_FORMAT:
        g_value_set_enum(value, self->output_fmt);
        break;
    case PROP_AUTO_RESTART:
        g_value_set_boolean(value, self->auto_restart);
        break;
    case PROP_SIGNAL_TIMEOUT:
        g_value_set_uint(value, self->signal_timeout_ms);
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
        return "NV21";
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

    self->cap_ctx = vfmcap_open(dev);
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
        self->fps_n = 60;
        self->fps_d = 1;
        vfmcap_release_frame(self->cap_ctx, &test_frame);
    } else {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                          ("No signal or cannot acquire test frame"),
                          ("%s", vfmcap_last_error(self->cap_ctx)));
        vfmcap_stop(self->cap_ctx);
        vfmcap_close(self->cap_ctx);
        self->cap_ctx = NULL;
        return FALSE;
    }

    vfmcap_output_fmt_t sdk_fmt = (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
                                   ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
    self->out_buf_size = vfmcap_output_size(self->width, self->height, sdk_fmt);

    self->streaming = TRUE;
    self->sig_state = GST_STREAMBOX_STATE_STREAMING;

    GST_INFO_OBJECT(self, "Path A started: %ux%u @ %u/%u, output=%s (%u bytes)",
                     self->width, self->height, self->fps_n, self->fps_d,
                     self->output_fmt == GST_STREAMBOX_OUTPUT_P010 ? "P010" : "NV12",
                     self->out_buf_size);
    return TRUE;
}

static void
stop_path_a(GstStreamboxSrc *self)
{
    if (self->cap_ctx) {
        vfmcap_stop(self->cap_ctx);
        vfmcap_close(self->cap_ctx);
        self->cap_ctx = NULL;
    }

    if (self->heap_fd >= 0) {
        close(self->heap_fd);
        self->heap_fd = -1;
    }
}

static GstFlowReturn
create_path_a(GstStreamboxSrc *self, GstBuffer **buf)
{
    if (!self->cap_ctx)
        return GST_FLOW_ERROR;

    /* Push caps on first frame */
    if (!self->caps_set)
        push_current_caps(self);

    /* Acquire frame */
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
        GST_WARNING_OBJECT(self, "No signal");
        if (self->auto_restart) {
            self->sig_state = GST_STREAMBOX_STATE_WAITING;
            /* Try to wait for signal recovery */
            GST_INFO_OBJECT(self, "Waiting for signal recovery...");
            ret = vfmcap_acquire_frame(self->cap_ctx, &frame,
                                       self->signal_timeout_ms);
            if (ret == VFMCAP_OK) {
                GST_INFO_OBJECT(self, "Signal recovered: %ux%u",
                                frame.width, frame.height);
                if (frame.width != self->width || frame.height != self->height) {
                    self->width = frame.width;
                    self->height = frame.height;
                    vfmcap_output_fmt_t sdk_fmt =
                        (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
                        ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
                    self->out_buf_size = vfmcap_output_size(
                        self->width, self->height, sdk_fmt);
                    self->caps_set = FALSE;
                    push_current_caps(self);
                }
                self->sig_state = GST_STREAMBOX_STATE_STREAMING;
                /* Fall through to process this frame */
            } else {
                GST_WARNING_OBJECT(self, "Signal not recovered within %u ms",
                                   self->signal_timeout_ms);
                return GST_FLOW_EOS;
            }
        } else {
            return GST_FLOW_EOS;
        }
    }

    if (ret != VFMCAP_OK) {
        GST_ERROR_OBJECT(self, "acquire_frame failed: %s",
                         vfmcap_last_error(self->cap_ctx));
        return GST_FLOW_ERROR;
    }

    /* Check for resolution change */
    if (frame.width != self->width || frame.height != self->height) {
        GST_INFO_OBJECT(self, "Resolution changed: %ux%u -> %ux%u",
                         self->width, self->height, frame.width, frame.height);
        self->width = frame.width;
        self->height = frame.height;

        vfmcap_output_fmt_t sdk_fmt = (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
                                       ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
        self->out_buf_size = vfmcap_output_size(self->width, self->height, sdk_fmt);
        self->caps_set = FALSE;

        vfmcap_release_frame(self->cap_ctx, &frame);

        /* Push new caps and continue (don't EOS) */
        push_current_caps(self);

        /* Re-acquire with new caps */
        ret = vfmcap_acquire_frame(self->cap_ctx, &frame, 1000);
        if (ret != VFMCAP_OK) {
            GST_WARNING_OBJECT(self, "Re-acquire after resolution change failed");
            return GST_FLOW_ERROR;
        }
    }

    /* Allocate output DMA-buf */
    int out_fd = alloc_output_dmabuf(self, self->out_buf_size);
    if (out_fd < 0) {
        GST_ERROR_OBJECT(self, "Failed to allocate output DMA-buf");
        vfmcap_release_frame(self->cap_ctx, &frame);
        return GST_FLOW_ERROR;
    }

    /* GPU convert: AMLY -> NV12 or P010 */
    vfmcap_output_fmt_t sdk_fmt = (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
                                   ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
    if (sdk_fmt == VFMCAP_FMT_P010)
        ret = vfmcap_convert_p010(self->cap_ctx, &frame, out_fd);
    else
        ret = vfmcap_convert_nv12(self->cap_ctx, &frame, out_fd);

    /* Release input frame immediately after GPU is done */
    uint64_t frame_ts = frame.timestamp_us;
    vfmcap_release_frame(self->cap_ctx, &frame);

    if (ret != VFMCAP_OK) {
        GST_ERROR_OBJECT(self, "GPU conversion failed: %s",
                         vfmcap_last_error(self->cap_ctx));
        close(out_fd);
        return GST_FLOW_ERROR;
    }

    /* Wrap output DMA-buf in GstBuffer */
    GstAllocator *dmabuf_alloc = gst_dmabuf_allocator_new();
    GstMemory *mem = gst_dmabuf_allocator_alloc(dmabuf_alloc, out_fd,
                                                 self->out_buf_size);
    gst_object_unref(dmabuf_alloc);

    if (!mem) {
        GST_ERROR_OBJECT(self, "Failed to wrap DMA-buf fd as GstMemory");
        close(out_fd);
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_buffer_new();
    gst_buffer_append_memory(buffer, mem);

    GST_BUFFER_PTS(buffer) = frame_ts * GST_USECOND;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(GST_SECOND,
                                                             self->fps_d,
                                                             self->fps_n);

    /* Add video meta */
    GstVideoFormat gst_fmt = (self->output_fmt == GST_STREAMBOX_OUTPUT_P010)
                              ? GST_VIDEO_FORMAT_P010_10LE
                              : GST_VIDEO_FORMAT_NV12;
    gsize offsets[GST_VIDEO_MAX_PLANES] = { 0, };
    gint strides[GST_VIDEO_MAX_PLANES] = { 0, };

    if (self->output_fmt == GST_STREAMBOX_OUTPUT_P010) {
        strides[0] = self->width * 2;
        strides[1] = self->width * 2;
        offsets[0] = 0;
        offsets[1] = (gsize)self->width * self->height * 2;
    } else {
        strides[0] = self->width;
        strides[1] = self->width;
        offsets[0] = 0;
        offsets[1] = (gsize)self->width * self->height;
    }

    gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                    gst_fmt, self->width, self->height,
                                    2, offsets, strides);

    self->frame_count++;

    if (self->frame_count == 1 || self->frame_count % 300 == 0) {
        GST_INFO_OBJECT(self, "Path A frame %lu: %ux%u %s",
                         (unsigned long)self->frame_count,
                         self->width, self->height,
                         self->output_fmt == GST_STREAMBOX_OUTPUT_P010 ? "P010" : "NV12");
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

    /* Request buffers */
    guint req_count = self->num_buffers;
    if (req_count > STREAMBOX_VDIN1_MAX_BUFFERS)
        req_count = STREAMBOX_VDIN1_MAX_BUFFERS;
    if (!vdin1_reqbufs(self, req_count))
        goto fail;

    /* mmap buffers */
    if (!vdin1_mmap_buffers(self))
        goto fail;

    /* Queue all buffers */
    if (!vdin1_qbuf_all(self))
        goto fail;

    /* Start streaming */
    if (!vdin1_streamon(self))
        goto fail;

    self->streaming = TRUE;
    self->sig_state = GST_STREAMBOX_STATE_STREAMING;
    self->fps_n = 60;
    self->fps_d = 1;

    GST_INFO_OBJECT(self, "Path B started: %ux%u NV21 (source %ux%u, %u bufs)",
                     self->width, self->height, src_w, src_h,
                     self->vdin1_n_bufs);
    return TRUE;

fail:
    vdin1_munmap_buffers(self);
    close(self->vdin1_fd);
    self->vdin1_fd = -1;
    return FALSE;
}

static void
stop_path_b(GstStreamboxSrc *self)
{
    if (self->vdin1_fd >= 0) {
        vdin1_streamoff(self);
        vdin1_munmap_buffers(self);

        /* Free buffers */
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 0;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(self->vdin1_fd, VIDIOC_REQBUFS, &req);

        close(self->vdin1_fd);
        self->vdin1_fd = -1;
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
 * Reconfigure vdin1 after a format change.
 * Returns TRUE if reconfiguration succeeded and streaming is resumed.
 */
static gboolean
vdin1_reconfigure(GstStreamboxSrc *self)
{
    GST_INFO_OBJECT(self, "vdin1 reconfiguring...");
    self->sig_state = GST_STREAMBOX_STATE_RECONFIGURE;

    /* Stop streaming */
    vdin1_streamoff(self);
    vdin1_munmap_buffers(self);

    /* Free old buffers */
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 0;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(self->vdin1_fd, VIDIOC_REQBUFS, &req);

    /* Small delay to let VPP stabilize after resolution change */
    g_usleep(100000);  /* 100ms */

    /* Re-select input (S_INPUT resets driver state) */
    if (!vdin1_set_input(self, self->vdin1_input))
        return FALSE;

    /* Query new source resolution from HDMI RX */
    guint src_w = 0, src_h = 0;
    if (!hdmirx_get_source_resolution(self, &src_w, &src_h)) {
        /* Fallback: read whatever vdin1 currently reports */
        guint np = 0;
        guint32 pf = 0;
        vdin1_get_format(self, &src_w, &src_h, &pf, &np);
    }

    if (src_w == 0 || src_h == 0) {
        GST_WARNING_OBJECT(self, "No resolution during reconfigure -- no signal");
        self->sig_state = GST_STREAMBOX_STATE_WAITING;
        return FALSE;
    }

    /* Set format at full source resolution */
    if (!vdin1_set_format(self, src_w, src_h))
        return FALSE;

    self->vdin1_prev_width = self->width;
    self->vdin1_prev_height = self->height;
    self->vdin1_prev_pixfmt = self->vdin1_pixfmt;

    /* Re-allocate buffers */
    guint req_count = self->num_buffers;
    if (req_count > STREAMBOX_VDIN1_MAX_BUFFERS)
        req_count = STREAMBOX_VDIN1_MAX_BUFFERS;
    if (!vdin1_reqbufs(self, req_count))
        return FALSE;
    if (!vdin1_mmap_buffers(self))
        return FALSE;
    if (!vdin1_qbuf_all(self))
        return FALSE;
    if (!vdin1_streamon(self))
        return FALSE;

    /* Push new caps downstream */
    self->caps_set = FALSE;

    self->sig_state = GST_STREAMBOX_STATE_STREAMING;
    GST_INFO_OBJECT(self, "vdin1 reconfigured: %ux%u pixfmt=0x%08x",
                     self->width, self->height, self->vdin1_pixfmt);
    return TRUE;
}

static GstFlowReturn
create_path_b(GstStreamboxSrc *self, GstBuffer **buf)
{
    if (self->vdin1_fd < 0)
        return GST_FLOW_ERROR;

    /* Push caps on first frame */
    if (!self->caps_set)
        push_current_caps(self);

    /* Periodically check for format change */
    if (vdin1_check_format_change(self)) {
        if (!vdin1_reconfigure(self)) {
            if (self->auto_restart) {
                /* Wait for signal to come back */
                GST_INFO_OBJECT(self, "Waiting for signal recovery...");
                for (guint i = 0; i < self->signal_timeout_ms / 500; i++) {
                    if (self->flushing)
                        return GST_FLOW_FLUSHING;
                    g_usleep(500000);
                    if (vdin1_reconfigure(self))
                        goto reconfigure_ok;
                }
                GST_WARNING_OBJECT(self, "Signal not recovered within %u ms",
                                   self->signal_timeout_ms);
                return GST_FLOW_EOS;
            }
            return GST_FLOW_EOS;
        }
reconfigure_ok:
        push_current_caps(self);
    }

    /*
     * Poll + DQBUF loop.
     * GstPushSrc::create() MUST return either a valid buffer or a non-OK flow.
     * We loop internally on EINTR / EAGAIN / poll timeouts instead of returning
     * GST_FLOW_OK without a buffer (which would confuse the base class).
     * After 5 consecutive poll timeouts (5 seconds) with no frame, return EOS
     * or error rather than looping forever.
     */
    guint timeout_count = 0;
    const guint max_timeouts = 5;

retry:
    if (self->flushing)
        return GST_FLOW_FLUSHING;

    {
        struct pollfd pfds[2];
        pfds[0].fd = self->vdin1_fd;
        pfds[0].events = POLLIN;
        pfds[1].fd = self->flush_pipefd[0];
        pfds[1].events = POLLIN;

        int pret = poll(pfds, 2, 1000);
        if (pret < 0) {
            if (errno == EINTR)
                goto retry;
            GST_ERROR_OBJECT(self, "poll failed: %s", strerror(errno));
            return GST_FLOW_ERROR;
        }

        if (self->flushing || (pfds[1].revents & POLLIN))
            return GST_FLOW_FLUSHING;

        if (pret == 0) {
            /* Timeout -- likely no signal or vdin1 not producing frames */
            timeout_count++;
            GST_WARNING_OBJECT(self, "vdin1 poll timeout (%u/%u)",
                               timeout_count, max_timeouts);
            if (timeout_count >= max_timeouts) {
                GST_ERROR_OBJECT(self,
                    "vdin1: no frames after %u seconds, giving up", max_timeouts);
                return GST_FLOW_EOS;
            }
            goto retry;
        }
    }

    /* DQBUF */
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[1];
    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(&planes, 0, sizeof(planes));
    v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    v4l2_buf.memory = V4L2_MEMORY_MMAP;
    v4l2_buf.m.planes = planes;
    v4l2_buf.length = 1;

    if (xioctl(self->vdin1_fd, VIDIOC_DQBUF, &v4l2_buf) < 0) {
        if (errno == EAGAIN)
            goto retry;
        GST_ERROR_OBJECT(self, "VIDIOC_DQBUF failed: %s", strerror(errno));
        return GST_FLOW_ERROR;
    }

    guint idx = v4l2_buf.index;
    guint32 bytesused = planes[0].bytesused;

    if (idx >= self->vdin1_n_bufs || !self->vdin1_bufs[idx].start) {
        GST_ERROR_OBJECT(self, "DQBUF returned invalid index %u", idx);
        return GST_FLOW_ERROR;
    }

    /*
     * Wrap frame data in a GstBuffer.
     * vdin1 uses MMAP, so we need to copy the data into a GstBuffer
     * since we must QBUF the V4L2 buffer back promptly.
     *
     * The vdin1 driver's sizeimage (and sometimes bytesused) can be
     * slightly larger than the actual NV21 frame data (W*H*1.5).
     * Amlogic's 00027-resize-gstbuffer-for-NV12-and-NV21.patch handles
     * this by trimming the buffer. We do the same: allocate and copy
     * only the actual NV21 data size.
     */
    guint32 actual_size = self->width * self->height * 3 / 2;
    if (bytesused > actual_size) {
        GST_LOG_OBJECT(self, "Trimming bytesused %u -> %u (NV21 W*H*1.5)",
                       bytesused, actual_size);
        bytesused = actual_size;
    }

    GstBuffer *buffer = gst_buffer_new_allocate(NULL, bytesused, NULL);
    if (!buffer) {
        GST_ERROR_OBJECT(self, "Failed to allocate GstBuffer (%u bytes)",
                         bytesused);
        goto qbuf_return;
    }

    /* Copy frame data */
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        memcpy(map.data, self->vdin1_bufs[idx].start, bytesused);
        gst_buffer_unmap(buffer, &map);
    } else {
        GST_ERROR_OBJECT(self, "Failed to map GstBuffer");
        gst_buffer_unref(buffer);
        buffer = NULL;
        goto qbuf_return;
    }

    /* Set timestamps */
    GST_BUFFER_PTS(buffer) = (guint64)v4l2_buf.timestamp.tv_sec * GST_SECOND +
                              (guint64)v4l2_buf.timestamp.tv_usec * GST_USECOND;
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

    self->frame_count++;

    if (self->frame_count == 1 || self->frame_count % 300 == 0) {
        GST_INFO_OBJECT(self, "Path B frame %lu: %ux%u NV21 (%u bytes)",
                         (unsigned long)self->frame_count,
                         self->width, self->height, bytesused);
    }

qbuf_return:
    /* QBUF -- return buffer to vdin1 */
    {
        struct v4l2_buffer qbuf;
        struct v4l2_plane qplanes[1];
        memset(&qbuf, 0, sizeof(qbuf));
        memset(&qplanes, 0, sizeof(qplanes));
        qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        qbuf.memory = V4L2_MEMORY_MMAP;
        qbuf.index = idx;
        qbuf.m.planes = qplanes;
        qbuf.length = 1;

        if (xioctl(self->vdin1_fd, VIDIOC_QBUF, &qbuf) < 0) {
            GST_ERROR_OBJECT(self, "VIDIOC_QBUF(%u) failed: %s",
                             idx, strerror(errno));
        }
    }

    if (!buffer)
        return GST_FLOW_ERROR;

    *buf = buffer;
    return GST_FLOW_OK;
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
