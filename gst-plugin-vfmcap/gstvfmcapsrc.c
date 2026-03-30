/*
 * gstvfmcapsrc.c - GStreamer vfmcap source element
 *
 * Zero-copy HDMI capture source using libvfmcap SDK.
 * Captures from /dev/video_cap (vfm_cap kernel module), converts
 * AMLY 10-bit frames to NV12 or P010 via Vulkan GPU, and pushes
 * DMA-buf backed GstBuffers downstream.
 *
 * Pipeline examples:
 *   gst-launch-1.0 vfmcapsrc ! video/x-raw,format=NV12 ! amlvenc ! ...
 *   gst-launch-1.0 vfmcapsrc output-format=p010 ! video/x-raw,format=P010 ! ...
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
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>

#include "gstvfmcapsrc.h"

GST_DEBUG_CATEGORY_STATIC(gst_vfmcap_src_debug);
#define GST_CAT_DEFAULT gst_vfmcap_src_debug

/* ---------- Properties ---------- */

enum
{
    PROP_0,
    PROP_DEVICE,
    PROP_NUM_BUFFERS,
    PROP_OUTPUT_FORMAT,
};

#define DEFAULT_DEVICE       "/dev/video_cap"
#define DEFAULT_NUM_BUFFERS  6
#define DEFAULT_OUTPUT_FMT   GST_VFMCAP_OUTPUT_NV12

#define DMA_HEAP_PATH "/dev/dma_heap/system-uncached"

/* ---------- GType for output format property ---------- */

#define GST_TYPE_VFMCAP_OUTPUT_FORMAT (gst_vfmcap_output_format_get_type())

static GType
gst_vfmcap_output_format_get_type(void)
{
    static GType type = 0;
    if (g_once_init_enter(&type)) {
        static const GEnumValue values[] = {
            { GST_VFMCAP_OUTPUT_NV12, "NV12 (8-bit)", "nv12" },
            { GST_VFMCAP_OUTPUT_P010, "P010 (10-bit)", "p010" },
            { 0, NULL, NULL }
        };
        GType tmp = g_enum_register_static("GstVfmCapOutputFormat", values);
        g_once_init_leave(&type, tmp);
    }
    return type;
}

/* ---------- Pad template ---------- */

/* Source pad: we can output NV12 or P010 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, "
        "format = (string) { NV12, P010_10LE }, "
        "width = (int) [ 1, 4096 ], "
        "height = (int) [ 1, 2160 ], "
        "framerate = (fraction) [ 0/1, 240/1 ]"
    )
);

/* ---------- GObject boilerplate ---------- */

#define gst_vfmcap_src_parent_class parent_class
G_DEFINE_TYPE(GstVfmCapSrc, gst_vfmcap_src, GST_TYPE_PUSH_SRC);

/* ---------- Forward declarations ---------- */

static void     gst_vfmcap_src_set_property(GObject *object, guint prop_id,
                                             const GValue *value, GParamSpec *pspec);
static void     gst_vfmcap_src_get_property(GObject *object, guint prop_id,
                                             GValue *value, GParamSpec *pspec);
static void     gst_vfmcap_src_finalize(GObject *object);

static gboolean gst_vfmcap_src_start(GstBaseSrc *basesrc);
static gboolean gst_vfmcap_src_stop(GstBaseSrc *basesrc);
static gboolean gst_vfmcap_src_unlock(GstBaseSrc *basesrc);
static gboolean gst_vfmcap_src_unlock_stop(GstBaseSrc *basesrc);
static GstCaps *gst_vfmcap_src_get_caps(GstBaseSrc *basesrc, GstCaps *filter);
static GstCaps *gst_vfmcap_src_fixate(GstBaseSrc *basesrc, GstCaps *caps);
static GstFlowReturn gst_vfmcap_src_create(GstPushSrc *pushsrc, GstBuffer **buf);

/* ---------- DMA-heap allocation ---------- */

static int
alloc_output_dmabuf(GstVfmCapSrc *self, guint32 size)
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

/* ---------- Class init ---------- */

static void
gst_vfmcap_src_class_init(GstVfmCapSrcClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS(klass);
    GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = gst_vfmcap_src_set_property;
    gobject_class->get_property = gst_vfmcap_src_get_property;
    gobject_class->finalize = gst_vfmcap_src_finalize;

    g_object_class_install_property(gobject_class, PROP_DEVICE,
        g_param_spec_string("device", "Device",
            "V4L2 device path for vfm_cap",
            DEFAULT_DEVICE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_NUM_BUFFERS,
        g_param_spec_uint("num-buffers", "Num Buffers",
            "Number of V4L2 capture buffers",
            2, 16, DEFAULT_NUM_BUFFERS,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(gobject_class, PROP_OUTPUT_FORMAT,
        g_param_spec_enum("output-format", "Output Format",
            "Output pixel format (NV12 or P010)",
            GST_TYPE_VFMCAP_OUTPUT_FORMAT,
            DEFAULT_OUTPUT_FMT,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    gst_element_class_set_static_metadata(element_class,
        "VFM Capture Source",
        "Source/Video",
        "Zero-copy HDMI capture source with Vulkan 10-bit conversion",
        "StreamBox");

    gst_element_class_add_static_pad_template(element_class, &src_template);

    basesrc_class->start = GST_DEBUG_FUNCPTR(gst_vfmcap_src_start);
    basesrc_class->stop = GST_DEBUG_FUNCPTR(gst_vfmcap_src_stop);
    basesrc_class->unlock = GST_DEBUG_FUNCPTR(gst_vfmcap_src_unlock);
    basesrc_class->unlock_stop = GST_DEBUG_FUNCPTR(gst_vfmcap_src_unlock_stop);
    basesrc_class->get_caps = GST_DEBUG_FUNCPTR(gst_vfmcap_src_get_caps);
    basesrc_class->fixate = GST_DEBUG_FUNCPTR(gst_vfmcap_src_fixate);

    pushsrc_class->create = GST_DEBUG_FUNCPTR(gst_vfmcap_src_create);
}

/* ---------- Instance init ---------- */

static void
gst_vfmcap_src_init(GstVfmCapSrc *self)
{
    self->device = g_strdup(DEFAULT_DEVICE);
    self->num_buffers = DEFAULT_NUM_BUFFERS;
    self->output_fmt = DEFAULT_OUTPUT_FMT;
    self->cap_ctx = NULL;
    self->streaming = FALSE;
    self->heap_fd = -1;
    self->caps_set = FALSE;
    self->frame_count = 0;

    /* We are a live source */
    gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
    gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
    gst_base_src_set_do_timestamp(GST_BASE_SRC(self), TRUE);
}

/* ---------- Properties ---------- */

static void
gst_vfmcap_src_set_property(GObject *object, guint prop_id,
                              const GValue *value, GParamSpec *pspec)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(object);

    switch (prop_id) {
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
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_vfmcap_src_get_property(GObject *object, guint prop_id,
                              GValue *value, GParamSpec *pspec)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(object);

    switch (prop_id) {
    case PROP_DEVICE:
        g_value_set_string(value, self->device);
        break;
    case PROP_NUM_BUFFERS:
        g_value_set_uint(value, self->num_buffers);
        break;
    case PROP_OUTPUT_FORMAT:
        g_value_set_enum(value, self->output_fmt);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_vfmcap_src_finalize(GObject *object)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(object);
    g_free(self->device);
    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* ---------- Start / Stop ---------- */

static gboolean
gst_vfmcap_src_start(GstBaseSrc *basesrc)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(basesrc);

    GST_INFO_OBJECT(self, "Starting: device=%s buffers=%u fmt=%s",
                     self->device, self->num_buffers,
                     self->output_fmt == GST_VFMCAP_OUTPUT_P010 ? "P010" : "NV12");

    /* Open the capture device */
    self->cap_ctx = vfmcap_open(self->device);
    if (!self->cap_ctx) {
        GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                          ("Failed to open %s", self->device),
                          ("%s", vfmcap_last_error(NULL)));
        return FALSE;
    }

    /* Start streaming */
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
     * Get resolution by acquiring a test frame.
     * The V4L2 G_FMT returns a default (1920x1080 NV21) until the first
     * vframe arrives from vdin0, so we must wait for a real frame to get
     * the actual signal resolution.
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

    /* Calculate output buffer size */
    vfmcap_output_fmt_t sdk_fmt = (self->output_fmt == GST_VFMCAP_OUTPUT_P010)
                                   ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
    self->out_buf_size = vfmcap_output_size(self->width, self->height, sdk_fmt);

    self->streaming = TRUE;
    self->caps_set = FALSE;
    self->frame_count = 0;

    GST_INFO_OBJECT(self, "Started: %ux%u @ %u/%u fps, output=%s (%u bytes)",
                     self->width, self->height, self->fps_n, self->fps_d,
                     self->output_fmt == GST_VFMCAP_OUTPUT_P010 ? "P010" : "NV12",
                     self->out_buf_size);

    return TRUE;
}

static gboolean
gst_vfmcap_src_stop(GstBaseSrc *basesrc)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(basesrc);

    GST_INFO_OBJECT(self, "Stopping: %lu frames captured",
                     (unsigned long)self->frame_count);

    self->streaming = FALSE;

    if (self->cap_ctx) {
        vfmcap_stop(self->cap_ctx);
        vfmcap_close(self->cap_ctx);
        self->cap_ctx = NULL;
    }

    if (self->heap_fd >= 0) {
        close(self->heap_fd);
        self->heap_fd = -1;
    }

    self->caps_set = FALSE;

    return TRUE;
}

static gboolean
gst_vfmcap_src_unlock(GstBaseSrc *basesrc)
{
    /* TODO: could signal a condition to unblock acquire_frame */
    (void)basesrc;
    return TRUE;
}

static gboolean
gst_vfmcap_src_unlock_stop(GstBaseSrc *basesrc)
{
    (void)basesrc;
    return TRUE;
}

/* ---------- Caps negotiation ---------- */

static GstCaps *
gst_vfmcap_src_get_caps(GstBaseSrc *basesrc, GstCaps *filter)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(basesrc);
    GstCaps *caps;

    if (self->streaming && self->width > 0 && self->height > 0) {
        const gchar *fmt_str;
        if (self->output_fmt == GST_VFMCAP_OUTPUT_P010) {
            fmt_str = "P010_10LE";
        } else {
            fmt_str = "NV12";
        }

        caps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, fmt_str,
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
gst_vfmcap_src_fixate(GstBaseSrc *basesrc, GstCaps *caps)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(basesrc);
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

/* ---------- Frame creation ---------- */

static GstFlowReturn
gst_vfmcap_src_create(GstPushSrc *pushsrc, GstBuffer **buf)
{
    GstVfmCapSrc *self = GST_VFMCAP_SRC(pushsrc);

    if (!self->streaming || !self->cap_ctx) {
        return GST_FLOW_ERROR;
    }

    /* Push caps downstream on first frame if not yet done */
    if (!self->caps_set) {
        const gchar *fmt_str = (self->output_fmt == GST_VFMCAP_OUTPUT_P010)
                                ? "P010_10LE" : "NV12";
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

    /* Acquire a frame from the capture device */
    vfmcap_frame_t frame;
    int ret = vfmcap_acquire_frame(self->cap_ctx, &frame, 1000);

    if (ret == VFMCAP_ERR_TIMEOUT) {
        GST_WARNING_OBJECT(self, "Frame acquire timeout (frame %lu)",
                           (unsigned long)self->frame_count);
        /* Return EOS or try again -- for live source, just try again
         * by returning FLOW_OK with NULL buffer is not valid;
         * we'll let the framework retry */
        return GST_FLOW_ERROR;
    }

    if (ret == VFMCAP_ERR_NOSIG) {
        GST_WARNING_OBJECT(self, "No signal");
        /* EOS on signal loss */
        return GST_FLOW_EOS;
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

        vfmcap_output_fmt_t sdk_fmt = (self->output_fmt == GST_VFMCAP_OUTPUT_P010)
                                       ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
        self->out_buf_size = vfmcap_output_size(self->width, self->height, sdk_fmt);
        self->caps_set = FALSE;

        /* Drop this frame and let the next create() call push new caps */
        vfmcap_release_frame(self->cap_ctx, &frame);

        /* Signal EOS so the pipeline can be restarted with new caps.
         * For a live HDMI source, the pipeline manager should handle
         * reconnection on resolution change. */
        GST_WARNING_OBJECT(self, "Resolution change detected, sending EOS");
        return GST_FLOW_EOS;
    }

    /* Allocate output DMA-buf */
    int out_fd = alloc_output_dmabuf(self, self->out_buf_size);
    if (out_fd < 0) {
        GST_ERROR_OBJECT(self, "Failed to allocate output DMA-buf");
        vfmcap_release_frame(self->cap_ctx, &frame);
        return GST_FLOW_ERROR;
    }

    /* GPU convert: AMLY -> NV12 or P010 */
    vfmcap_output_fmt_t sdk_fmt = (self->output_fmt == GST_VFMCAP_OUTPUT_P010)
                                   ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
    if (sdk_fmt == VFMCAP_FMT_P010) {
        ret = vfmcap_convert_p010(self->cap_ctx, &frame, out_fd);
    } else {
        ret = vfmcap_convert_nv12(self->cap_ctx, &frame, out_fd);
    }

    /* Release the input frame back to vdin0 immediately after GPU is done */
    vfmcap_release_frame(self->cap_ctx, &frame);

    if (ret != VFMCAP_OK) {
        GST_ERROR_OBJECT(self, "GPU conversion failed: %s",
                         vfmcap_last_error(self->cap_ctx));
        close(out_fd);
        return GST_FLOW_ERROR;
    }

    /* Wrap the output DMA-buf fd in a GstBuffer using GstDmaBufAllocator */
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

    /* Set buffer metadata */
    GST_BUFFER_PTS(buffer) = frame.timestamp_us * GST_USECOND;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(GST_SECOND,
                                                             self->fps_d,
                                                             self->fps_n);

    /* Add video meta */
    GstVideoFormat gst_fmt = (self->output_fmt == GST_VFMCAP_OUTPUT_P010)
                              ? GST_VIDEO_FORMAT_P010_10LE
                              : GST_VIDEO_FORMAT_NV12;
    gsize y_plane_size, uv_plane_size;
    gsize offsets[2];
    gint strides[2];

    if (self->output_fmt == GST_VFMCAP_OUTPUT_P010) {
        strides[0] = self->width * 2;
        strides[1] = self->width * 2;
        y_plane_size = (gsize)self->width * self->height * 2;
        uv_plane_size = (gsize)self->width * self->height;
    } else {
        strides[0] = self->width;
        strides[1] = self->width;
        y_plane_size = (gsize)self->width * self->height;
        uv_plane_size = (gsize)self->width * self->height / 2;
    }
    offsets[0] = 0;
    offsets[1] = y_plane_size;

    gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
                                    gst_fmt, self->width, self->height,
                                    2, offsets, strides);

    self->frame_count++;

    if (self->frame_count == 1 || self->frame_count % 300 == 0) {
        GST_INFO_OBJECT(self, "Frame %lu: %ux%u %s",
                         (unsigned long)self->frame_count,
                         self->width, self->height,
                         self->output_fmt == GST_VFMCAP_OUTPUT_P010 ? "P010" : "NV12");
    }

    *buf = buffer;
    return GST_FLOW_OK;
}

/* ---------- Plugin registration ---------- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_vfmcap_src_debug, "vfmcapsrc", 0,
                            "VFM capture source");

    return gst_element_register(plugin, "vfmcapsrc", GST_RANK_PRIMARY,
                                GST_TYPE_VFMCAP_SRC);
}

#ifndef VERSION
#define VERSION "1.0.0"
#endif

#ifndef PACKAGE
#define PACKAGE "gst-plugin-vfmcap"
#endif

#ifndef PACKAGE_NAME
#define PACKAGE_NAME "gst-plugin-vfmcap"
#endif

#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "https://streambox.ai/"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    vfmcapsrc,
    "Zero-copy HDMI capture source with Vulkan 10-bit conversion",
    plugin_init,
    VERSION,
    "LGPL",
    PACKAGE_NAME,
    GST_PACKAGE_ORIGIN
)
