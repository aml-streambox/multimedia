/*
 * gstvfmcapsrc.h - GStreamer vfmcap source element
 *
 * Zero-copy HDMI capture source using libvfmcap SDK.
 * Captures from /dev/video_cap (vfm_cap kernel module) and pushes
 * DMA-buf backed NV12 or P010 buffers downstream.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_VFMCAP_SRC_H__
#define __GST_VFMCAP_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <vfmcap.h>

G_BEGIN_DECLS

#define GST_TYPE_VFMCAP_SRC            (gst_vfmcap_src_get_type())
#define GST_VFMCAP_SRC(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_VFMCAP_SRC, GstVfmCapSrc))
#define GST_VFMCAP_SRC_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_VFMCAP_SRC, GstVfmCapSrcClass))
#define GST_IS_VFMCAP_SRC(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_VFMCAP_SRC))
#define GST_IS_VFMCAP_SRC_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_VFMCAP_SRC))

typedef struct _GstVfmCapSrc      GstVfmCapSrc;
typedef struct _GstVfmCapSrcClass GstVfmCapSrcClass;

typedef enum {
    GST_VFMCAP_OUTPUT_NV12 = 0,
    GST_VFMCAP_OUTPUT_P010 = 1,
} GstVfmCapOutputFormat;

struct _GstVfmCapSrc
{
    GstPushSrc parent;

    /* Properties */
    gchar    *device;             /* /dev/video_cap */
    guint     num_buffers;        /* V4L2 buffer count (default 6) */
    GstVfmCapOutputFormat output_fmt;  /* NV12 or P010 */

    /* Runtime state */
    vfmcap_ctx_t *cap_ctx;       /* libvfmcap context */
    gboolean  streaming;          /* TRUE while streaming */
    guint     width;
    guint     height;
    guint     fps_n;              /* framerate numerator */
    guint     fps_d;              /* framerate denominator */

    /* DMA-buf output pool */
    int       heap_fd;            /* /dev/dma_heap/system-uncached fd */
    guint32   out_buf_size;       /* Output buffer size for current fmt */

    /* Caps negotiation */
    gboolean  caps_set;           /* Whether caps have been pushed */

    /* Frame counter for logging */
    guint64   frame_count;
};

struct _GstVfmCapSrcClass
{
    GstPushSrcClass parent_class;
};

GType gst_vfmcap_src_get_type(void);

G_END_DECLS

#endif /* __GST_VFMCAP_SRC_H__ */
