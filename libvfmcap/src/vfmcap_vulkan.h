/*
 * vfmcap_vulkan.h - Internal Vulkan converter header
 *
 * Ported from amlvenc yuv422_converter_vulkan.c/h.
 * Provides AMLY -> P010 and AMLY -> NV12 compute shader conversion.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#ifndef VFMCAP_VULKAN_H
#define VFMCAP_VULKAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Output format selection */
typedef enum {
    VFMCAP_VK_FMT_P010 = 0,  /* AMLY -> P010 (10-bit, val << 6) */
    VFMCAP_VK_FMT_NV12 = 1,  /* AMLY -> NV12 (8-bit, val >> 2) */
} vfmcap_vk_fmt_t;

/**
 * vfmcap_vk_init - Initialize Vulkan compute pipeline
 * @width: Initial frame width
 * @height: Initial frame height
 *
 * Creates Vulkan instance, device, command pool, descriptor sets,
 * and loads both P010 and NV12 compute shaders.
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_init(uint32_t width, uint32_t height);

/**
 * vfmcap_vk_convert_submit - Submit async GPU conversion
 * @in_fd: Input DMA-buf fd (AMLY 40-bit packed)
 * @out_fd: Output DMA-buf fd (P010 or NV12, caller-allocated)
 * @width: Frame width
 * @height: Frame height
 * @fmt: Output format
 * @hdr_mode: 0 = SDR passthrough, 1 = HDR BT.2020 PQ -> SDR BT.709
 *            (only affects NV12 pipeline; ignored for P010)
 *
 * Dispatches compute shader and returns immediately.
 * Must call vfmcap_vk_convert_wait() before next submit.
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_convert_submit(int in_fd, int out_fd, uint32_t width,
                             uint32_t height, vfmcap_vk_fmt_t fmt,
                             uint32_t hdr_mode);

/**
 * vfmcap_vk_convert_wait - Wait for GPU work to complete
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_convert_wait(void);

/**
 * vfmcap_vk_convert - Synchronous conversion (submit + wait)
 */
int vfmcap_vk_convert(int in_fd, int out_fd, uint32_t width,
                      uint32_t height, vfmcap_vk_fmt_t fmt,
                      uint32_t hdr_mode);

/**
 * vfmcap_vk_cleanup - Destroy all Vulkan resources
 */
void vfmcap_vk_cleanup(void);

/**
 * vfmcap_vk_last_error - Get last Vulkan error message
 */
const char *vfmcap_vk_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* VFMCAP_VULKAN_H */
