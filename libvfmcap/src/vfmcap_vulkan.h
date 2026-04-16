/*
 * vfmcap_vulkan.h - Internal Vulkan converter header
 *
 * Provides GPU-based format conversion for vfmcap using Vulkan
 * graphics and compute pipelines. Per-instance context.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#ifndef VFMCAP_VULKAN_H
#define VFMCAP_VULKAN_H

#include <stdint.h>
#include "../include/vfmcap.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration of opaque Vulkan context */
typedef struct VulkanCtx VulkanCtx;

/* Output format selection (internal, maps from vfmcap_output_fmt_t) */
typedef enum {
    VFMCAP_VK_FMT_P010 = 0,
    VFMCAP_VK_FMT_NV12 = 1,
    VFMCAP_VK_FMT_NV21 = 2,
    VFMCAP_VK_FMT_NV12_AFBC = 3,
    VFMCAP_VK_FMT_A2B10G10R10_AFBC = 4,
} vfmcap_vk_fmt_t;

/**
 * vfmcap_vk_init - Initialize Vulkan pipeline for a context
 * @vk: Pointer to store the allocated VulkanCtx
 * @width: Initial frame width
 * @height: Initial frame height
 * @fmt: Output format (determines which pipelines to create)
 * @color_mode: HDR/color conversion mode (0=passthrough, 1=HDR10->SDR, 2=HLG->SDR)
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_init(VulkanCtx **vk, uint32_t width, uint32_t height,
                    vfmcap_vk_fmt_t fmt, uint32_t color_mode);

/**
 * vfmcap_vk_convert_submit - Submit async GPU conversion
 * @vk: Vulkan context
 * @in_fd: Input DMA-buf fd
 * @out_fd: Output DMA-buf fd (legacy path, caller-allocated)
 * @width: Frame width
 * @height: Frame height
 * @fmt: Output format
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_convert_submit(VulkanCtx *vk, int in_fd, int out_fd, uint32_t width,
                             uint32_t height, vfmcap_vk_fmt_t fmt);

/**
 * vfmcap_vk_convert_wait - Wait for GPU work to complete
 * @vk: Vulkan context
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_convert_wait(VulkanCtx *vk);

/**
 * vfmcap_vk_convert - Synchronous conversion (submit + wait)
 * @vk: Vulkan context
 */
int vfmcap_vk_convert(VulkanCtx *vk, int in_fd, int out_fd, uint32_t width,
                      uint32_t height, vfmcap_vk_fmt_t fmt);

/**
 * vfmcap_vk_render_submit - Submit async GPU graphics rendering
 * @vk: Vulkan context
 * @in_fd: Input DMA-buf fd (NV12/NV21)
 * @out_y_fd: Output DMA-buf fd for Y plane
 * @out_uv_fd: Output DMA-buf fd for UV plane
 * @src_width: Source frame width
 * @src_height: Source frame height
 * @dst_width: Output frame width
 * @dst_height: Output frame height
 * @fmt: Output format (NV12, NV21, or P010)
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_render_submit(VulkanCtx *vk, int in_fd,
                            int out_y_fd, int out_uv_fd,
                            uint32_t src_width, uint32_t src_height,
                            uint32_t dst_width, uint32_t dst_height,
                            vfmcap_vk_fmt_t fmt);

/**
 * vfmcap_vk_render_and_wait - Acquire output pool, render, and wait
 * @vk: Vulkan context
 * @in_fd: Input DMA-buf fd
 * @src_width: Source frame width
 * @src_height: Source frame height
 * @dst_width: Output frame width
 * @dst_height: Output frame height
 * @fmt: Output format
 * @out_y_fd: Output Y plane DMA-buf fd
 * @out_uv_fd: Output UV plane DMA-buf fd
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_render_and_wait(VulkanCtx *vk, int in_fd,
                              uint32_t src_width, uint32_t src_height,
                              uint32_t dst_width, uint32_t dst_height,
                              vfmcap_vk_fmt_t fmt,
                              int *out_y_fd, int *out_uv_fd);

/**
 * vfmcap_vk_release_output - Release output pool entries back to pool
 * @vk: Vulkan context
 * @y_fd: Y plane DMA-buf fd
 * @uv_fd: UV plane DMA-buf fd
 * @fmt: Output format
 */
void vfmcap_vk_release_output(VulkanCtx *vk, int y_fd, int uv_fd, vfmcap_vk_fmt_t fmt);

/**
 * vfmcap_vk_render_10bit_and_wait - 10-bit AMLY -> R8G8B8A8 -> NV12/P010
 * @vk: Vulkan context
 * @in_fd: Input AMLY DMA-buf fd
 * @src_width: Source frame width
 * @src_height: Source frame height
 * @dst_width: Output frame width
 * @dst_height: Output frame height
 * @fmt: Output format (NV12 or P010)
 * @out_y_fd: Output Y plane DMA-buf fd
 * @out_uv_fd: Output UV plane DMA-buf fd
 *
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_render_10bit_and_wait(VulkanCtx *vk, int in_fd,
                                    uint32_t src_width, uint32_t src_height,
                                    uint32_t dst_width, uint32_t dst_height,
                                    vfmcap_vk_fmt_t fmt,
                                    int *out_y_fd, int *out_uv_fd);

/**
 * vfmcap_vk_render_predecode_and_wait - Strategy A: compute pre-decode + fragment tonemap
 * @vk: Vulkan context
 * @in_fd: Input AMLY DMA-buf fd
 * @src_width: Source frame width
 * @src_height: Source frame height
 * @dst_width: Output frame width
 * @dst_height: Output frame height
 * @fmt: Output format (NV12 or P010)
 * @out_y_fd: Output Y plane DMA-buf fd
 * @out_uv_fd: Output UV plane DMA-buf fd
 *
 * Hybrid pipeline: compute shader unpacks AMLY to intermediate textures,
 * then fragment shaders do texture fetch + 3D LUT tone mapping.
 * Returns 0 on success, -1 on failure.
 */
int vfmcap_vk_render_predecode_and_wait(VulkanCtx *vk, int in_fd,
                                         uint32_t src_width, uint32_t src_height,
                                         uint32_t dst_width, uint32_t dst_height,
                                         vfmcap_vk_fmt_t fmt,
                                         int *out_y_fd, int *out_uv_fd);

/**
 * vfmcap_vk_cleanup - Destroy all Vulkan resources
 * @vk: Vulkan context
 */
void vfmcap_vk_cleanup(VulkanCtx *vk);

/**
 * vfmcap_vk_last_error - Get last Vulkan error message
 * @vk: Vulkan context (may be NULL)
 */
const char *vfmcap_vk_last_error(VulkanCtx *vk);

#ifdef __cplusplus
}
#endif

#endif /* VFMCAP_VULKAN_H */
