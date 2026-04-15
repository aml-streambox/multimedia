/*
 * hdr_lut3d.h - 3D LUT generation for HDR-to-SDR color conversion
 *
 * Generates 33x33x33 RGBA16_UNORM lookup tables that map BT.2020 YCbCr
 * input through the full HDR-to-SDR pipeline to BT.709 YCbCr output.
 * Used by the Vulkan fragment shaders as a sampler3D texture.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#ifndef HDR_LUT3D_H
#define HDR_LUT3D_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LUT dimensions: 33 nodes per axis (32 intervals) */
#define LUT3D_SIZE      33
#define LUT3D_ENTRIES   (LUT3D_SIZE * LUT3D_SIZE * LUT3D_SIZE)  /* 35937 */
#define LUT3D_CHANNELS  4   /* RGBA16 */
#define LUT3D_BYTES     (LUT3D_ENTRIES * LUT3D_CHANNELS * sizeof(uint16_t))  /* 287496 */

/**
 * hdr_lut3d_generate - Generate a 3D LUT for HDR-to-SDR conversion
 * @out: Output buffer, must be at least LUT3D_BYTES (287,496 bytes)
 *       Layout: RGBA16 texels in X-fastest order (Y_in varies fastest,
 *       then Cb_in, then Cr_in) for direct upload to VkImage 3D.
 * @color_mode: 1 = HDR10 (PQ+BT.2020 -> SDR BT.709)
 *              2 = HLG (HLG+BT.2020 -> SDR BT.709)
 *
 * Each output texel: R=Y_out, G=Cb_out, B=Cr_out, A=0xFFFF
 * All values are UNORM [0, 65535] mapping to [0.0, 1.0]
 *
 * Returns 0 on success, -1 on invalid color_mode.
 */
int hdr_lut3d_generate(uint16_t *out, uint32_t color_mode);

/**
 * hdr_lut3d_sanity_check - Quick validation of LUT output
 * @lut: Generated LUT data
 * @color_mode: Mode used to generate
 *
 * Checks a few known points (black, mid-gray, peak white).
 * Prints results to stderr. For debug/development use.
 */
void hdr_lut3d_sanity_check(const uint16_t *lut, uint32_t color_mode);

#ifdef __cplusplus
}
#endif

#endif /* HDR_LUT3D_H */
