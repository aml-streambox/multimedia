/*
 * hdr_lut3d.c - 3D LUT generation for HDR-to-SDR color conversion
 *
 * CPU-side generation of 33x33x33 RGBA16_UNORM lookup tables.
 * Replicates the exact same math pipeline as hdr_colorconv.glsl
 * but runs on the CPU once at init time.
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#include "hdr_lut3d.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- BT.2020 limited-range YCbCr -> RGB ---- */

#define LR_Y_SCALE   (1023.0 / 876.0)
#define LR_Y_OFFSET  (64.0 / 1023.0)
#define LR_C_SCALE   (1023.0 / 896.0)
#define LR_C_OFFSET  (512.0 / 1023.0)

static void ycbcr_to_rgb_bt2020(double y_norm, double cb_norm, double cr_norm,
                                 double *r, double *g, double *b)
{
    double yp = (y_norm - LR_Y_OFFSET) * LR_Y_SCALE;
    double cb = (cb_norm - LR_C_OFFSET) * LR_C_SCALE;
    double cr = (cr_norm - LR_C_OFFSET) * LR_C_SCALE;

    *r = yp + 1.4746 * cr;
    *g = yp - 0.1646 * cb - 0.5714 * cr;
    *b = yp + 1.8814 * cb;

    if (*r < 0.0) *r = 0.0; if (*r > 1.0) *r = 1.0;
    if (*g < 0.0) *g = 0.0; if (*g > 1.0) *g = 1.0;
    if (*b < 0.0) *b = 0.0; if (*b > 1.0) *b = 1.0;
}

/* ---- PQ (ST 2084) EOTF ---- */

#define PQ_M1  0.1593017578125
#define PQ_M2  78.84375
#define PQ_C1  0.8359375
#define PQ_C2  18.8515625
#define PQ_C3  18.6875

static double pq_eotf(double e)
{
    if (e <= 0.0) return 0.0;
    double ep = pow(e, 1.0 / PQ_M2);
    double num = ep - PQ_C1;
    if (num < 0.0) num = 0.0;
    double den = PQ_C2 - PQ_C3 * ep;
    if (den <= 0.0) return 0.0;
    return pow(num / den, 1.0 / PQ_M1);
}

/* ---- HLG inverse OETF + OOTF ---- */

#define HLG_A  0.17883277
#define HLG_B  0.28466892
#define HLG_C  0.55991073

static double hlg_inv_oetf(double e)
{
    if (e <= 0.5)
        return (e * e) / 3.0;
    else
        return (exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0;
}

static void hlg_to_linear(double r_in, double g_in, double b_in,
                           double *r_out, double *g_out, double *b_out)
{
    double sr = hlg_inv_oetf(r_in);
    double sg = hlg_inv_oetf(g_in);
    double sb = hlg_inv_oetf(b_in);

    /* BT.2020 luminance coefficients */
    double y_scene = 0.2627 * sr + 0.6780 * sg + 0.0593 * sb;
    double boost = pow(y_scene > 1e-6 ? y_scene : 1e-6, 0.2);

    *r_out = sr * boost;
    *g_out = sg * boost;
    *b_out = sb * boost;
}

/* ---- BT.2020 -> BT.709 gamut mapping ---- */

/* Direct BT.2020 -> BT.709 matrix (column-major in GLSL, row-major here) */
static void gamut_bt2020_to_bt709(double r_in, double g_in, double b_in,
                                   double *r_out, double *g_out, double *b_out)
{
    /* Row-major multiplication matching GLSL column-major mat3:
     * GLSL: mat3(col0, col1, col2) * vec3 = vec3(dot(row0,v), dot(row1,v), dot(row2,v))
     * where row0 = (col0.x, col1.x, col2.x), etc.
     *
     * From the GLSL: mat3(1.6605, -0.1246, -0.0182,
     *                     -0.5876,  1.1329, -0.1006,
     *                     -0.0728, -0.0083,  1.1187)
     * This is 3 columns: col0=(1.6605, -0.1246, -0.0182),
     *                     col1=(-0.5876, 1.1329, -0.1006),
     *                     col2=(-0.0728, -0.0083, 1.1187)
     * Row0 = (1.6605, -0.5876, -0.0728)
     * Row1 = (-0.1246, 1.1329, -0.0083)
     * Row2 = (-0.0182, -0.1006, 1.1187)
     */
    double r =  1.6605 * r_in - 0.5876 * g_in - 0.0728 * b_in;
    double g = -0.1246 * r_in + 1.1329 * g_in - 0.0083 * b_in;
    double b = -0.0182 * r_in - 0.1006 * g_in + 1.1187 * b_in;

    /* Soft-clip desaturation (same as GLSL) */
    double max_c = r; if (g > max_c) max_c = g; if (b > max_c) max_c = b;
    double min_c = r; if (g < min_c) min_c = g; if (b < min_c) min_c = b;

    if (min_c < 0.0 || max_c > 1.0) {
        double lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        double t = 1.0;
        if (min_c < 0.0) {
            double t2 = lum / (lum - min_c);
            if (t2 < t) t = t2;
        }
        if (max_c > 1.0) {
            double t2 = (1.0 - lum) / (max_c - lum + 1e-6);
            if (t2 < t) t = t2;
        }
        if (t < 0.0) t = 0.0;
        r = lum + t * (r - lum);
        g = lum + t * (g - lum);
        b = lum + t * (b - lum);
    }

    if (r < 0.0) r = 0.0; if (r > 1.0) r = 1.0;
    if (g < 0.0) g = 0.0; if (g > 1.0) g = 1.0;
    if (b < 0.0) b = 0.0; if (b > 1.0) b = 1.0;

    *r_out = r;
    *g_out = g;
    *b_out = b;
}

/* ---- Reinhard extended tone mapping (luminance-based) ----
 *
 * PQ EOTF produces linear light in [0,1] normalized to 10000 nits.
 * Typical HDR10 content peaks around 1000 nits (= 0.1 PQ linear).
 * SDR reference white is ~100 nits (= 0.01 PQ linear).
 *
 * We first scale PQ linear values so SDR reference white (100 nits) = 1.0,
 * then use Reinhard-extended with a white point to roll off highlights.
 *
 * Scale factor: 10000 / SDR_REF_NITS = 100 (so 100 nits → 1.0)
 * White point Lw: controls highlight rolloff.  Lw = PEAK_NITS / SDR_REF_NITS.
 *   For 1000-nit content: Lw = 10.  Content up to 1000 nits is preserved.
 *
 * Reinhard extended: L' = L * (1 + L/Lw²) / (1 + L)
 *   At L=1 (100 nits): L' = (1 + 1/100) / 2 ≈ 0.505  (SDR mid-gray)
 *   At L=10 (1000 nits): L' = 10 * (1 + 10/100) / 11 ≈ 1.0 (SDR peak)
 *   At L=0.1 (10 nits): L' ≈ 0.091 (dim but visible)
 */
#define HDR_SDR_REF_NITS   100.0
#define HDR_EXPOSURE       (10000.0 / HDR_SDR_REF_NITS)   /* = 100 */
#define HDR_PEAK_NITS      1000.0
#define HDR_LW             (HDR_PEAK_NITS / HDR_SDR_REF_NITS)  /* = 10 */

static void tonemap_reinhard_extended(double r_in, double g_in, double b_in,
                                       double *r_out, double *g_out, double *b_out)
{
    /* Scale to SDR-relative: 100 nits = 1.0 */
    double rs = r_in * HDR_EXPOSURE;
    double gs = g_in * HDR_EXPOSURE;
    double bs = b_in * HDR_EXPOSURE;

    double lum = 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    if (lum < 1e-6) {
        *r_out = *g_out = *b_out = 0.0;
        return;
    }

    double lw2 = HDR_LW * HDR_LW;
    double lum_mapped = lum * (1.0 + lum / lw2) / (1.0 + lum);
    double scale = lum_mapped / lum;
    *r_out = rs * scale;
    *g_out = gs * scale;
    *b_out = bs * scale;

    /* Clamp to [0,1] (Reinhard-extended may slightly exceed 1.0) */
    if (*r_out > 1.0) *r_out = 1.0;
    if (*g_out > 1.0) *g_out = 1.0;
    if (*b_out > 1.0) *b_out = 1.0;
}

/* ---- BT.709 OETF ---- */

static double bt709_oetf(double l)
{
    if (l < 0.018)
        return 4.5 * l;
    else
        return 1.099 * pow(l, 0.45) - 0.099;
}

/* ---- BT.709 RGB -> YCbCr (limited range) ---- */

#define OUT_Y_SCALE   (876.0 / 1023.0)
#define OUT_Y_OFFSET  (64.0 / 1023.0)
#define OUT_C_SCALE   (896.0 / 1023.0)
#define OUT_C_OFFSET  (512.0 / 1023.0)

static void rgb_to_ycbcr_bt709(double r, double g, double b,
                                double *y_out, double *cb_out, double *cr_out)
{
    double y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    double cb = (b - y) / 1.8556;
    double cr = (r - y) / 1.5748;

    *y_out  = y * OUT_Y_SCALE + OUT_Y_OFFSET;
    *cb_out = cb * OUT_C_SCALE + OUT_C_OFFSET;
    *cr_out = cr * OUT_C_SCALE + OUT_C_OFFSET;

    if (*y_out < 0.0) *y_out = 0.0;   if (*y_out > 1.0) *y_out = 1.0;
    if (*cb_out < 0.0) *cb_out = 0.0;  if (*cb_out > 1.0) *cb_out = 1.0;
    if (*cr_out < 0.0) *cr_out = 0.0;  if (*cr_out > 1.0) *cr_out = 1.0;
}

/* ---- Full pipeline: input BT.2020 YCbCr (normalized) -> output BT.709 YCbCr ---- */

static void hdr10_pipeline(double y_in, double cb_in, double cr_in,
                            double *y_out, double *cb_out, double *cr_out)
{
    double r, g, b;

    /* 1. BT.2020 YCbCr -> RGB (non-linear) */
    ycbcr_to_rgb_bt2020(y_in, cb_in, cr_in, &r, &g, &b);

    /* 2. PQ EOTF -> linear light */
    r = pq_eotf(r);
    g = pq_eotf(g);
    b = pq_eotf(b);

    /* 3. BT.2020 -> BT.709 gamut map */
    gamut_bt2020_to_bt709(r, g, b, &r, &g, &b);

    /* 4. Reinhard tone map */
    tonemap_reinhard_extended(r, g, b, &r, &g, &b);

    /* 5. BT.709 OETF */
    r = bt709_oetf(r);
    g = bt709_oetf(g);
    b = bt709_oetf(b);

    /* 6. RGB -> BT.709 YCbCr (limited range) */
    rgb_to_ycbcr_bt709(r, g, b, y_out, cb_out, cr_out);
}

/* ---- HLG tone mapping ----
 *
 * HLG inverse OETF + OOTF produces scene-referred linear light in [0, ~1.2]
 * where 0.75 nominal peak = ~1.0 after OOTF boost.
 * Unlike PQ, HLG linear output is already in a reasonable range.
 * Use moderate Reinhard-extended with exposure=3.0, Lw=4.0.
 */
#define HLG_EXPOSURE   3.0
#define HLG_LW         4.0

static void tonemap_reinhard_hlg(double r_in, double g_in, double b_in,
                                  double *r_out, double *g_out, double *b_out)
{
    double rs = r_in * HLG_EXPOSURE;
    double gs = g_in * HLG_EXPOSURE;
    double bs = b_in * HLG_EXPOSURE;

    double lum = 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    if (lum < 1e-6) {
        *r_out = *g_out = *b_out = 0.0;
        return;
    }

    double lw2 = HLG_LW * HLG_LW;
    double lum_mapped = lum * (1.0 + lum / lw2) / (1.0 + lum);
    double scale = lum_mapped / lum;
    *r_out = rs * scale;
    *g_out = gs * scale;
    *b_out = bs * scale;

    if (*r_out > 1.0) *r_out = 1.0;
    if (*g_out > 1.0) *g_out = 1.0;
    if (*b_out > 1.0) *b_out = 1.0;
}

static void hlg_pipeline(double y_in, double cb_in, double cr_in,
                          double *y_out, double *cb_out, double *cr_out)
{
    double r, g, b;

    /* 1. BT.2020 YCbCr -> RGB (non-linear) */
    ycbcr_to_rgb_bt2020(y_in, cb_in, cr_in, &r, &g, &b);

    /* 2. HLG inverse OETF + OOTF -> linear light */
    hlg_to_linear(r, g, b, &r, &g, &b);

    /* 3. BT.2020 -> BT.709 gamut map */
    gamut_bt2020_to_bt709(r, g, b, &r, &g, &b);

    /* 4. HLG-specific Reinhard tone map (exposure=3x, Lw=4) */
    tonemap_reinhard_hlg(r, g, b, &r, &g, &b);

    /* 5. BT.709 OETF */
    r = bt709_oetf(r);
    g = bt709_oetf(g);
    b = bt709_oetf(b);

    /* 6. RGB -> BT.709 YCbCr (limited range) */
    rgb_to_ycbcr_bt709(r, g, b, y_out, cb_out, cr_out);
}

/* ---- Public API ---- */

int hdr_lut3d_generate(uint16_t *out, uint32_t color_mode)
{
    if (color_mode != 1 && color_mode != 2)
        return -1;

    void (*pipeline)(double, double, double, double*, double*, double*);
    pipeline = (color_mode == 1) ? hdr10_pipeline : hlg_pipeline;

    /*
     * LUT layout: X=Y_input, Y=Cb_input, Z=Cr_input (all [0,1])
     * VkImage 3D ordering: X varies fastest (width), then Y, then Z.
     * texel at (ix, iy, iz) = out[(iz * LUT3D_SIZE * LUT3D_SIZE + iy * LUT3D_SIZE + ix) * 4]
     */
    for (int iz = 0; iz < LUT3D_SIZE; iz++) {
        double cr_in = (double)iz / (double)(LUT3D_SIZE - 1);
        for (int iy = 0; iy < LUT3D_SIZE; iy++) {
            double cb_in = (double)iy / (double)(LUT3D_SIZE - 1);
            for (int ix = 0; ix < LUT3D_SIZE; ix++) {
                double y_in = (double)ix / (double)(LUT3D_SIZE - 1);

                double y_out, cb_out, cr_out;
                pipeline(y_in, cb_in, cr_in, &y_out, &cb_out, &cr_out);

                int idx = (iz * LUT3D_SIZE * LUT3D_SIZE + iy * LUT3D_SIZE + ix) * 4;
                out[idx + 0] = (uint16_t)(y_out * 65535.0 + 0.5);   /* R = Y */
                out[idx + 1] = (uint16_t)(cb_out * 65535.0 + 0.5);  /* G = Cb */
                out[idx + 2] = (uint16_t)(cr_out * 65535.0 + 0.5);  /* B = Cr */
                out[idx + 3] = 0xFFFF;                                /* A = 1.0 */
            }
        }
    }

    return 0;
}

void hdr_lut3d_sanity_check(const uint16_t *lut, uint32_t color_mode)
{
    const char *mode_name = (color_mode == 1) ? "HDR10" : "HLG";

    /* Helper: sample LUT at exact node */
    #define SAMPLE(ix, iy, iz) \
        (lut + ((iz) * LUT3D_SIZE * LUT3D_SIZE + (iy) * LUT3D_SIZE + (ix)) * 4)

    /* Black: Y=64/1023 ≈ idx 2, Cb=Cr=512/1023 ≈ idx 16 */
    int bk_ix = (int)(64.0 / 1023.0 * 32.0 + 0.5);
    int bk_iy = (int)(512.0 / 1023.0 * 32.0 + 0.5);
    int bk_iz = bk_iy;
    const uint16_t *bk = SAMPLE(bk_ix, bk_iy, bk_iz);
    fprintf(stderr, "[LUT3D %s] Black (idx %d,%d,%d): Y=%u/65535 Cb=%u Cr=%u\n",
            mode_name, bk_ix, bk_iy, bk_iz, bk[0], bk[1], bk[2]);

    /* Mid-gray: Y=502/1023, Cb=Cr=512/1023 */
    int mg_ix = (int)(502.0 / 1023.0 * 32.0 + 0.5);
    int mg_iy = (int)(512.0 / 1023.0 * 32.0 + 0.5);
    int mg_iz = mg_iy;
    const uint16_t *mg = SAMPLE(mg_ix, mg_iy, mg_iz);
    fprintf(stderr, "[LUT3D %s] Mid-gray (idx %d,%d,%d): Y=%u/65535 (%.4f) Cb=%u Cr=%u\n",
            mode_name, mg_ix, mg_iy, mg_iz, mg[0], mg[0] / 65535.0, mg[1], mg[2]);

    /* Peak white: Y=940/1023, Cb=Cr=512/1023 */
    int pw_ix = (int)(940.0 / 1023.0 * 32.0 + 0.5);
    int pw_iy = mg_iy;
    int pw_iz = mg_iz;
    const uint16_t *pw = SAMPLE(pw_ix, pw_iy, pw_iz);
    fprintf(stderr, "[LUT3D %s] Peak white (idx %d,%d,%d): Y=%u/65535 (%.4f) Cb=%u Cr=%u\n",
            mode_name, pw_ix, pw_iy, pw_iz, pw[0], pw[0] / 65535.0, pw[1], pw[2]);

    #undef SAMPLE
}
