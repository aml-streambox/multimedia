// hdr_colorconv.glsl - HDR color space conversion functions
//
// Shared between Y and UV TBDR shaders.
// Push constant field `color_mode` selects the conversion:
//   0 = passthrough (no color conversion)
//   1 = HDR10 (BT.2020 + PQ) -> SDR (BT.709 + gamma)
//   2 = HLG (BT.2020 + HLG) -> SDR (BT.709 + gamma)
//
// All math is done in float (mediump where possible for Mali-G52 perf).
// The pipeline for modes 1,2:
//   1. Normalized YCbCr -> linear BT.2020 RGB (matrix + EOTF)
//   2. BT.2020 -> BT.709 gamut map (3x3 matrix + soft-clip)
//   3. Tone map (Reinhard on luminance channel)
//   4. BT.709 OETF (gamma ~1/2.4)
//   5. Linear BT.709 RGB -> output YCbCr (matrix)
//
// Performance: ~25-30 ALU cycles on Mali-G52 SFU, bandwidth-dominated.

// ---- BT.2020 YCbCr -> RGB matrix (full range 10-bit: Y 0-1023, Cb/Cr 0-1023) ----
// Input: normalized [0,1] where 0=code 0, 1=code 1023 (full range)
// BT.2020 non-constant luminance:
//   R = Y + 1.4746 * (Cr - 0.5)
//   G = Y - 0.1646 * (Cb - 0.5) - 0.5714 * (Cr - 0.5)
//   B = Y + 1.8814 * (Cb - 0.5)
//
// For limited range (64-940 Y, 64-960 CbCr in 10-bit):
//   Y' = (Y_norm * 1023 - 64) / (940 - 64)
//   C' = (C_norm * 1023 - 512) / (960 - 64)

// Limited-range scale factors for 10-bit
const float LR_Y_SCALE  = 1023.0 / 876.0;   // 1/(940-64)*1023
const float LR_Y_OFFSET = 64.0 / 1023.0;
const float LR_C_SCALE  = 1023.0 / 896.0;   // 1/(960-64)*1023
const float LR_C_OFFSET = 512.0 / 1023.0;

vec3 ycbcr_to_rgb_bt2020(float y_norm, float cb_norm, float cr_norm) {
    // Limited range -> normalized Y'CbCr
    float yp = (y_norm - LR_Y_OFFSET) * LR_Y_SCALE;
    float cb = (cb_norm - LR_C_OFFSET) * LR_C_SCALE;
    float cr = (cr_norm - LR_C_OFFSET) * LR_C_SCALE;

    // BT.2020 matrix
    float r = yp + 1.4746 * cr;
    float g = yp - 0.1646 * cb - 0.5714 * cr;
    float b = yp + 1.8814 * cb;

    return clamp(vec3(r, g, b), 0.0, 1.0);
}

// ---- PQ (ST 2084) EOTF: electrical -> linear (0..10000 nits normalized to 0..1) ----
// L = ((max(E^(1/m2) - c1, 0)) / (c2 - c3 * E^(1/m2)))^(1/m1)
// where m1=2610/16384*4, m2=2523/32*128, c1=3424/4096, c2=2413/128, c3=2392/128

const float PQ_M1 = 0.1593017578125;     // 2610/16384
const float PQ_M2 = 78.84375;            // 2523/32
const float PQ_C1 = 0.8359375;           // 3424/4096
const float PQ_C2 = 18.8515625;          // 2413/128
const float PQ_C3 = 18.6875;             // 2392/128

float pq_eotf(float e) {
    float ep = pow(max(e, 0.0), 1.0 / PQ_M2);
    float num = max(ep - PQ_C1, 0.0);
    float den = PQ_C2 - PQ_C3 * ep;
    return pow(num / den, 1.0 / PQ_M1);
}

vec3 pq_eotf_vec3(vec3 e) {
    return vec3(pq_eotf(e.r), pq_eotf(e.g), pq_eotf(e.b));
}

// ---- HLG OETF inverse (electrical -> scene-linear 0..1) ----
// HLG inverse OETF:
//   if E <= 0.5: L = (E^2) / 3
//   if E >  0.5: L = (exp((E - c) / a) + b) / 12
// where a=0.17883277, b=1-4a=0.28466892, c=0.5-a*ln(4a)=0.55991073

const float HLG_A = 0.17883277;
const float HLG_B = 0.28466892;
const float HLG_C = 0.55991073;

float hlg_inv_oetf(float e) {
    if (e <= 0.5)
        return (e * e) / 3.0;
    else
        return (exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0;
}

// HLG OOTF: scene linear -> display linear
// L_d = alpha * L_s^(gamma-1) * L_s  (per-channel with luminance gamma boost)
// For 1000 nit display: gamma=1.2, alpha=1.0
// Simplified: apply pow(Y_scene, 0.2) boost to each channel
vec3 hlg_to_linear(vec3 e) {
    vec3 scene = vec3(hlg_inv_oetf(e.r), hlg_inv_oetf(e.g), hlg_inv_oetf(e.b));
    float y_scene = dot(scene, vec3(0.2627, 0.6780, 0.0593)); // BT.2020 luminance
    float boost = pow(max(y_scene, 1e-6), 0.2); // gamma-1 = 1.2-1 = 0.2
    return scene * boost;
}

// ---- BT.2020 -> BT.709 gamut mapping (3x3 matrix) ----
// This is the NPM conversion: BT.2020 XYZ -> BT.709 XYZ
// Combined: M_709_from_2020 (direct, no XYZ intermediate)
const mat3 BT2020_TO_BT709 = mat3(
     1.6605, -0.1246, -0.0182,
    -0.5876,  1.1329, -0.1006,
    -0.0728, -0.0083,  1.1187
);

vec3 gamut_bt2020_to_bt709(vec3 rgb2020) {
    vec3 rgb709 = BT2020_TO_BT709 * rgb2020;
    // Soft-clip: desaturate out-of-gamut instead of hard clamp
    // Simple approach: if any channel is negative or >1, compress
    float max_c = max(max(rgb709.r, rgb709.g), rgb709.b);
    float min_c = min(min(rgb709.r, rgb709.g), rgb709.b);
    if (min_c < 0.0 || max_c > 1.0) {
        float lum = dot(rgb709, vec3(0.2126, 0.7152, 0.0722));
        // Pull toward luma to bring within range
        float t = 1.0;
        if (min_c < 0.0) t = min(t, lum / (lum - min_c));
        if (max_c > 1.0) t = min(t, (1.0 - lum) / (max_c - lum + 1e-6));
        t = max(t, 0.0);
        rgb709 = mix(vec3(lum), rgb709, t);
    }
    return clamp(rgb709, 0.0, 1.0);
}

// ---- Reinhard tone mapping (luminance-based) ----
// L_out = L / (1 + L)
// Applied to luminance, then scale RGB proportionally to preserve hue.
// Input: linear light (0..unbounded), output: linear light (0..1)
vec3 tonemap_reinhard(vec3 linear_rgb) {
    float lum = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
    if (lum < 1e-6) return vec3(0.0);
    float lum_mapped = lum / (1.0 + lum);
    return linear_rgb * (lum_mapped / lum);
}

// ---- BT.709 OETF (gamma encoding) ----
// E = 1.099 * L^0.45 - 0.099  for L >= 0.018
// E = 4.5 * L                 for L < 0.018
float bt709_oetf(float l) {
    if (l < 0.018)
        return 4.5 * l;
    else
        return 1.099 * pow(l, 0.45) - 0.099;
}

vec3 bt709_oetf_vec3(vec3 l) {
    return vec3(bt709_oetf(l.r), bt709_oetf(l.g), bt709_oetf(l.b));
}

// ---- BT.709 RGB -> YCbCr (limited range output) ----
// Y  = 0.2126 R + 0.7152 G + 0.0722 B
// Cb = (B - Y) / 1.8556
// Cr = (R - Y) / 1.5748
// Scale to limited range (16-235 Y, 16-240 CbCr for 8-bit; scaled for 10-bit)
// Output as normalized [0,1] for the framebuffer format

const float OUT_Y_SCALE  = 876.0 / 1023.0;   // (940-64)/1023
const float OUT_Y_OFFSET = 64.0 / 1023.0;
const float OUT_C_SCALE  = 896.0 / 1023.0;   // (960-64)/1023
const float OUT_C_OFFSET = 512.0 / 1023.0;

float rgb_to_y_bt709(vec3 rgb) {
    float y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    return clamp(y * OUT_Y_SCALE + OUT_Y_OFFSET, 0.0, 1.0);
}

vec2 rgb_to_cbcr_bt709(vec3 rgb) {
    float y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float cb = (rgb.b - y) / 1.8556;
    float cr = (rgb.r - y) / 1.5748;
    cb = clamp(cb * OUT_C_SCALE + OUT_C_OFFSET, 0.0, 1.0);
    cr = clamp(cr * OUT_C_SCALE + OUT_C_OFFSET, 0.0, 1.0);
    return vec2(cb, cr);
}

// ---- Top-level conversion functions ----

// Convert a single pixel's YCbCr (normalized 0..1 from 10-bit) through
// the full HDR->SDR pipeline and return the output Y value.
float hdr_to_sdr_y(float y_in, float cb_in, float cr_in, uint color_mode) {
    if (color_mode == 0u) return y_in;  // passthrough

    // Step 1: YCbCr -> BT.2020 RGB (non-linear)
    vec3 rgb_nl = ycbcr_to_rgb_bt2020(y_in, cb_in, cr_in);

    // Step 2: EOTF -> linear light
    vec3 linear_rgb;
    if (color_mode == 1u) {
        // PQ (HDR10)
        linear_rgb = pq_eotf_vec3(rgb_nl);
    } else {
        // HLG
        linear_rgb = hlg_to_linear(rgb_nl);
    }

    // Step 3: BT.2020 -> BT.709 gamut map
    vec3 rgb709 = gamut_bt2020_to_bt709(linear_rgb);

    // Step 4: Tone map (Reinhard)
    vec3 tonemapped = tonemap_reinhard(rgb709);

    // Step 5: BT.709 OETF
    vec3 gamma_rgb = bt709_oetf_vec3(tonemapped);

    // Step 6: RGB -> BT.709 Y
    return rgb_to_y_bt709(gamma_rgb);
}

// Convert a single pixel's YCbCr through HDR->SDR and return output CbCr.
vec2 hdr_to_sdr_cbcr(float y_in, float cb_in, float cr_in, uint color_mode) {
    if (color_mode == 0u) return vec2(cb_in, cr_in);  // passthrough

    vec3 rgb_nl = ycbcr_to_rgb_bt2020(y_in, cb_in, cr_in);

    vec3 linear_rgb;
    if (color_mode == 1u) {
        linear_rgb = pq_eotf_vec3(rgb_nl);
    } else {
        linear_rgb = hlg_to_linear(rgb_nl);
    }

    vec3 rgb709 = gamut_bt2020_to_bt709(linear_rgb);
    vec3 tonemapped = tonemap_reinhard(rgb709);
    vec3 gamma_rgb = bt709_oetf_vec3(tonemapped);

    return rgb_to_cbcr_bt709(gamma_rgb);
}
