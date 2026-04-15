#version 450
#extension GL_GOOGLE_include_directive : enable

// TBDR-optimized NV12 UV output: reads AMLY SSBO directly in the fragment shader.
//
// Same decode logic as amly_p010_uv.frag, but output is R8G8_UNORM.
// 10-bit chroma values are naturally truncated to 8-bit by the R8G8 framebuffer.
//
// When color_mode != 0, extracts Y from both rows, runs HDR->SDR conversion,
// and outputs BT.709 CbCr.
//
// Push constants: { src_width, src_height, pairs_per_row, color_mode }
//   color_mode: 0=passthrough, 1=HDR10 (PQ)->SDR, 2=HLG->SDR
// Binding 0: AMLY input buffer (SSBO)
// Output: R8G8_UNORM color attachment at half resolution

#include "hdr_colorconv.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec2 out_uv;

layout(set = 0, binding = 0, std430) readonly buffer AMLYBuffer {
    uint data[];
} amly;

layout(push_constant) uniform Params {
    uint src_width;
    uint src_height;
    uint pairs_per_row;
    uint color_mode;
} params;

uint bswap32(uint val) {
    return ((val & 0x000000FFu) << 24) |
           ((val & 0x0000FF00u) <<  8) |
           ((val & 0x00FF0000u) >>  8) |
           ((val & 0xFF000000u) >> 24);
}

uint read_corrected_word(uint p) {
    uint base = p & ~1u;
    uint raw_a = amly.data[base];
    uint raw_b = amly.data[base + 1u];
    return ((p & 1u) == 0u) ? bswap32(raw_b) : bswap32(raw_a);
}

void read_pair(uint pair_idx, out uint lo, out uint hi) {
    uint byte_offset = pair_idx * 5u;
    uint word_idx = byte_offset >> 2;
    uint shift = (byte_offset & 3u) << 3;

    uint cw0 = read_corrected_word(word_idx);
    uint cw1 = read_corrected_word(word_idx + 1u);

    if (shift == 0u) {
        lo = cw0;
        hi = cw1 & 0xFFu;
    } else {
        uint cw2 = read_corrected_word(word_idx + 2u);
        lo = (cw0 >> shift) | (cw1 << (32u - shift));
        hi = ((cw1 >> shift) | (cw2 << (32u - shift))) & 0xFFu;
    }
}

void main() {
    uint uv_x = uint(v_uv.x * float(params.pairs_per_row));
    uint uv_y = uint(v_uv.y * float(params.src_height / 2u));

    uv_x = min(uv_x, params.pairs_per_row - 1u);
    uv_y = min(uv_y, params.src_height / 2u - 1u);

    uint src_row = uv_y * 2u;
    uint pair_in_row = uv_x;

    uint pair_idx0 = src_row * params.pairs_per_row + pair_in_row;
    uint lo0, hi0;
    read_pair(pair_idx0, lo0, hi0);
    uint bs0 = bswap32(lo0);
    uint u0 = (bs0 >> 12) & 0x3FFu;
    uint v0 = ((bs0 & 0x3u) << 8) | hi0;
    uint y0_val = (bs0 >> 22) & 0x3FFu;  // Y0 from current row (for HDR path)

    uint u_avg, v_avg;
    uint y1_val = y0_val;  // default: same as current row

    if (src_row + 1u < params.src_height) {
        uint pair_idx1 = (src_row + 1u) * params.pairs_per_row + pair_in_row;
        uint lo1, hi1;
        read_pair(pair_idx1, lo1, hi1);
        uint bs1 = bswap32(lo1);
        uint nu0 = (bs1 >> 12) & 0x3FFu;
        uint nv0 = ((bs1 & 0x3u) << 8) | hi1;
        y1_val = (bs1 >> 22) & 0x3FFu;  // Y0 from next row

        u_avg = (u0 + nu0 + 1u) >> 1;
        v_avg = (v0 + nv0 + 1u) >> 1;
    } else {
        u_avg = u0;
        v_avg = v0;
    }

    if (params.color_mode == 0u) {
        // Passthrough: output as normalized float; R8G8_UNORM truncates 10->8 bit
        float u_f = float(u_avg << 6u) / 65535.0;
        float v_f = float(v_avg << 6u) / 65535.0;
        out_uv = vec2(u_f, v_f);
    } else {
        // HDR->SDR: use averaged Y with averaged chroma
        uint y_avg = (y0_val + y1_val + 1u) >> 1;

        float y_n  = float(y_avg)  / 1023.0;
        float cb_n = float(u_avg)  / 1023.0;
        float cr_n = float(v_avg)  / 1023.0;

        vec2 cbcr_sdr = hdr_to_sdr_cbcr(y_n, cb_n, cr_n, params.color_mode);

        // NV12 R8G8_UNORM: hardware writes round(out_float * 255.0).
        // hdr_to_sdr_cbcr returns normalized BT.709 CbCr values.
        out_uv = cbcr_sdr;
    }
}
