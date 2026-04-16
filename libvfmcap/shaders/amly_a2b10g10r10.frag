#version 450

// TBDR-optimized A2B10G10R10 AFBC output with 3D LUT HDR conversion.
//
// Reads AMLY SSBO, decodes 40-bit packed YCbCr 422, converts to RGB,
// applies HDR tone mapping via 3D LUT, outputs to A2B10G10R10 color attachment.
// Mali compresses to AFBC automatically when the VkImage uses AFBC tiling.
//
// Push constants: { src_width, src_height, pairs_per_row, color_mode }
//   color_mode: 0=passthrough, 1=HDR10 (PQ)->SDR, 2=HLG->SDR
// Binding 0: AMLY input buffer (SSBO)
// Binding 1: 3D LUT texture (sampler3D)
// Output: A2B10G10R10_UNORM_PACK32 color attachment

precision mediump float;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0, std430) readonly buffer AMLYBuffer {
    uint data[];
} amly;

layout(set = 0, binding = 1) uniform sampler3D lut3d;

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

void read_pair(uint pair_idx, out uint lo, out uint hi) {
    uint byte_offset = pair_idx * 5u;
    uint word_idx = byte_offset >> 2;
    uint shift = (byte_offset & 3u) << 3;

    uint base0 = word_idx & ~1u;
    uint blk0_lo = amly.data[base0];
    uint blk0_hi = amly.data[base0 + 1u];
    uint cw0 = ((word_idx & 1u) == 0u) ? bswap32(blk0_hi) : bswap32(blk0_lo);

    uint p1 = word_idx + 1u;
    uint base1 = p1 & ~1u;

    uint blk1_lo = blk0_lo;
    uint blk1_hi = blk0_hi;
    if (base1 != base0) {
        blk1_lo = amly.data[base1];
        blk1_hi = amly.data[base1 + 1u];
    }
    uint cw1 = ((p1 & 1u) == 0u) ? bswap32(blk1_hi) : bswap32(blk1_lo);

    if (shift == 0u) {
        lo = cw0;
        hi = cw1 & 0xFFu;
    } else {
        uint p2 = word_idx + 2u;
        uint base2 = p2 & ~1u;

        uint cw2;
        if (base2 == base0) {
            cw2 = ((p2 & 1u) == 0u) ? bswap32(blk0_hi) : bswap32(blk0_lo);
        } else if (base2 == base1) {
            cw2 = ((p2 & 1u) == 0u) ? bswap32(blk1_hi) : bswap32(blk1_lo);
        } else {
            uint blk2_lo = amly.data[base2];
            uint blk2_hi = amly.data[base2 + 1u];
            cw2 = ((p2 & 1u) == 0u) ? bswap32(blk2_hi) : bswap32(blk2_lo);
        }
        lo = (cw0 >> shift) | (cw1 << (32u - shift));
        hi = ((cw1 >> shift) | (cw2 << (32u - shift))) & 0xFFu;
    }
}

vec3 ycbcr_to_rgb(float y, float cb, float cr) {
    float r = y + 1.402 * (cr - 0.5);
    float g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5);
    float b = y + 1.772 * (cb - 0.5);
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main() {
    uint src_x = uint(v_uv.x * float(params.src_width));
    uint src_y = uint(v_uv.y * float(params.src_height));

    src_x = min(src_x, params.src_width - 1u);
    src_y = min(src_y, params.src_height - 1u);

    uint pair_in_row = src_x >> 1;
    uint pair_idx = src_y * params.pairs_per_row + pair_in_row;

    uint lo, hi;
    read_pair(pair_idx, lo, hi);

    uint bs = bswap32(lo);
    uint y0 = (bs >> 22) & 0x3FFu;
    uint cb = (bs >> 12) & 0x3FFu;
    uint y1 = (bs >> 2) & 0x3FFu;
    uint cr = ((bs & 0x3u) << 8) | hi;

    uint y_val = ((src_x & 1u) == 0u) ? y0 : y1;

    float y_f = float(y_val) / 1023.0;
    float cb_f = float(cb) / 1023.0;
    float cr_f = float(cr) / 1023.0;

    if (params.color_mode != 0u) {
        vec3 coord = vec3(y_f, cb_f, cr_f);
        vec4 sdr = texture(lut3d, coord);
        y_f = sdr.r;
        cb_f = sdr.g;
        cr_f = sdr.b;
    }

    vec3 rgb = ycbcr_to_rgb(y_f, cb_f, cr_f);
    out_color = vec4(rgb, 1.0);
}
