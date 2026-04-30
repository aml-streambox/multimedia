#version 450

// TBDR-optimized NV12 Y output with 3D LUT HDR conversion.
//
// Same decode logic as amly_p010_y.frag, but output is R8_UNORM.
// When color_mode != 0: samples 3D LUT for BT.709 Y.
//
// Push constants: { src_width, src_height, pairs_per_row, color_mode }
//   color_mode: 0=passthrough, 1=HDR10 (PQ)->SDR, 2=HLG->SDR
// Binding 0: AMLY input buffer (SSBO)
// Binding 1: 3D LUT texture (sampler3D) — HDR YCbCr->YCbCr lookup
// Output: R8_UNORM color attachment at source (or scaled) resolution

precision mediump float;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

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

// Optimized read_pair: inline bswap64 correction, reuse shared 64-bit blocks.
// The vdin path presents each 64-bit block word-swapped and byte-reversed.
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
    uint y_val;
    if ((src_x & 1u) == 0u) {
        y_val = (bs >> 22) & 0x3FFu;
    } else {
        y_val = (bs >> 2) & 0x3FFu;
    }

    if (params.color_mode == 0u) {
        // Passthrough: R8_UNORM truncates 10->8 bit naturally
        out_y = float(y_val << 6u) / 65535.0;
    } else {
        // HDR->SDR via 3D LUT
        uint cb_val = (bs >> 12) & 0x3FFu;
        uint cr_val = ((bs & 0x3u) << 8) | hi;

        vec3 coord = vec3(float(y_val) / 1023.0,
                          float(cb_val) / 1023.0,
                          float(cr_val) / 1023.0);

        vec4 sdr = texture(lut3d, coord);

        // NV12 R8_UNORM: hardware writes round(out_float * 255.0)
        out_y = sdr.r;
    }
}
