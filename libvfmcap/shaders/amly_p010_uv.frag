#version 450

// TBDR-optimized P010 UV output with 3D LUT HDR conversion.
//
// Runs at half resolution (width/2 x height/2).
// Reads two AMLY rows for 4:2:2 -> 4:2:0 vertical chroma subsampling.
// If HDR mode: normalizes averaged YCbCr, samples 3D LUT for BT.709 CbCr.
//
// Push constants: { src_width, src_height, pairs_per_row, color_mode }
//   color_mode: 0=passthrough, 1=HDR10 (PQ)->SDR, 2=HLG->SDR
// Binding 0: AMLY input buffer (SSBO)
// Binding 1: 3D LUT texture (sampler3D) — HDR YCbCr->YCbCr lookup
// Output: R16G16_UNORM color attachment at half resolution

precision mediump float;
precision mediump sampler3D;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec2 out_uv;

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
    uint uv_x = uint(v_uv.x * float(params.pairs_per_row));
    uint uv_y = uint(v_uv.y * float(params.src_height / 2u));

    uv_x = min(uv_x, params.pairs_per_row - 1u);
    uv_y = min(uv_y, params.src_height / 2u - 1u);

    uint src_row = uv_y * 2u;
    uint pair_in_row = uv_x;

    // Read current row pair
    uint pair_idx0 = src_row * params.pairs_per_row + pair_in_row;
    uint lo0, hi0;
    read_pair(pair_idx0, lo0, hi0);
    uint bs0 = bswap32(lo0);
    uint u0 = (bs0 >> 12) & 0x3FFu;
    uint v0 = ((bs0 & 0x3u) << 8) | hi0;
    uint y0_val = (bs0 >> 22) & 0x3FFu;

    // Read next row pair for vertical averaging
    uint u_avg, v_avg;
    uint y1_val = y0_val;

    if (src_row + 1u < params.src_height) {
        uint pair_idx1 = (src_row + 1u) * params.pairs_per_row + pair_in_row;
        uint lo1, hi1;
        read_pair(pair_idx1, lo1, hi1);
        uint bs1 = bswap32(lo1);
        uint nu0 = (bs1 >> 12) & 0x3FFu;
        uint nv0 = ((bs1 & 0x3u) << 8) | hi1;
        y1_val = (bs1 >> 22) & 0x3FFu;

        u_avg = (u0 + nu0 + 1u) >> 1;
        v_avg = (v0 + nv0 + 1u) >> 1;
    } else {
        u_avg = u0;
        v_avg = v0;
    }

    if (params.color_mode == 0u) {
        // Passthrough: P010 left-justified 10-bit in 16-bit
        float u_f = float(u_avg << 6u) / 65535.0;
        float v_f = float(v_avg << 6u) / 65535.0;
        out_uv = vec2(u_f, v_f);
    } else {
        // HDR->SDR via 3D LUT (modes 1=HDR10, 2=HLG)
        uint y_avg = (y0_val + y1_val + 1u) >> 1;

        vec3 coord = vec3(float(y_avg) / 1023.0,
                          float(u_avg) / 1023.0,
                          float(v_avg) / 1023.0);

        vec4 sdr = texture(lut3d, coord);

        // P010: left-justified 10-bit in 16-bit. Use Cb (G) and Cr (B).
        out_uv = sdr.gb * (65472.0 / 65535.0);
    }
}
