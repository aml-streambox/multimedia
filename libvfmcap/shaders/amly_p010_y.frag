#version 450
#extension GL_GOOGLE_include_directive : enable

// TBDR-optimized P010 Y output: reads AMLY SSBO directly in the fragment shader.
//
// Each fragment maps to one luma pixel in the source. The fragment shader:
//   1. Computes source pixel coords from gl_FragCoord
//   2. Computes the AMLY pair index and extracts the 40-bit group
//   3. Applies bswap64 correction (hardware endianness)
//   4. Extracts Y0 or Y1 depending on even/odd x
//   5. Outputs left-justified 10-bit (val << 6) as R16_UNORM
//
// When color_mode != 0, also extracts Cb/Cr and runs HDR->SDR conversion,
// outputting the BT.709 Y component instead of the raw BT.2020 Y.
//
// No intermediate images — Mali TBDR keeps the output in tile memory.
//
// Push constants: { src_width, src_height, pairs_per_row, color_mode }
//   color_mode: 0=passthrough, 1=HDR10 (PQ)->SDR, 2=HLG->SDR
// Binding 0: AMLY input buffer (SSBO)
// Output: R16_UNORM color attachment at source (or scaled) resolution

#include "hdr_colorconv.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

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

// Read a corrected word at global index p, applying bswap64 correction.
// The hardware writes big-endian 64-bit words; each 64-bit-aligned pair
// of uint32s must be swapped and byte-reversed.
uint read_corrected_word(uint p) {
    uint base = p & ~1u;
    uint raw_a = amly.data[base];
    uint raw_b = amly.data[base + 1u];
    return ((p & 1u) == 0u) ? bswap32(raw_b) : bswap32(raw_a);
}

// Extract a 40-bit AMLY pair from the buffer at the given pair index.
// Returns {lo: bits 0-31, hi: bits 32-39} after bswap64 correction.
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
    // gl_FragCoord.xy gives the output pixel center (0.5, 0.5) based.
    // Map to source coordinates using UV (supports scaling).
    uint src_x = uint(v_uv.x * float(params.src_width));
    uint src_y = uint(v_uv.y * float(params.src_height));

    // Clamp to valid range
    src_x = min(src_x, params.src_width - 1u);
    src_y = min(src_y, params.src_height - 1u);

    // Each pair encodes 2 pixels: pair_idx = x / 2, pixel within pair = x & 1
    uint pair_in_row = src_x >> 1;
    uint pair_idx = src_y * params.pairs_per_row + pair_in_row;

    uint lo, hi;
    read_pair(pair_idx, lo, hi);

    uint bs = bswap32(lo);
    uint y_val;
    if ((src_x & 1u) == 0u) {
        y_val = (bs >> 22) & 0x3FFu;  // Y0
    } else {
        y_val = (bs >> 2) & 0x3FFu;   // Y1
    }

    if (params.color_mode == 0u) {
        // Passthrough: P010 left-justified 10-bit in 16-bit
        out_y = float(y_val << 6u) / 65535.0;
    } else {
        // HDR->SDR: need Cb and Cr too for full RGB conversion
        uint cb_val = (bs >> 12) & 0x3FFu;
        uint cr_val = ((bs & 0x3u) << 8) | hi;

        // Normalize to [0,1] (10-bit range 0..1023)
        float y_n  = float(y_val)  / 1023.0;
        float cb_n = float(cb_val) / 1023.0;
        float cr_n = float(cr_val) / 1023.0;

        // Full HDR->SDR pipeline, get output Y
        float out_y_sdr = hdr_to_sdr_y(y_n, cb_n, cr_n, params.color_mode);

        // Output as P010: left-justified 10-bit in 16-bit
        // hdr_to_sdr_y returns a value already in limited-range [0,1] for R16_UNORM
        // We need to convert back: the function returns a normalized Y that maps
        // to limited range 10-bit. For P010 we left-justify: val10 << 6 in 16-bit.
        // Since out_y_sdr is already [0,1] representing the final 10-bit normalized
        // value, we need: output = out_y_sdr * 1023/65535 * 64 ... simplify:
        // Actually, out_y_sdr from rgb_to_y_bt709() is already scaled to represent
        // the fraction of 10-bit range. For P010, the convention is left-justified:
        //   stored_u16 = round(val_10bit << 6) = round(val_10bit * 64)
        //   normalized = stored_u16 / 65535
        // Since out_y_sdr = val_10bit / 1023 (it's already the 10-bit normalized),
        //   stored_u16 = out_y_sdr * 1023 * 64 = out_y_sdr * 65472
        //   normalized = out_y_sdr * 65472 / 65535 ≈ out_y_sdr * 0.99904
        // Close enough to just output out_y_sdr directly, but for exact P010:
        out_y = out_y_sdr * 65472.0 / 65535.0;
    }
}
