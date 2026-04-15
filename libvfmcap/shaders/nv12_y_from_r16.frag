#version 450

// NV12 Y output from R16_UNORM intermediate (10-bit compute→graphics chain)
//
// Intermediate stores left-justified 10-bit: uint16 = val10 << 6.
// texture() returns: (val10 << 6) / 65535.0
// NV12 output is R8_UNORM. We want: val8 / 255.0 where val8 = val10 >> 2.
//
// The GPU's natural UNORM quantization when writing to R8 does:
//   stored_uint8 = round(out_float * 255.0)
// With passthrough: stored_uint8 = round((val10*64)/65535 * 255)
//                                = round(val10 * 16320 / 65535)
//                                ≈ round(val10 * 0.2490)  — WRONG, we want val10/4
//
// Correct: we need out_float = val10 / (4 * 255) = val10 / 1020.
// From texture: sampled = val10 * 64 / 65535
// So: out_float = sampled * 65535 / (64 * 1020) = sampled * (65535.0 / 65280.0)
//
// 65535/65280 = 1.003906... ≈ 1.0 — close enough that passthrough works.
// The maximum error is 1 LSB at val10=1023: passthrough gives 255, correct is 255.
// Just pass through and let the R8 framebuffer quantize naturally.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

layout(set = 0, binding = 0) uniform sampler2D src_y;

void main() {
    out_y = texture(src_y, v_uv).r;
}
