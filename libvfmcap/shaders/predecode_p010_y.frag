#version 450

// P010 Y output from pre-decoded intermediates with 3D LUT HDR conversion.
//
// Compute shader has already unpacked AMLY into:
//   - R16_UNORM Y image: left-justified 10-bit (val << 6) / 65535.0
//   - R16G16_UNORM UV image: left-justified 10-bit Cb, Cr
//
// Passthrough: sample Y, output directly (both R16_UNORM, identity map).
// HDR mode: reconstruct YCbCr from both textures, sample 3D LUT.
//
// Push constants: { color_mode } (0=passthrough, 1=HDR10, 2=HLG)
// Binding 0: pre-decoded Y (sampler2D R16_UNORM)
// Binding 1: pre-decoded UV (sampler2D R16G16_UNORM)
// Binding 2: 3D LUT (sampler3D RGBA16_UNORM)
// Output: R16_UNORM color attachment

precision mediump float;
precision mediump sampler3D;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

layout(set = 0, binding = 0) uniform sampler2D src_y;
layout(set = 0, binding = 1) uniform sampler2D src_uv;
layout(set = 0, binding = 2) uniform sampler3D lut3d;

layout(push_constant) uniform Params {
    uint color_mode;
} params;

void main() {
    float y_raw = texture(src_y, v_uv).r;

    if (params.color_mode == 0u) {
        // Passthrough: R16_UNORM -> R16_UNORM identity
        out_y = y_raw;
    } else {
        // HDR->SDR via 3D LUT
        // Recover 10-bit normalized: stored = (val10 << 6) / 65535
        // val10/1023 = stored * 65535 / (64 * 1023) = stored * 1.00044
        // Close enough to identity for LUT indexing.
        vec2 uv_raw = texture(src_uv, v_uv).rg;

        // Use the raw UNORM values directly as LUT coordinates
        // (they're approximately val10/1023 after the <<6 /65535 round-trip)
        vec3 coord = vec3(y_raw, uv_raw.r, uv_raw.g);
        vec4 sdr = texture(lut3d, coord);

        // Output as P010: left-justified 10-bit in 16-bit
        out_y = sdr.r * (65472.0 / 65535.0);
    }
}
