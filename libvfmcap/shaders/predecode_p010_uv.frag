#version 450

// P010 UV output from pre-decoded intermediates with 3D LUT HDR conversion.
//
// Runs at half resolution (width/2 x height/2).
// The compute shader already handled 4:2:2 -> 4:2:0 vertical chroma averaging.
// So src_uv is already at half-height — just sample and apply LUT.
//
// Push constants: { color_mode } (0=passthrough, 1=HDR10, 2=HLG)
// Binding 0: pre-decoded Y (sampler2D R16_UNORM)
// Binding 1: pre-decoded UV (sampler2D R16G16_UNORM)
// Binding 2: 3D LUT (sampler3D RGBA16_UNORM)
// Output: R16G16_UNORM color attachment at half resolution

precision mediump float;
precision mediump sampler3D;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec2 out_uv;

layout(set = 0, binding = 0) uniform sampler2D src_y;
layout(set = 0, binding = 1) uniform sampler2D src_uv;
layout(set = 0, binding = 2) uniform sampler3D lut3d;

layout(push_constant) uniform Params {
    uint color_mode;
} params;

void main() {
    vec2 uv_raw = texture(src_uv, v_uv).rg;

    if (params.color_mode == 0u) {
        // Passthrough: R16G16_UNORM -> R16G16_UNORM identity
        out_uv = uv_raw;
    } else {
        // HDR->SDR via 3D LUT
        // Sample Y at the corresponding position (center of the 2x2 luma block)
        float y_raw = texture(src_y, v_uv).r;

        vec3 coord = vec3(y_raw, uv_raw.r, uv_raw.g);
        vec4 sdr = texture(lut3d, coord);

        // P010: left-justified 10-bit in 16-bit. Use Cb (G) and Cr (B).
        out_uv = sdr.gb * (65472.0 / 65535.0);
    }
}
