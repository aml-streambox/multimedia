#version 450

// NV12 Y output from pre-decoded intermediates with 3D LUT HDR conversion.
//
// Same as predecode_p010_y.frag but output is R8_UNORM.
//
// Push constants: { color_mode } (0=passthrough, 1=HDR10, 2=HLG)
// Binding 0: pre-decoded Y (sampler2D R16_UNORM)
// Binding 1: pre-decoded UV (sampler2D R16G16_UNORM)
// Binding 2: 3D LUT (sampler3D RGBA16_UNORM)
// Output: R8_UNORM color attachment

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
        // Passthrough: R16 -> R8 natural truncation
        out_y = y_raw;
    } else {
        // HDR->SDR via 3D LUT
        vec2 uv_raw = texture(src_uv, v_uv).rg;

        vec3 coord = vec3(y_raw, uv_raw.r, uv_raw.g);
        vec4 sdr = texture(lut3d, coord);

        out_y = sdr.r;
    }
}
