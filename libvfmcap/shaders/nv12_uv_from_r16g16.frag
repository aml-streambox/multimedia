#version 450

// NV12 UV output from R16G16_UNORM intermediate (10-bit compute→graphics chain)
//
// Same logic as NV12 Y: passthrough works because the R8G8 framebuffer
// naturally quantizes the 16-bit UNORM value to 8-bit, giving the correct
// 10→8 bit truncation with at most 1 LSB rounding difference.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec2 out_uv;

layout(set = 0, binding = 1) uniform sampler2D src_uv;

void main() {
    out_uv = texture(src_uv, v_uv).rg;
}
