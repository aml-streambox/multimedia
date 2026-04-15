#version 450

// P010 UV output from R16G16_UNORM intermediate (10-bit compute→graphics chain)
//
// Same UNORM passthrough logic as the Y plane: both intermediate and output
// are 16-bit UNORM, so the normalized [0,1] float maps identically.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec2 out_uv;

layout(set = 0, binding = 1) uniform sampler2D src_uv;

void main() {
    out_uv = texture(src_uv, v_uv).rg;
}
