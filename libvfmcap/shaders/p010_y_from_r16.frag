#version 450

// P010 Y output from R16_UNORM intermediate (10-bit compute→graphics chain)
//
// The intermediate R16_UNORM image stores left-justified 10-bit values:
//   stored_uint16 = val10 << 6
// texture() returns: stored_uint16 / 65535.0 → a value in [0,1]
//
// P010 output is also R16_UNORM. We want the same left-justified 10-bit
// layout in the output. Since both are R16_UNORM, the normalized [0,1]
// value maps back to the same uint16 — pure passthrough.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

layout(set = 0, binding = 0) uniform sampler2D src_y;

void main() {
    out_y = texture(src_y, v_uv).r;
}
