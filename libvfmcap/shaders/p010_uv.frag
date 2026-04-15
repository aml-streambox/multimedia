#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec2 out_uv;

layout(set = 0, binding = 0) uniform sampler2D src_yuv;

void main() {
    vec3 yuv = texture(src_yuv, v_uv).rgb;
    out_uv = yuv.gb * 65535.0 / 64.0; // U and V, 10-bit left-justified
}
