#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

layout(set = 0, binding = 0) uniform sampler2D src_y;

void main() {
    // Sample and scale to 10-bit in upper bits of 16-bit
    float y = texture(src_y, v_uv).r;
    out_y = y * 65535.0 / 64.0; // 10-bit left-justified in 16-bit
}
