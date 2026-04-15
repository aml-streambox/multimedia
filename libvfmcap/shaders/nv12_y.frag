#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float out_y;

layout(set = 0, binding = 0) uniform sampler2D src_y;

void main() {
    out_y = texture(src_y, v_uv).r;
}
