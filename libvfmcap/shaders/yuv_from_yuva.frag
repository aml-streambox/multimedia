#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_rgba;

layout(set = 0, binding = 0) uniform sampler2D src_yuva;

void main() {
    // For AFBC A2B10G10R10 output from YUV-domain intermediate:
    // We may need to convert YUV -> RGB here depending on downstream needs.
    // For now, pass through as a placeholder.
    vec4 yuva = texture(src_yuva, v_uv);
    out_rgba = yuva;
}
