#!/bin/bash
# Compile GLSL compute shaders to SPIR-V and generate C headers.
#
# Produces:
#   amly_to_p010_spv.h  (included by vfmcap_vulkan.c)
#   amly_to_nv12_spv.h  (included by vfmcap_vulkan.c)
#
# Requires glslangValidator (from KhronosGroup/glslang) or glslc, plus xxd.
#
# Usage:
#   bash compile_shaders.sh
#   rm -f ../src/vfmcap_vulkan.o   # force recompile
#   bitbake libvfmcap -f -c compile

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SHADERS=("amly_to_p010" "amly_to_nv12")

for shader in "${SHADERS[@]}"; do
    INPUT="${shader}.comp"
    SPV="${shader}.spv"
    HEADER="${shader}_spv.h"

    if [ ! -f "$INPUT" ]; then
        echo "Error: $INPUT not found"
        exit 1
    fi

    if command -v glslangValidator &> /dev/null; then
        glslangValidator -V "$INPUT" -o "$SPV"
        echo "Compiled $INPUT -> $SPV (glslangValidator)"
    elif command -v glslc &> /dev/null; then
        glslc "$INPUT" -o "$SPV"
        echo "Compiled $INPUT -> $SPV (glslc)"
    else
        echo "Error: Neither glslangValidator nor glslc found"
        echo "Install Vulkan SDK to compile shaders"
        exit 1
    fi

    # Convert SPIR-V binary to C header for embedding
    xxd -i "$SPV" > "$HEADER"
    echo "Generated $HEADER"
done

echo "All shaders compiled successfully."
