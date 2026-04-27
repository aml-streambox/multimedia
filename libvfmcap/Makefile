#
# Makefile for libvfmcap.so + vfmcap-demo
#
# Builds:
#   libvfmcap.so   - Shared library (V4L2 capture + Vulkan GPU converter)
#   vfmcap-demo    - Demo/validation program
#
# Yocto sets: CC, CFLAGS, LDFLAGS, STAGING_DIR, PKG_CONFIG, TARGET_DIR

LOCAL_PATH := $(shell pwd)

# --- Library ---
LIB_TARGET = libvfmcap.so
LIB_SRCS = src/vfmcap.c src/vfmcap_vulkan.c src/hdr_lut3d.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

# --- Demo ---
DEMO_TARGET = vfmcap-demo
DEMO_SRCS = demo/vfmcap-demo.c
DEMO_OBJS = $(DEMO_SRCS:.c=.o)

# --- Flags ---
CFLAGS += -Wall -Wextra -fPIC
CFLAGS += -I$(LOCAL_PATH)/include
CFLAGS += -I$(LOCAL_PATH)/src

ifdef STAGING_DIR
  CFLAGS += -I$(STAGING_DIR)/usr/include
  CFLAGS += -I$(STAGING_DIR)/usr/include/vulkan
  LDFLAGS += -L$(STAGING_DIR)/usr/lib
endif

LIB_LDFLAGS = $(LDFLAGS) -lvulkan -lm -shared -Wl,-Bsymbolic
DEMO_LDFLAGS = $(LDFLAGS) -L$(LOCAL_PATH) -lvfmcap -lm -Wl,-rpath,/usr/lib

# --- Targets ---
.PHONY: all clean install

all: $(LIB_TARGET) $(DEMO_TARGET)

$(LIB_TARGET): $(LIB_OBJS)
	$(CC) $^ $(LIB_LDFLAGS) -o $@

$(DEMO_TARGET): $(DEMO_OBJS) $(LIB_TARGET)
	$(CC) $(DEMO_OBJS) $(DEMO_LDFLAGS) -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

# Ensure shader SPIR-V headers exist
src/vfmcap_vulkan.o: shaders/amly_to_p010_spv.h shaders/amly_to_nv12_spv.h \
    shaders/amly_decode_spv.h \
    shaders/fullscreen.vert_spv.h \
    shaders/nv12_y.frag_spv.h shaders/nv12_uv.frag_spv.h \
    shaders/p010_y.frag_spv.h shaders/p010_uv.frag_spv.h \
    shaders/nv12_y_from_r16.frag_spv.h shaders/nv12_uv_from_r16g16.frag_spv.h \
    shaders/p010_y_from_r16.frag_spv.h shaders/p010_uv_from_r16g16.frag_spv.h \
    shaders/amly_p010_y_spv.h shaders/amly_p010_uv_spv.h \
    shaders/amly_nv12_y_spv.h shaders/amly_nv12_uv_spv.h \
    shaders/amly_a2b10g10r10_spv.h

clean:
	rm -f $(LIB_OBJS) $(DEMO_OBJS) $(LIB_TARGET) $(DEMO_TARGET)

install:
	install -d -m 0755 $(TARGET_DIR)/usr/lib
	install -d -m 0755 $(TARGET_DIR)/usr/bin
	install -d -m 0755 $(TARGET_DIR)/usr/include
	install -D -m 0755 $(LIB_TARGET) $(TARGET_DIR)/usr/lib/$(LIB_TARGET)
	install -D -m 0755 $(DEMO_TARGET) $(TARGET_DIR)/usr/bin/$(DEMO_TARGET)
	install -D -m 0644 include/vfmcap.h $(TARGET_DIR)/usr/include/vfmcap.h
