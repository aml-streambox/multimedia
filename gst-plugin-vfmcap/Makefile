TARGET = libgststreamboxsrc.so

SRCS = gst_streambox_src.c
OBJS = $(patsubst %.c, %.o, $(SRCS))

CFLAGS = -Wall -Wextra -Wno-unused-parameter -fPIC
CFLAGS += $(shell $(PKG_CONFIG) --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0)
CFLAGS += -I$(STAGING_DIR)/usr/include/

LDFLAGS += $(shell $(PKG_CONFIG) --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0)
LDFLAGS += -L$(STAGING_DIR)/usr/lib/ -lvfmcap -lvulkan

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $^ $(LDFLAGS) -shared -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

install:
	cp $(TARGET) $(TARGET_DIR)/usr/lib/gstreamer-1.0/

clean:
	rm -f $(OBJS) $(TARGET) gstvfmcapsrc.o libgstvfmcapsrc.so

.PHONY: all install clean
