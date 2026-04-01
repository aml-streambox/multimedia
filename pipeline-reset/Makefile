# Makefile for pipeline-reset tool
#
# Native build:    make
# Cross-compile:   make CC=aarch64-none-linux-gnu-gcc
# Or via Yocto:    bitbake pipeline-reset

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter
TARGET = pipeline-reset

all: $(TARGET)

$(TARGET): pipeline-reset.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
