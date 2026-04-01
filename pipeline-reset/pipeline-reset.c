/*
 * pipeline-reset.c — Reset vdin1 and Wave521 encoder to clean state
 *
 * After an unclean pipeline shutdown (SIGKILL, double Ctrl+C, etc.),
 * vdin1's V4L2 device (/dev/video71) can become permanently stuck
 * and the Wave521 encoder (/dev/amvenc_multi) may be in a bad state.
 *
 * This tool resets both devices to a post-reboot-equivalent state
 * without requiring a reboot.
 *
 * Usage: pipeline-reset [-v] [-n]
 *   -v   Verbose output
 *   -n   Dry run (diagnose only, don't reset)
 *
 * Must be run as root.
 *
 * Cross-compile:
 *   aarch64-none-linux-gnu-gcc -O2 -Wall -o pipeline-reset pipeline-reset.c
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/videodev2.h>

/* ------------------------------------------------------------------ */
/* Wave521 encoder ioctl definition (from vpu_multi.h)                */
/* ------------------------------------------------------------------ */
#define VDI_MAGIC       'V'
#define VDI_IOCTL_RESET _IOW(VDI_MAGIC, 4, unsigned int)

/* ------------------------------------------------------------------ */
/* Device paths                                                       */
/* ------------------------------------------------------------------ */
#define VDIN1_V4L2_DEV   "/dev/video71"
#define VDIN1_CHARDEV    "/dev/vdin1"
#define VDIN1_SYSFS_ATTR "/sys/class/vdin/vdin1/attr"
#define ENCODER_DEV      "/dev/amvenc_multi"
#define ENCODER_STATUS   "/sys/class/amvenc_multi/encode_status"

/* ------------------------------------------------------------------ */
/* Globals                                                            */
/* ------------------------------------------------------------------ */
static int verbose = 0;
static int dry_run = 0;
static int errors  = 0;

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static void info(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("[INFO]  ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

static void warn(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("[WARN]  ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

static void err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    errors++;
}

static void dbg(const char *fmt, ...) {
    if (!verbose) return;
    va_list ap;
    va_start(ap, fmt);
    printf("[DBG]   ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

/* Read a sysfs file into buf (nul-terminated, trailing newline stripped).
 * Returns 0 on success, -1 on failure. */
static int sysfs_read(const char *path, char *buf, size_t bufsz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        dbg("sysfs_read(%s): open failed: %s", path, strerror(errno));
        return -1;
    }
    ssize_t n = read(fd, buf, bufsz - 1);
    close(fd);
    if (n < 0) {
        dbg("sysfs_read(%s): read failed: %s", path, strerror(errno));
        return -1;
    }
    buf[n] = '\0';
    /* strip trailing newline */
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r'))
        buf[--n] = '\0';
    return 0;
}

/* Write a string to a sysfs file. Returns 0 on success, -1 on failure. */
static int sysfs_write(const char *path, const char *val) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        err("sysfs_write(%s): open failed: %s", path, strerror(errno));
        return -1;
    }
    size_t len = strlen(val);
    ssize_t n = write(fd, val, len);
    close(fd);
    if (n != (ssize_t)len) {
        err("sysfs_write(%s, \"%s\"): write failed: %s", path, val, strerror(errno));
        return -1;
    }
    return 0;
}

/* Check if a device node exists and is accessible */
static int device_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0);
}

/* ------------------------------------------------------------------ */
/* Kill any running gst-launch-1.0 processes                          */
/* ------------------------------------------------------------------ */
static int kill_gstreamer_processes(void) {
    int killed = 0;
    DIR *dir = opendir("/proc");
    if (!dir) {
        err("Cannot open /proc: %s", strerror(errno));
        return 0;
    }

    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        /* Only look at numeric entries (PIDs) */
        char *endp;
        long pid = strtol(de->d_name, &endp, 10);
        if (*endp != '\0' || pid <= 0)
            continue;

        /* Read /proc/<pid>/comm */
        char comm_path[64];
        snprintf(comm_path, sizeof(comm_path), "/proc/%ld/comm", pid);
        char comm[256];
        if (sysfs_read(comm_path, comm, sizeof(comm)) != 0)
            continue;

        if (strcmp(comm, "gst-launch-1.0") == 0) {
            if (dry_run) {
                info("Would kill gst-launch-1.0 (pid %ld)", pid);
                killed++;
            } else {
                info("Killing gst-launch-1.0 (pid %ld) with SIGKILL", pid);
                if (kill((pid_t)pid, SIGKILL) == 0) {
                    killed++;
                } else {
                    warn("Failed to kill pid %ld: %s", pid, strerror(errno));
                }
            }
        }
    }
    closedir(dir);

    if (killed > 0 && !dry_run) {
        /* Give kernel time to run release handlers */
        info("Waiting 2 seconds for kernel cleanup after kill...");
        sleep(2);
    }

    return killed;
}

/* ------------------------------------------------------------------ */
/* Diagnose vdin1 state                                               */
/* ------------------------------------------------------------------ */
static void diagnose_vdin1(void) {
    info("--- vdin1 diagnosis ---");

    if (!device_exists(VDIN1_V4L2_DEV)) {
        warn("V4L2 device %s does not exist", VDIN1_V4L2_DEV);
        return;
    }

    /* Try to open V4L2 device and query capabilities */
    int fd = open(VDIN1_V4L2_DEV, O_RDWR);
    if (fd < 0) {
        warn("Cannot open %s: %s", VDIN1_V4L2_DEV, strerror(errno));
        if (errno == EBUSY)
            info("  -> Device is busy (likely stuck from previous session)");
        return;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
        dbg("V4L2 driver: %s, card: %s, bus: %s",
            cap.driver, cap.card, cap.bus_info);
        dbg("V4L2 capabilities: 0x%08x, device_caps: 0x%08x",
            cap.capabilities, cap.device_caps);
    }

    /* Try REQBUFS with count=0 to test if the queue owner is stale.
     * If another fd was the owner and the process died, this returns -EBUSY. */
    struct v4l2_requestbuffers reqbufs;
    memset(&reqbufs, 0, sizeof(reqbufs));
    reqbufs.count  = 0;
    reqbufs.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbufs.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(fd, VIDIOC_REQBUFS, &reqbufs) < 0) {
        if (errno == EBUSY) {
            info("  vb2 queue has stale owner — device is STUCK");
            info("  (This is the -EPERM bug; kernel fix + open/close will clear it)");
        } else {
            dbg("REQBUFS(0) returned %d (%s) — may be normal", errno, strerror(errno));
        }
    } else {
        info("  vb2 queue is clean (REQBUFS succeeded)");
    }

    close(fd);
}

/* ------------------------------------------------------------------ */
/* Reset vdin1                                                        */
/* ------------------------------------------------------------------ */
static int reset_vdin1(void) {
    info("--- Resetting vdin1 ---");

    if (!device_exists(VDIN1_V4L2_DEV)) {
        warn("V4L2 device %s does not exist, skipping vdin1 reset", VDIN1_V4L2_DEV);
        return 0;
    }

    /* Step 1: Stop hardware via sysfs ("v4l2stop").
     * This clears VDIN_FLAG_DEC_STARTED and VDIN_FLAG_DEC_OPENED,
     * stops the ISR/DMA, and releases CMA buffers. */
    if (device_exists(VDIN1_SYSFS_ATTR)) {
        info("  Step 1: Sending v4l2stop via sysfs...");
        if (!dry_run) {
            if (sysfs_write(VDIN1_SYSFS_ATTR, "v4l2stop") == 0)
                info("    v4l2stop sent OK");
            else
                warn("    v4l2stop failed (may already be stopped)");
            /* Brief pause for hardware to settle */
            usleep(200000);
        }
    } else {
        warn("  sysfs attr %s not found, skipping v4l2stop", VDIN1_SYSFS_ATTR);
    }

    /* Step 2: Open+close the V4L2 device.
     * With the kernel fix (vdin_v4l2_release calls stop_tvin instead of
     * returning -EPERM), this triggers vb2_fop_release() which clears the
     * stale vb2 queue owner.
     * Without the kernel fix, this is a no-op but harmless. */
    info("  Step 2: Open+close %s to clear vb2 queue owner...", VDIN1_V4L2_DEV);
    if (!dry_run) {
        int fd = open(VDIN1_V4L2_DEV, O_RDWR);
        if (fd >= 0) {
            close(fd);
            info("    open+close OK");
        } else {
            warn("    open(%s) failed: %s", VDIN1_V4L2_DEV, strerror(errno));
            if (errno == EBUSY)
                warn("    Device still busy — kernel fix may not be applied");
        }
    }

    /* Step 3: Open+close the chardev as a fallback.
     * The chardev release (vdin_release in vdin_drv.c) has its own
     * cleanup path that properly handles DEC_STARTED. */
    if (device_exists(VDIN1_CHARDEV)) {
        info("  Step 3: Open+close %s (chardev fallback)...", VDIN1_CHARDEV);
        if (!dry_run) {
            int fd = open(VDIN1_CHARDEV, O_RDWR);
            if (fd >= 0) {
                close(fd);
                info("    chardev open+close OK");
            } else {
                dbg("    chardev open failed: %s (non-critical)", strerror(errno));
            }
        }
    }

    /* Verify */
    if (!dry_run) {
        usleep(200000);
        int fd = open(VDIN1_V4L2_DEV, O_RDWR);
        if (fd >= 0) {
            struct v4l2_requestbuffers reqbufs;
            memset(&reqbufs, 0, sizeof(reqbufs));
            reqbufs.count  = 0;
            reqbufs.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            reqbufs.memory = V4L2_MEMORY_DMABUF;
            if (ioctl(fd, VIDIOC_REQBUFS, &reqbufs) == 0) {
                info("  Verification: vdin1 vb2 queue is CLEAN");
            } else if (errno == EBUSY) {
                err("  Verification FAILED: vb2 queue still has stale owner");
                info("  -> Kernel fix for vdin_v4l2_release() is required");
            } else {
                dbg("  Verification: REQBUFS returned %d (%s)", errno, strerror(errno));
            }
            close(fd);
        } else {
            err("  Verification: cannot open %s: %s", VDIN1_V4L2_DEV, strerror(errno));
        }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Diagnose Wave521 encoder state                                     */
/* ------------------------------------------------------------------ */
static void diagnose_encoder(void) {
    info("--- Wave521 encoder diagnosis ---");

    if (!device_exists(ENCODER_DEV)) {
        warn("Encoder device %s does not exist", ENCODER_DEV);
        return;
    }

    char status[1024];
    if (sysfs_read(ENCODER_STATUS, status, sizeof(status)) == 0) {
        info("  Encoder status:\n%s", status);
    } else {
        warn("  Cannot read encoder status from %s", ENCODER_STATUS);
    }
}

/* ------------------------------------------------------------------ */
/* Reset Wave521 encoder                                              */
/* ------------------------------------------------------------------ */
static int reset_encoder(void) {
    info("--- Resetting Wave521 encoder ---");

    if (!device_exists(ENCODER_DEV)) {
        warn("Encoder device %s does not exist, skipping", ENCODER_DEV);
        return 0;
    }

    /* Read status before reset */
    char status_before[1024];
    if (sysfs_read(ENCODER_STATUS, status_before, sizeof(status_before)) == 0) {
        dbg("  Status before reset: %s", status_before);
    }

    if (dry_run) {
        info("  Would open + ioctl(VDI_IOCTL_RESET) + close %s", ENCODER_DEV);
        return 0;
    }

    /* Open the encoder device.
     * This increments open_count and powers up the encoder. */
    info("  Step 1: Opening %s...", ENCODER_DEV);
    int fd = open(ENCODER_DEV, O_RDWR);
    if (fd < 0) {
        err("  Cannot open %s: %s", ENCODER_DEV, strerror(errno));
        return -1;
    }
    info("    open OK");

    /* Issue hardware reset.
     * This only works if open_count <= 1 (i.e., we're the sole opener).
     * If another process has the encoder open, this will be rejected
     * by the multi-instance guard in vpu_hw_reset(). */
    info("  Step 2: Issuing VDI_IOCTL_RESET...");
    unsigned int reset_arg = 0;
    if (ioctl(fd, VDI_IOCTL_RESET, &reset_arg) == 0) {
        info("    Hardware reset OK");
    } else {
        if (errno == EBUSY || errno == EPERM) {
            warn("    Reset rejected — encoder may have other open sessions");
            info("    (close() will still trigger cleanup if we're the last fd)");
        } else {
            warn("    Reset ioctl returned error: %s", strerror(errno));
        }
    }

    /* Close the fd.
     * If this was the last open fd (open_count drops to 0), vpu_release()
     * will perform a full teardown: hw_reset + power off. */
    info("  Step 3: Closing %s...", ENCODER_DEV);
    close(fd);
    info("    close OK (triggers full teardown if last fd)");

    /* Brief pause for hardware to settle */
    usleep(200000);

    /* Read status after reset */
    char status_after[1024];
    if (sysfs_read(ENCODER_STATUS, status_after, sizeof(status_after)) == 0) {
        dbg("  Status after reset: %s", status_after);
        /* Check if open_count is 0 */
        if (strstr(status_after, "open_count=0") ||
            strstr(status_after, "open count: 0") ||
            strstr(status_after, "open_count: 0")) {
            info("  Verification: encoder open_count is 0 (clean)");
        } else {
            info("  Encoder status after reset: %s", status_after);
        }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
static void usage(const char *prog) {
    printf("Usage: %s [-v] [-n] [-k]\n", prog);
    printf("  -v   Verbose output\n");
    printf("  -n   Dry run (diagnose only, don't reset)\n");
    printf("  -k   Skip killing gst-launch-1.0 processes\n");
    printf("\n");
    printf("Resets vdin1 (%s) and Wave521 encoder (%s)\n",
           VDIN1_V4L2_DEV, ENCODER_DEV);
    printf("to a clean post-reboot-equivalent state.\n");
    printf("Must be run as root.\n");
}

int main(int argc, char *argv[]) {
    int skip_kill = 0;
    int opt;

    while ((opt = getopt(argc, argv, "vnkh")) != -1) {
        switch (opt) {
        case 'v': verbose = 1; break;
        case 'n': dry_run = 1; break;
        case 'k': skip_kill = 1; break;
        case 'h':
        default:
            usage(argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    /* Check for root */
    if (geteuid() != 0) {
        err("This tool must be run as root");
        return 1;
    }

    if (dry_run)
        info("=== DRY RUN MODE (no changes will be made) ===");

    info("=== Pipeline Reset Tool ===");

    /* Step 0: Kill any running GStreamer pipeline processes */
    if (!skip_kill) {
        int killed = kill_gstreamer_processes();
        if (killed > 0) {
            info("Killed %d gst-launch-1.0 process(es)", killed);
        } else {
            info("No gst-launch-1.0 processes found");
        }
    } else {
        info("Skipping gst-launch-1.0 kill (--skip-kill)");
    }

    /* Diagnose current state */
    diagnose_vdin1();
    diagnose_encoder();

    if (dry_run) {
        info("=== Dry run complete. Re-run without -n to perform reset. ===");
        return errors > 0 ? 1 : 0;
    }

    /* Perform resets */
    printf("\n");
    reset_vdin1();
    printf("\n");
    reset_encoder();

    /* Summary */
    printf("\n");
    if (errors == 0) {
        info("=== Reset complete. Pipeline devices should be ready. ===");
    } else {
        err("=== Reset completed with %d error(s). See above. ===", errors);
    }

    return errors > 0 ? 1 : 0;
}
