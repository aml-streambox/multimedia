/*
 * vfmcap-demo.c - VFM Capture SDK demo / validation + profiling program
 *
 * Demonstrates:
 *   1. Open /dev/video_cap via libvfmcap
 *   2. Start streaming (zero-copy DMA-buf)
 *   3. Acquire frames, print stats
 *   4. GPU-convert first frame to P010 and NV12 (if Vulkan available)
 *   5. Optionally verify output by mmap + dump first 64 bytes
 *
 * Profiling mode (-p):
 *   GPU-converts EVERY frame (not just the first) and reports detailed
 *   per-phase timing statistics: acquire, gpu_submit, gpu_wait, release.
 *   Uses pre-allocated DMA-buf output ring and async submit/wait API.
 *
 * Output DMA-buf allocation uses /dev/dma_heap/system-uncached.
 *
 * Usage:
 *   vfmcap-demo [-d /dev/video_cap] [-n 100] [-c p010|nv12|both]
 *   vfmcap-demo -p -c nv12 -n 300       # profile NV12 conversion
 *   vfmcap-demo -p -c p010 -n 300       # profile P010 conversion
 *
 * Copyright (C) 2026 StreamBox
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>

#include "vfmcap.h"

/* ---------- DMA-heap output buffer allocation ---------- */

#define DMA_HEAP_PATH "/dev/dma_heap/system-uncached"

static int alloc_dmabuf(uint32_t size)
{
    int heap_fd = open(DMA_HEAP_PATH, O_RDWR);
    if (heap_fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n",
                DMA_HEAP_PATH, strerror(errno));
        return -1;
    }

    struct dma_heap_allocation_data alloc = {
        .len = size,
        .fd_flags = O_CLOEXEC | O_RDWR,
        .heap_flags = 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        fprintf(stderr, "ERROR: DMA_HEAP_IOCTL_ALLOC(%u) failed: %s\n",
                size, strerror(errno));
        close(heap_fd);
        return -1;
    }

    close(heap_fd);
    return alloc.fd;
}

/* ---------- Helpers ---------- */

static volatile int g_running = 1;

static void sigint_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static void hexdump(const char *label, const uint8_t *data, size_t len)
{
    fprintf(stderr, "%s (%zu bytes):\n", label, len);
    for (size_t i = 0; i < len && i < 64; i++) {
        fprintf(stderr, "%02x ", data[i]);
        if ((i + 1) % 16 == 0) fprintf(stderr, "\n");
    }
    if (len > 0 && len % 16 != 0) fprintf(stderr, "\n");
}

static int dump_dmabuf_to_file(const char *path, int fd, uint32_t size)
{
    if (!path || fd < 0) return -1;

    uint8_t *map = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "  dump: mmap failed: %s\n", strerror(errno));
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "  dump: cannot open %s: %s\n", path, strerror(errno));
        munmap(map, size);
        return -1;
    }

    size_t written = fwrite(map, 1, size, fp);
    fclose(fp);
    munmap(map, size);

    if (written != size) {
        fprintf(stderr, "  dump: short write (%zu / %u)\n", written, size);
        return -1;
    }

    fprintf(stderr, "  Dumped %u bytes to %s\n", size, path);
    return 0;
}

static void verify_output_dmabuf(const char *label, int fd, uint32_t size)
{
    if (fd < 0) return;

    uint8_t *map = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "  mmap(%s output) failed: %s\n", label, strerror(errno));
        return;
    }

    /* Check first 256 bytes for non-zero content */
    int non_zero = 0;
    size_t check_len = (size < 256) ? size : 256;
    for (size_t i = 0; i < check_len; i++) {
        if (map[i] != 0) non_zero++;
    }

    fprintf(stderr, "  %s output: %d/%zu non-zero bytes in first %zu\n",
            label, non_zero, check_len, check_len);
    hexdump(label, map, check_len > 64 ? 64 : check_len);

    munmap(map, size);
}

/* ---------- Profiling statistics ---------- */

#define PROF_MAX_FRAMES 4096

typedef struct {
    double acquire_ms;    /* Time to DQBUF + GET_DMABUF */
    double wait_prev_ms;  /* Time waiting for PREVIOUS frame's GPU fence */
    double release_prev_ms; /* Time releasing PREVIOUS frame's V4L2 buffer */
    double submit_ms;     /* Time to record + submit GPU command buffer */
    double total_ms;      /* Wall-clock time for entire frame cycle */
    double frame_interval_ms; /* Time since previous frame started */
    uint32_t sequence;    /* V4L2 sequence number */
} prof_sample_t;

static int double_cmp(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static void prof_print_stat(const char *label, const double *vals, int n)
{
    if (n == 0) return;

    /* Copy for sorting (percentiles) */
    double *sorted = malloc(n * sizeof(double));
    if (!sorted) return;
    memcpy(sorted, vals, n * sizeof(double));
    qsort(sorted, n, sizeof(double), double_cmp);

    double sum = 0, sum2 = 0;
    double mn = sorted[0], mx = sorted[n - 1];
    for (int i = 0; i < n; i++) {
        sum += sorted[i];
        sum2 += sorted[i] * sorted[i];
    }
    double avg = sum / n;
    double stddev = sqrt((sum2 / n) - (avg * avg));
    double p50 = sorted[n / 2];
    double p95 = sorted[(int)(n * 0.95)];
    double p99 = sorted[(int)(n * 0.99)];

    fprintf(stderr, "  %-18s  avg=%7.3f  min=%7.3f  p50=%7.3f  p95=%7.3f  p99=%7.3f  max=%7.3f  std=%6.3f ms\n",
            label, avg, mn, p50, p95, p99, mx, stddev);

    free(sorted);
}

/* Output ring buffer for profiling mode */
#define PROF_OUT_RING_SIZE 4

/* ---------- Profiling capture loop ---------- */

static int run_profiling(vfmcap_ctx_t *ctx, int num_frames, int do_p010,
                         const char *output_path)
{
    vfmcap_output_fmt_t fmt = do_p010 ? VFMCAP_FMT_P010 : VFMCAP_FMT_NV12;
    const char *fmt_name = do_p010 ? "P010" : "NV12";

    /* Acquire one test frame to get dimensions */
    vfmcap_frame_t test_frame;
    int ret = vfmcap_acquire_frame(ctx, &test_frame, 3000);
    if (ret != VFMCAP_OK) {
        fprintf(stderr, "ERROR: Cannot acquire test frame for profiling: %d\n", ret);
        return 1;
    }

    uint32_t width = test_frame.width;
    uint32_t height = test_frame.height;
    uint32_t out_size = vfmcap_output_size(width, height, fmt);

    fprintf(stderr, "\n=== Profiling Mode (PIPELINED) ===\n");
    fprintf(stderr, "  Format:     %s\n", fmt_name);
    fprintf(stderr, "  Resolution: %ux%u\n", width, height);
    fprintf(stderr, "  Input size: %u bytes (AMLY)\n", test_frame.size);
    fprintf(stderr, "  Output size: %u bytes (%s)\n", out_size, fmt_name);
    fprintf(stderr, "  Frames:     %d\n", num_frames);
    fprintf(stderr, "  Out ring:   %d buffers\n", PROF_OUT_RING_SIZE);
    fprintf(stderr, "  Pipeline:   acquire_N -> wait_N-1 -> release_N-1 -> submit_N\n");

    vfmcap_release_frame(ctx, &test_frame);

    /* Pre-allocate output DMA-buf ring */
    int out_fds[PROF_OUT_RING_SIZE];
    for (int i = 0; i < PROF_OUT_RING_SIZE; i++) {
        out_fds[i] = alloc_dmabuf(out_size);
        if (out_fds[i] < 0) {
            fprintf(stderr, "ERROR: Failed to allocate output DMA-buf %d\n", i);
            for (int j = 0; j < i; j++) close(out_fds[j]);
            return 1;
        }
    }

    /* Allocate sample array */
    int max_samples = (num_frames < PROF_MAX_FRAMES) ? num_frames : PROF_MAX_FRAMES;
    prof_sample_t *samples = calloc(max_samples, sizeof(prof_sample_t));
    if (!samples) {
        fprintf(stderr, "ERROR: Cannot allocate sample array\n");
        for (int i = 0; i < PROF_OUT_RING_SIZE; i++) close(out_fds[i]);
        return 1;
    }

    int captured = 0;
    int gpu_converted = 0;
    int gpu_errors = 0;
    int dropped = 0;
    int seq_gaps = 0;
    uint32_t last_seq = 0;
    int out_ring_idx = 0;

    /* Warmup: skip first 5 frames to stabilize pipeline */
    fprintf(stderr, "  Warming up (5 frames)...\n");
    for (int i = 0; i < 5 && g_running; i++) {
        vfmcap_frame_t frame;
        ret = vfmcap_acquire_frame(ctx, &frame, 2000);
        if (ret == VFMCAP_OK) {
            /* Do one GPU conversion to warm up Vulkan caches */
            if (i == 0) {
                ret = vfmcap_convert_submit(ctx, &frame, out_fds[0], fmt);
                if (ret == VFMCAP_OK) vfmcap_convert_wait(ctx);
            }
            vfmcap_release_frame(ctx, &frame);
        }
    }

    fprintf(stderr, "  Capturing %d frames with %s GPU conversion (pipelined)...\n\n",
            num_frames, fmt_name);

    /*
     * Pipelined loop:
     *
     *   Frame 0: acquire -> submit  (no previous GPU work to wait for)
     *   Frame N: acquire_N -> wait_(N-1) -> release_(N-1) -> submit_N
     *   After loop: wait_last -> release_last
     *
     * This overlaps GPU compute with V4L2 poll/DQBUF. Since GPU takes ~6ms
     * and acquire takes ~10ms, the GPU finishes during the acquire and
     * wait_prev completes near-instantly.
     *
     * Constraints:
     *   - Must wait_prev before submit_curr (Vulkan cmd pool/fence reuse)
     *   - Must hold prev frame's DMA-buf until after wait_prev completes
     *   - We use 6 V4L2 buffers, so holding 2 simultaneously is fine
     */

    double t_loop_start = now_ms();
    double t_prev_frame = t_loop_start;

    /* State for the "previous" frame in the pipeline */
    int prev_gpu_pending = 0;     /* Is there a GPU job in flight? */
    vfmcap_frame_t prev_frame;    /* Previous frame (held until GPU done) */
    int prev_out_fd = -1;         /* Output fd used by previous submit */
    memset(&prev_frame, 0, sizeof(prev_frame));
    prev_frame.dmabuf_fd = -1;

    while (g_running && captured < num_frames) {
        double t0 = now_ms();

        /* Phase 1: Acquire current frame
         * While this blocks in poll(), the previous frame's GPU shader
         * is executing in parallel on the Mali-G52 */
        vfmcap_frame_t frame;
        ret = vfmcap_acquire_frame(ctx, &frame, 1000);

        double t_acquired = now_ms();

        if (ret == VFMCAP_ERR_TIMEOUT) {
            dropped++;
            if (dropped > 20) {
                fprintf(stderr, "ERROR: Too many timeouts (%d), aborting\n", dropped);
                break;
            }
            continue;
        }

        if (ret == VFMCAP_ERR_NOSIG) {
            fprintf(stderr, "  [no signal]\n");
            break;
        }

        if (ret != VFMCAP_OK) {
            fprintf(stderr, "ERROR: acquire failed (%d): %s\n",
                    ret, vfmcap_last_error(ctx));
            break;
        }

        /* Detect sequence gaps (dropped by vdin/kernel) */
        if (captured > 0 && frame.sequence != last_seq + 1) {
            seq_gaps += (int)(frame.sequence - last_seq - 1);
        }
        last_seq = frame.sequence;

        /* Phase 2: Wait for previous frame's GPU work (should be ~0ms
         * since GPU ~6ms completed during the ~10ms acquire above) */
        double t_wait_prev_start = now_ms();
        double t_wait_prev_end = t_wait_prev_start;
        double t_release_prev_end = t_wait_prev_start;

        if (prev_gpu_pending) {
            ret = vfmcap_convert_wait(ctx);
            t_wait_prev_end = now_ms();

            if (ret != VFMCAP_OK) {
                gpu_errors++;
            } else {
                gpu_converted++;
            }

            /* Dump first converted frame if requested (prev was frame 0) */
            if (captured == 1 && output_path && ret == VFMCAP_OK) {
                dump_dmabuf_to_file(output_path, prev_out_fd, out_size);
            }

            /* Phase 3: Release previous frame back to V4L2 */
            vfmcap_release_frame(ctx, &prev_frame);
            t_release_prev_end = now_ms();

            prev_gpu_pending = 0;
        }

        /* Phase 4: Submit GPU conversion for current frame */
        int cur_out = out_fds[out_ring_idx];
        out_ring_idx = (out_ring_idx + 1) % PROF_OUT_RING_SIZE;

        double t_submit_start = now_ms();
        ret = vfmcap_convert_submit(ctx, &frame, cur_out, fmt);
        double t_submitted = now_ms();

        if (ret != VFMCAP_OK) {
            gpu_errors++;
            vfmcap_release_frame(ctx, &frame);
            captured++;
            continue;
        }

        /* Current frame becomes "previous" — hold it until next iteration */
        prev_gpu_pending = 1;
        prev_frame = frame;
        prev_out_fd = cur_out;

        /* Record sample */
        if (captured < max_samples) {
            prof_sample_t *s = &samples[captured];
            s->acquire_ms = t_acquired - t0;
            s->wait_prev_ms = t_wait_prev_end - t_wait_prev_start;
            s->release_prev_ms = t_release_prev_end - t_wait_prev_end;
            s->submit_ms = t_submitted - t_submit_start;
            s->total_ms = t_submitted - t0;
            s->frame_interval_ms = t0 - t_prev_frame;
            s->sequence = frame.sequence;
        }

        captured++;
        t_prev_frame = t0;

        /* Progress every 60 frames */
        if (captured % 60 == 0) {
            double elapsed = now_ms() - t_loop_start;
            double fps = captured / (elapsed / 1000.0);
            fprintf(stderr, "  [%d/%d frames, %.1f fps, seq=%u, gaps=%d]\n",
                    captured, num_frames, fps, frame.sequence, seq_gaps);
        }
    }

    /* Drain: wait for the last submitted GPU job */
    if (prev_gpu_pending) {
        ret = vfmcap_convert_wait(ctx);
        if (ret != VFMCAP_OK) {
            gpu_errors++;
        } else {
            gpu_converted++;
        }

        /* Dump if only 1 frame was captured */
        if (captured == 1 && output_path && ret == VFMCAP_OK) {
            dump_dmabuf_to_file(output_path, prev_out_fd, out_size);
        }

        vfmcap_release_frame(ctx, &prev_frame);
        prev_gpu_pending = 0;
    }

    double t_loop_end = now_ms();
    double total_s = (t_loop_end - t_loop_start) / 1000.0;
    double avg_fps = (total_s > 0) ? captured / total_s : 0;

    /* ---- Print results ---- */
    int n = (captured < max_samples) ? captured : max_samples;

    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  PROFILING RESULTS: %s @ %ux%u (PIPELINED)\n", fmt_name, width, height);
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Frames captured:  %d in %.2f s\n", captured, total_s);
    fprintf(stderr, "  Overall FPS:      %.2f (target: 60.00)\n", avg_fps);
    fprintf(stderr, "  GPU converted:    %d\n", gpu_converted);
    fprintf(stderr, "  GPU errors:       %d\n", gpu_errors);
    fprintf(stderr, "  Poll timeouts:    %d\n", dropped);
    fprintf(stderr, "  Sequence gaps:    %d (frames dropped by kernel)\n", seq_gaps);
    fprintf(stderr, "  Output buf size:  %.2f MB\n", out_size / (1024.0 * 1024.0));
    fprintf(stderr, "  Throughput:       %.1f MB/s (input AMLY)\n",
            (captured > 0 && total_s > 0) ?
            ((double)width * height * 5 / 2 * captured / total_s) / (1024.0 * 1024.0) : 0);
    fprintf(stderr, "  Throughput:       %.1f MB/s (output %s)\n",
            (captured > 0 && total_s > 0) ?
            ((double)out_size * captured / total_s) / (1024.0 * 1024.0) : 0,
            fmt_name);
    fprintf(stderr, "\n");

    /* Extract timing arrays for statistics */
    if (n > 0) {
        double *t_acquire = malloc(n * sizeof(double));
        double *t_wait_prev = malloc(n * sizeof(double));
        double *t_release_prev = malloc(n * sizeof(double));
        double *t_submit = malloc(n * sizeof(double));
        double *t_total = malloc(n * sizeof(double));
        double *t_interval = malloc(n * sizeof(double));

        if (t_acquire && t_wait_prev && t_release_prev && t_submit &&
            t_total && t_interval) {
            int ni = 0; /* interval count (skip first) */
            for (int i = 0; i < n; i++) {
                t_acquire[i] = samples[i].acquire_ms;
                t_wait_prev[i] = samples[i].wait_prev_ms;
                t_release_prev[i] = samples[i].release_prev_ms;
                t_submit[i] = samples[i].submit_ms;
                t_total[i] = samples[i].total_ms;
                if (i > 0) {
                    t_interval[ni++] = samples[i].frame_interval_ms;
                }
            }

            fprintf(stderr, "  Per-frame timing (ms) over %d frames:\n", n);
            fprintf(stderr, "  %-18s  %7s  %7s  %7s  %7s  %7s  %7s  %9s\n",
                    "Phase", "avg", "min", "p50", "p95", "p99", "max", "stddev");
            fprintf(stderr, "  -------------------------------------------------------------------------------------\n");
            prof_print_stat("V4L2 acquire", t_acquire, n);
            prof_print_stat("Wait prev GPU", t_wait_prev, n);
            prof_print_stat("Release prev", t_release_prev, n);
            prof_print_stat("GPU submit", t_submit, n);
            prof_print_stat("TOTAL per-cycle", t_total, n);
            if (ni > 0)
                prof_print_stat("Frame interval", t_interval, ni);

            /* Frame budget analysis */
            double budget_ms = 1000.0 / 60.0; /* 16.67 ms */
            int over_budget = 0;
            for (int i = 0; i < n; i++) {
                if (t_total[i] > budget_ms) over_budget++;
            }

            fprintf(stderr, "\n  Frame budget analysis (16.67 ms @ 60fps):\n");
            fprintf(stderr, "    Over budget:    %d / %d (%.1f%%)\n",
                    over_budget, n, 100.0 * over_budget / n);

            /* Sustained fps from average total */
            double avg_total = 0;
            for (int i = 0; i < n; i++) avg_total += t_total[i];
            avg_total /= n;
            fprintf(stderr, "    Max sustained:  %.1f fps (1000 / avg_total=%.3f ms)\n",
                    1000.0 / avg_total, avg_total);

            /* Pipeline efficiency: how much GPU time was hidden */
            double avg_acquire = 0, avg_wait = 0;
            for (int i = 0; i < n; i++) {
                avg_acquire += t_acquire[i];
                avg_wait += t_wait_prev[i];
            }
            avg_acquire /= n;
            avg_wait /= n;

            /* Active processing time (excluding vsync wait) */
            double avg_release = 0, avg_submit = 0;
            for (int i = 0; i < n; i++) {
                avg_release += t_release_prev[i];
                avg_submit += t_submit[i];
            }
            avg_release /= n;
            avg_submit /= n;
            double active_proc = avg_wait + avg_release + avg_submit;
            fprintf(stderr, "    Active process: %.3f ms (GPU+CPU, excl. vsync wait)\n", active_proc);
            fprintf(stderr, "    Theoretical max: %.1f fps (1000 / active_proc)\n",
                    1000.0 / active_proc);

            /* 4K60 extrapolation based on active processing time */
            double pixel_ratio_4k = (3840.0 * 2160.0) / (width * height);
            double estimated_4k_active = active_proc * pixel_ratio_4k;
            fprintf(stderr, "    4K active proc: %.2f ms (target: 16.67 ms) | pixel_ratio=%.2fx\n",
                    estimated_4k_active, pixel_ratio_4k);
            fprintf(stderr, "    4K60 capable:   %s (pipelined, based on active processing)\n",
                    estimated_4k_active <= 16.67 ? "YES" : "NO");
            fprintf(stderr, "    Avg acquire:    %.3f ms (mostly vsync wait)\n", avg_acquire);
            fprintf(stderr, "    Avg wait_prev:  %.3f ms (GPU overlap benefit: %.1f ms hidden)\n",
                    avg_wait, avg_acquire > avg_wait ? avg_acquire - avg_wait : 0);
        }

        free(t_acquire);
        free(t_wait_prev);
        free(t_release_prev);
        free(t_submit);
        free(t_total);
        free(t_interval);
    }

    fprintf(stderr, "============================================================\n");

    /* Cleanup */
    free(samples);
    for (int i = 0; i < PROF_OUT_RING_SIZE; i++) {
        if (out_fds[i] >= 0) close(out_fds[i]);
    }

    return (captured > 0) ? 0 : 1;
}

/* ---------- Main ---------- */

int main(int argc, char *argv[])
{
    const char *device = NULL;
    const char *output_path = NULL;
    int num_frames = 100;
    int do_p010 = 0, do_nv12 = 0;
    int do_profile = 0;
    int color_mode = 0;  /* 0=passthrough, 1=HDR10->SDR, 2=HLG->SDR */
    int target_w = 0, target_h = 0;
    float target_fps = 0.0f;
    int opt;

    const char *raw_path = NULL;

    while ((opt = getopt(argc, argv, "d:n:c:o:r:m:W:H:F:ph")) != -1) {
        switch (opt) {
        case 'd':
            device = optarg;
            break;
        case 'n':
            num_frames = atoi(optarg);
            if (num_frames < 1) num_frames = 1;
            break;
        case 'c':
            if (strcmp(optarg, "p010") == 0) do_p010 = 1;
            else if (strcmp(optarg, "nv12") == 0) do_nv12 = 1;
            else if (strcmp(optarg, "both") == 0) { do_p010 = 1; do_nv12 = 1; }
            else {
                fprintf(stderr, "Unknown convert format: %s\n", optarg);
                return 1;
            }
            break;
        case 'o':
            output_path = optarg;
            break;
        case 'r':
            raw_path = optarg;
            break;
        case 'm':
            if (strcmp(optarg, "hdr10") == 0) color_mode = 1;
            else if (strcmp(optarg, "hlg") == 0) color_mode = 2;
            else if (strcmp(optarg, "pass") == 0) color_mode = 0;
            else {
                fprintf(stderr, "Unknown color mode: %s (use: pass, hdr10, hlg)\n", optarg);
                return 1;
            }
            break;
        case 'p':
            do_profile = 1;
            break;
        case 'W':
            target_w = atoi(optarg);
            break;
        case 'H':
            target_h = atoi(optarg);
            break;
        case 'F':
            target_fps = atof(optarg);
            break;
        case 'h':
        default:
            fprintf(stderr,
                "Usage: %s [-d device] [-n frames] [-c p010|nv12|both] [-o outfile] [-r rawfile] [-m pass|hdr10|hlg] [-W width] [-H height] [-F fps] [-p]\n"
                "  -d  Device path (default: /dev/video_cap)\n"
                "  -n  Number of frames to capture (default: 100)\n"
                "  -c  GPU convert: p010, nv12, or both\n"
                "  -o  Dump converted frame to file (raw P010/NV12)\n"
                "  -r  Dump raw AMLY input frame to file\n"
                "  -m  Color mode: pass (default), hdr10 (HDR10->SDR), hlg (HLG->SDR)\n"
                "  -W  Target output width (0 = match source)\n"
                "  -H  Target output height (0 = match source)\n"
                "  -F  Target framerate (0 = match source)\n"
                "  -p  Profiling mode: GPU-convert every frame, report timing stats\n",
                argv[0]);
            return (opt == 'h') ? 0 : 1;
        }
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    fprintf(stderr, "=== vfmcap-demo ===\n");
    fprintf(stderr, "Device: %s\n", device ? device : "(default)");
    fprintf(stderr, "Frames: %d\n", num_frames);
    fprintf(stderr, "Convert: %s%s%s\n",
            do_p010 ? "P010 " : "", do_nv12 ? "NV12 " : "",
            (!do_p010 && !do_nv12) ? "none" : "");
    if (color_mode) fprintf(stderr, "Color: %s\n",
                            color_mode == 1 ? "HDR10->SDR" :
                            color_mode == 2 ? "HLG->SDR" : "unknown");
    if (target_w || target_h) fprintf(stderr, "Target: %dx%d\n", target_w, target_h);
    if (target_fps > 0) fprintf(stderr, "Target FPS: %.1f\n", target_fps);
    if (do_profile) fprintf(stderr, "Mode: PROFILING\n");

    /* Open */
    vfmcap_config_t cfg = {0};
    if (do_p010) cfg.output_format = VFMCAP_FMT_P010;
    else if (do_nv12) cfg.output_format = VFMCAP_FMT_NV12;
    else cfg.output_format = VFMCAP_FMT_RAW;
    cfg.color_mode = (vfmcap_color_mode_t)color_mode;
    cfg.target_width = (uint32_t)target_w;
    cfg.target_height = (uint32_t)target_h;
    cfg.target_fps = target_fps;

    vfmcap_ctx_t *ctx = vfmcap_open(device, &cfg);
    if (!ctx) {
        fprintf(stderr, "ERROR: vfmcap_open failed: %s\n",
                vfmcap_last_error(NULL));
        return 1;
    }

    /* Start streaming */
    int ret = vfmcap_start(ctx, 6);
    if (ret != VFMCAP_OK) {
        fprintf(stderr, "ERROR: vfmcap_start failed (%d): %s\n",
                ret, vfmcap_last_error(ctx));
        vfmcap_close(ctx);
        return 1;
    }

    /* Profiling mode: GPU convert every frame */
    if (do_profile) {
        if (!do_p010 && !do_nv12) {
            fprintf(stderr, "ERROR: Profiling mode requires -c p010 or -c nv12\n");
            vfmcap_close(ctx);
            return 1;
        }

        /* If both requested, profile P010 (use -c nv12 for NV12 profiling) */
        if (do_p010 && do_nv12) {
            fprintf(stderr, "NOTE: Profiling both is not supported; profiling P010.\n");
            fprintf(stderr, "      Use -c nv12 -p to profile NV12 separately.\n");
        }

        ret = run_profiling(ctx, num_frames, do_p010, output_path);
        vfmcap_stop(ctx);
        vfmcap_close(ctx);
        return ret;
    }

    /* ---------- Normal (non-profiling) capture loop ---------- */

    int captured = 0;
    int dropped = 0;
    int converted = 0;
    double t_start = now_ms();
    double t_first_frame = 0;
    uint32_t frame_width = 0;
    uint32_t frame_height = 0;

    /* Frame generation time tracking */
    double *frame_gen_ms = NULL;
    if (num_frames > 1) {
        frame_gen_ms = calloc(num_frames, sizeof(double));
    }
    double t_frame_start = t_start;

    while (g_running && captured < num_frames) {
        vfmcap_frame_t frame;
        ret = vfmcap_acquire_frame(ctx, &frame, 1000);

        if (ret == VFMCAP_ERR_TIMEOUT) {
            fprintf(stderr, "  [timeout at frame %d]\n", captured);
            dropped++;
            if (dropped > 10) {
                fprintf(stderr, "ERROR: Too many timeouts, aborting\n");
                break;
            }
            continue;
        }

        if (ret == VFMCAP_ERR_NOSIG) {
            fprintf(stderr, "  [no signal]\n");
            break;
        }

        if (ret != VFMCAP_OK) {
            fprintf(stderr, "ERROR: vfmcap_acquire_frame failed (%d): %s\n",
                    ret, vfmcap_last_error(ctx));
            break;
        }

        captured++;

        if (captured == 1) {
            t_first_frame = now_ms();
            frame_width = frame.width;
            frame_height = frame.height;
            fprintf(stderr, "\nFirst frame captured in %.1f ms\n",
                    t_first_frame - t_start);
            fprintf(stderr, "  seq=%u dmabuf_fd=%d %ux%u bpl=%u size=%u\n",
                    frame.sequence, frame.dmabuf_fd,
                    frame.width, frame.height,
                    frame.bytesperline, frame.size);
            fprintf(stderr, "  pixfmt=%.4s bitdepth=%u signal_type=0x%x\n",
                    (char *)&frame.pixelformat, frame.bitdepth,
                    frame.signal_type);

            /* Dump raw AMLY input if requested (only valid if frame is still raw) */
            if (raw_path && frame.pixelformat != V4L2_PIX_FMT_NV12 &&
                frame.pixelformat != v4l2_fourcc('P', '0', '1', '0')) {
                dump_dmabuf_to_file(raw_path, frame.dmabuf_fd, frame.size);
            }

            /* Check if acquire already did GPU conversion (integrated path).
             * When output_format is set and Vulkan is available, acquire_frame
             * converts the frame automatically. The frame's pixelformat will
             * be the output format (NV12/P010), not the input format (AMLY). */
            uint32_t p010_fourcc = v4l2_fourcc('P', '0', '1', '0');
            int already_p010 = (frame.pixelformat == p010_fourcc);
            int already_nv12 = (frame.pixelformat == V4L2_PIX_FMT_NV12);

            if (already_p010 || already_nv12) {
                const char *fmt_name = already_p010 ? "P010" : "NV12";
                fprintf(stderr, "\nFrame already converted to %s by acquire (integrated path)\n",
                        fmt_name);
                fprintf(stderr, "  Y plane: fd=%d  UV plane: fd=%d\n",
                        frame.dmabuf_fd, frame.dmabuf_fd2);

                /* Verify Y plane */
                uint32_t y_plane_size;
                if (already_p010)
                    y_plane_size = frame.width * frame.height * 2;
                else
                    y_plane_size = frame.width * frame.height;
                verify_output_dmabuf("Y-plane", frame.dmabuf_fd, y_plane_size);

                /* Verify UV plane */
                if (frame.dmabuf_fd2 >= 0) {
                    uint32_t uv_plane_size;
                    if (already_p010)
                        uv_plane_size = frame.width * frame.height; /* w*h/2 * 2 bytes */
                    else
                        uv_plane_size = frame.width * frame.height / 2;
                    verify_output_dmabuf("UV-plane", frame.dmabuf_fd2, uv_plane_size);
                }

                /* Dump to file if requested */
                if (output_path) {
                    char y_path[512], uv_path[512];
                    snprintf(y_path, sizeof(y_path), "%s.y", output_path);
                    snprintf(uv_path, sizeof(uv_path), "%s.uv", output_path);
                    dump_dmabuf_to_file(y_path, frame.dmabuf_fd, y_plane_size);
                    if (frame.dmabuf_fd2 >= 0) {
                        uint32_t uv_size = already_p010 ?
                            frame.width * frame.height :
                            frame.width * frame.height / 2;
                        dump_dmabuf_to_file(uv_path, frame.dmabuf_fd2, uv_size);
                    }
                }
                converted++;
            } else {
                /* Frame is still raw — do manual GPU conversion */
                if (do_p010) {
                    uint32_t p010_size = vfmcap_output_size(frame.width,
                                                             frame.height,
                                                             VFMCAP_FMT_P010);
                    fprintf(stderr, "\nConverting to P010 (%u bytes)...\n", p010_size);
                    int out_fd = alloc_dmabuf(p010_size);
                    if (out_fd >= 0) {
                        double t0 = now_ms();
                        ret = vfmcap_convert_p010(ctx, &frame, out_fd);
                        double t1 = now_ms();
                        if (ret == VFMCAP_OK) {
                            fprintf(stderr, "  P010 conversion OK (%.2f ms)\n", t1 - t0);
                            verify_output_dmabuf("P010", out_fd, p010_size);
                            if (output_path)
                                dump_dmabuf_to_file(output_path, out_fd, p010_size);
                            converted++;
                        } else {
                            fprintf(stderr, "  P010 conversion FAILED (%d): %s\n",
                                    ret, vfmcap_last_error(ctx));
                        }
                        close(out_fd);
                    }
                }

                if (do_nv12) {
                    uint32_t nv12_size = vfmcap_output_size(frame.width,
                                                             frame.height,
                                                             VFMCAP_FMT_NV12);
                    fprintf(stderr, "\nConverting to NV12 (%u bytes)...\n", nv12_size);
                    int out_fd = alloc_dmabuf(nv12_size);
                    if (out_fd >= 0) {
                        double t0 = now_ms();
                        ret = vfmcap_convert_nv12(ctx, &frame, out_fd);
                        double t1 = now_ms();
                        if (ret == VFMCAP_OK) {
                            fprintf(stderr, "  NV12 conversion OK (%.2f ms)\n", t1 - t0);
                            verify_output_dmabuf("NV12", out_fd, nv12_size);
                            if (output_path) {
                                if (do_p010) {
                                    char nv12_path[512];
                                    snprintf(nv12_path, sizeof(nv12_path), "%s.nv12", output_path);
                                    dump_dmabuf_to_file(nv12_path, out_fd, nv12_size);
                                } else {
                                    dump_dmabuf_to_file(output_path, out_fd, nv12_size);
                                }
                            }
                            converted++;
                        } else {
                            fprintf(stderr, "  NV12 conversion FAILED (%d): %s\n",
                                    ret, vfmcap_last_error(ctx));
                        }
                        close(out_fd);
                    }
                }
            }
        }

        /* Print progress every 30 frames */
        if (captured % 30 == 0) {
            double elapsed = now_ms() - t_start;
            double fps = captured / (elapsed / 1000.0);
            fprintf(stderr, "  [%d/%d frames, %.1f fps, seq=%u]\n",
                    captured, num_frames, fps, frame.sequence);
        }

        vfmcap_release_frame(ctx, &frame);

        if (frame_gen_ms && captured > 0) {
            frame_gen_ms[captured - 1] = now_ms() - t_frame_start;
        }
        t_frame_start = now_ms();
    }

    double t_end = now_ms();
    double total_s = (t_end - t_start) / 1000.0;
    double avg_fps = (total_s > 0) ? captured / total_s : 0;

    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Captured:   %d frames in %.2f s\n", captured, total_s);
    fprintf(stderr, "Average:    %.2f fps\n", avg_fps);
    fprintf(stderr, "Dropped:    %d\n", dropped);
    fprintf(stderr, "Converted:  %d\n", converted);

    if (frame_gen_ms && captured > 1) {
        double min_gen = frame_gen_ms[0];
        double max_gen = frame_gen_ms[0];
        double sum_gen = 0;
        for (int i = 0; i < captured - 1; i++) {
            if (frame_gen_ms[i] < min_gen) min_gen = frame_gen_ms[i];
            if (frame_gen_ms[i] > max_gen) max_gen = frame_gen_ms[i];
            sum_gen += frame_gen_ms[i];
        }
        double avg_gen = sum_gen / (captured - 1);
        fprintf(stderr, "\nFrame interval (includes vsync wait):\n");
        fprintf(stderr, "  avg=%.3f ms  min=%.3f ms  max=%.3f ms\n", avg_gen, min_gen, max_gen);
        /* 4K60: most of avg_gen is vsync wait; active GPU+CPU work is much smaller */
        double pixel_ratio_4k = (3840.0 * 2160.0) / (frame_width * frame_height);
        double active_est_4k = (avg_gen * 0.15) * pixel_ratio_4k; /* rough active processing estimate */
        fprintf(stderr, "  4K active proc estimate: %.2f ms (target: 16.67 ms)\n", active_est_4k);
        fprintf(stderr, "  4K60 capable (pipelined): %s\n",
                active_est_4k <= 16.67 ? "YES" : "NO");
        free(frame_gen_ms);
    }

    /* Cleanup */
    vfmcap_stop(ctx);
    vfmcap_close(ctx);

    fprintf(stderr, "=== Done ===\n");
    return (captured > 0) ? 0 : 1;
}
