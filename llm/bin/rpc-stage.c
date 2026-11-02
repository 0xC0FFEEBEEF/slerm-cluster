/* rpc-stage: pre-fill a ggml-rpc-server --cache directory from a GGUF file.
 *
 * ggml-rpc-server --cache stores every tensor over 10 MB it receives as
 * CACHE/<fnv-1a 64 of the bytes, %016llx>, and on the next load the client
 * sends only that hash. The bytes it receives are the tensor's bytes exactly
 * as they sit in the GGUF, so the same file can be written ahead of time from
 * the model file alone -- which is what staging does.
 *
 *   rpc-stage GGUF OUTDIR MBPS < "offset size" lines
 *
 * For each line: read the range, hash it (same fnv_hash as ggml-rpc.cpp),
 * write OUTDIR/<hash> via a temp file + rename, print "<offset> <hash>".
 * MBPS throttles the read (0 = unthrottled). Python drives it (bin/model_stage.py).
 * Build: gcc -O2 -o /data/llm/bin/rpc-stage rpc-stage.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CHUNK (8u << 20)

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: rpc-stage GGUF OUTDIR MBPS < offset-size lines\n");
        return 2;
    }
    const char *outdir = argv[2];
    double mbps = atof(argv[3]);
    int src = open(argv[1], O_RDONLY);
    if (src < 0) { perror(argv[1]); return 1; }
    unsigned char *buf = malloc(CHUNK);
    if (!buf) return 1;
    unsigned long long off, size;
    double t0 = now(), done = 0;
    char tmp[4096], dst[4096];
    snprintf(tmp, sizeof tmp, "%s/.stage-%d.part", outdir, (int)getpid());
    while (scanf("%llu %llu", &off, &size) == 2) {
        int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0) { perror(tmp); return 1; }
        uint64_t h = 0xcbf29ce484222325ULL;
        unsigned long long left = size, pos = off;
        while (left) {
            size_t want = left < CHUNK ? (size_t)left : CHUNK;
            ssize_t n = pread(src, buf, want, (off_t)pos);
            if (n <= 0) { fprintf(stderr, "short read at %llu: %s\n", pos, n ? strerror(errno) : "EOF"); unlink(tmp); return 1; }
            for (ssize_t i = 0; i < n; i++) { h ^= buf[i]; h *= 0x100000001b3ULL; }
            for (ssize_t w = 0; w < n;) {
                ssize_t k = write(out, buf + w, n - w);
                if (k < 0) { perror("write"); unlink(tmp); return 1; }
                w += k;
            }
            posix_fadvise(src, (off_t)pos, n, POSIX_FADV_DONTNEED);
            pos += n; left -= n; done += n;
            if (mbps > 0) {
                double ahead = done / (mbps * 1e6) - (now() - t0);
                if (ahead > 0) usleep((useconds_t)(ahead * 1e6));
            }
        }
        if (fsync(out) || close(out)) { perror("close"); unlink(tmp); return 1; }
        snprintf(dst, sizeof dst, "%s/%016" PRIx64, outdir, h);
        if (rename(tmp, dst)) { perror("rename"); unlink(tmp); return 1; }
        printf("%llu %016" PRIx64 "\n", off, h);
        fflush(stdout);
    }
    return 0;
}
