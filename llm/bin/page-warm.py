#!/usr/bin/env python3
"""Read files (or byte ranges of them) once, sequentially, so they sit in the page cache.

    page-warm.py FILE[:OFFSET:LENGTH] ...

Big sequential reads are what NFS from the head's disks is good at (~150-250 MB/s); the engines' own access pattern
(experts and n-gram rows in profile order) is not (~32 MB/s, 11 ms per 4 KiB miss). Pages already cached cost
nothing to read again. Prints one line per file with the rate. 2026-10-17, for the Strata jobs (jobs/strata.sh).
"""
import os
import sys
import time

CHUNK = 16 << 20


def warm(spec):
    path, off, length = spec, 0, None
    parts = spec.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        path, off, length = parts[0], int(parts[1]), int(parts[2])
    size = os.path.getsize(path)
    end = size if length is None else min(size, off + length)
    t0 = time.time()
    done = 0
    buf = bytearray(CHUNK)
    with open(path, "rb", buffering=0) as f:
        os.posix_fadvise(f.fileno(), off, end - off, os.POSIX_FADV_SEQUENTIAL)
        f.seek(off)
        pos = off
        while pos < end:
            n = f.readinto(memoryview(buf)[:min(CHUNK, end - pos)])
            if not n:
                break
            pos += n
            done += n
    dt = max(time.time() - t0, 1e-6)
    print("page-warm: %s [%d+%d] %.1f GB in %.0f s (%.0f MB/s)" % (
        os.path.basename(path), off, end - off, done / 1e9, dt, done / 1e6 / dt), flush=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for spec in sys.argv[1:]:
        try:
            warm(spec)
        except OSError as e:
            print("page-warm: %s: %s" % (spec, e), flush=True)


if __name__ == "__main__":
    main()
