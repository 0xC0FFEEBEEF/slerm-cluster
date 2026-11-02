#!/usr/bin/env python3
"""randread.py FILE OFFSET LENGTH COUNT [BLOCK]: uncached (O_DIRECT) random reads inside one byte range, QD1.
Prints p50/p90/p99 latency and reads/s. Used 2026-10-17 to compare the n-gram table on NFS vs node04's /ssd."""
import mmap, os, random, sys, time

path, off, length, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
block = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
buf = mmap.mmap(-1, block)
rng = random.Random(1017)
lat = []
t_all = time.perf_counter()
for _ in range(count):
    pos = (off + rng.randrange(0, (length - block) // block) * block) // 4096 * 4096   # O_DIRECT needs 4K alignment
    t = time.perf_counter()
    os.preadv(fd, [buf], pos)
    lat.append((time.perf_counter() - t) * 1000)
total = time.perf_counter() - t_all
lat.sort()
q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
print("%s: %d x %d B reads, p50 %.3f ms, p90 %.3f ms, p99 %.3f ms, max %.1f ms, %.0f reads/s"
      % (path, count, block, q(0.5), q(0.9), q(0.99), lat[-1], count / total))
