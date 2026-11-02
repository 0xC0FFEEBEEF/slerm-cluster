#!/usr/bin/env python3
"""Publish this node GPU telemetry to NFS while a server job is running.

The panel cannot ssh to the compute nodes, and slurmapi.gpu_probe costs a real
job slot, so it can never be polled. A long-running server job is already
sitting on the GPU, so it publishes nvidia-smi here for free and the panel just
reads the file. Readers MUST treat a stale file as absent: nothing removes it
when the job dies, because llama-server is started with exec and no trap runs.

Writes /data/llm/run/gpu-<shorthost>.json atomically (unique temp + rename), so
a reader never sees a half-written file.
"""
import json, os, socket, subprocess, sys, time

RUN = "/data/llm/run"
INTERVAL = int(os.environ.get("LLM_GPU_PUBLISH_INTERVAL", "5"))
QUERY = ("name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,fan.speed,"
         "power.limit,pcie.link.gen.current,pcie.link.width.current")


def num(v, cast=int):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def sample():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + QUERY, "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        gpus.append({
            "name": p[0],
            "temp_c": num(p[1]),
            "util_pct": num(p[2]),
            "mem_used_mb": num(p[3]),
            "mem_total_mb": num(p[4]),
            "power_w": num(p[5], float) if len(p) > 5 else None,
            "fan_pct": num(p[6]) if len(p) > 6 else None,   # "[N/A]" -> None
            "power_limit_w": num(p[7], float) if len(p) > 7 else None,
            "pcie_gen": num(p[8]) if len(p) > 8 else None,
            "pcie_width": num(p[9]) if len(p) > 9 else None,
        })
    return gpus


# PCIe throughput per card (2026-10-21, user: "missing pcie to gpu speed"): NVML through ctypes, no package needed.
# Each reading covers 20 ms, so three are averaged; in nvidia-smi's (PCI bus) order. MB/s; rx = host -> GPU.
_nvml = {}


def pcie_rates(count, samples=3):
    if "lib" not in _nvml:
        import ctypes
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            _nvml["lib"] = lib if lib.nvmlInit_v2() == 0 else None
        except OSError:
            _nvml["lib"] = None
        _nvml["c"] = ctypes
    lib, c = _nvml["lib"], _nvml["c"]
    if lib is None:
        return []
    out = []
    for i in range(count):
        h = c.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(i, c.byref(h)) != 0:
            out.append(None)
            continue
        v, tot = c.c_uint(), [0, 0]
        for _ in range(samples):
            for k in (0, 1):                                        # 0 = TX (GPU -> host), 1 = RX (host -> GPU), KB/s
                tot[k] += v.value if lib.nvmlDeviceGetPcieThroughput(h, k, c.byref(v)) == 0 else 0
        out.append((round(tot[1] / samples / 1000.0, 1), round(tot[0] / samples / 1000.0, 1)))
    return out


# Node counters for the panel's Strata monitor and cluster tab (2026-10-21): CPU busy %, RAM, and disk / network
# throughput as rates between two samples. Whole disks only (sd*, vd*, nvme*n*), so partitions and device-mapper
# volumes are not counted twice; network = every interface but lo.
_prev = {}


def _cpu():
    f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(f), f[3] + (f[4] if len(f) > 4 else 0)          # total, idle + iowait


def _disks():
    rd = wr = 0
    for line in open("/proc/diskstats"):
        p = line.split()
        dev = p[2]
        if not (dev.startswith(("sd", "vd")) and dev[-1].isalpha() or dev.startswith("nvme") and "p" not in dev[5:]):
            continue
        rd += int(p[5]); wr += int(p[9])                            # 512-byte sectors
    return rd * 512, wr * 512


def _net():
    rx = tx = 0
    for line in open("/proc/net/dev").readlines()[2:]:
        name, data = line.split(":", 1)
        if name.strip() == "lo":
            continue
        d = data.split()
        rx += int(d[0]); tx += int(d[8])
    return rx, tx


def host_sample():
    now = time.time()
    cur = {"t": now, "cpu": _cpu(), "disk": _disks(), "net": _net()}
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        mem[k] = int(v.split()[0]) // 1024                         # MiB
    out = {"mem_total_mb": mem.get("MemTotal"),
           "mem_used_mb": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
           "cache_mb": mem.get("Cached"), "cpus": os.cpu_count()}
    prev = _prev.get("s")
    if prev:
        dt = max(0.001, now - prev["t"])
        tot, idle = cur["cpu"][0] - prev["cpu"][0], cur["cpu"][1] - prev["cpu"][1]
        out["cpu_pct"] = round(100.0 * (tot - idle) / tot, 1) if tot > 0 else None
        mb = lambda a, b: round(max(0, a - b) / dt / 1e6, 2)
        out["disk_read_mb_s"] = mb(cur["disk"][0], prev["disk"][0])
        out["disk_write_mb_s"] = mb(cur["disk"][1], prev["disk"][1])
        out["net_rx_mb_s"] = mb(cur["net"][0], prev["net"][0])
        out["net_tx_mb_s"] = mb(cur["net"][1], prev["net"][1])
    _prev["s"] = cur
    return out


def main():
    host = socket.gethostname().split(".")[0]
    final = os.path.join(RUN, "gpu-%s.json" % host)
    tmp = os.path.join(RUN, ".gpu-%s.%d.tmp" % (host, os.getpid()))
    os.makedirs(RUN, exist_ok=True)
    while True:
        try:
            doc = {"ts": time.time(), "host": host,
                   "job_id": os.environ.get("SLURM_JOB_ID"),
                   "gpus": sample()}
            try:
                for g, r in zip(doc["gpus"], pcie_rates(len(doc["gpus"]))):
                    if r:
                        g["pcie_rx_mb_s"], g["pcie_tx_mb_s"] = r
            except Exception as e:                                  # never lose the GPU feed over it
                print("gpu-publish: pcie: %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)
            try:
                doc["host_stats"] = host_sample()
            except Exception as e:                                  # never lose the GPU feed over it
                print("gpu-publish: host: %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)
            with open(tmp, "w") as fh:
                json.dump(doc, fh)
            os.replace(tmp, final)
        except Exception as e:
            print("gpu-publish: %s: %s" % (type(e).__name__, e), file=sys.stderr, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
