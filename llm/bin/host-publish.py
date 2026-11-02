#!/usr/bin/env python3
"""Publish what a slerm node is doing OUTSIDE Slurm, so the orchestrator can
tell "deadlock is idle" from "the user is using deadlock".

deadlock is a bare-metal box the user also uses for other things (its 2.6 TB
/archive volume, logins), so it must never be powered off, or have its RAM
handed to slerm, while that is going on. Everything Slurm runs lives in one
cgroup (system.slice/slurmstepd.scope), so host totals minus that cgroup are
exactly the non-slerm load.

Writes /data/llm/run/host-<node>.json every INTERVAL seconds:
  busy     true when any signal below crosses its threshold
  reasons  human-readable, shown on the panel's model cards
  free_for_slerm_mb   RAM slerm may use without squeezing the user's work

Python 3.9 (node version). Needs no privileges.
"""
import json
import os
import socket
import subprocess
import time

INTERVAL = 15
NODE = socket.gethostname().split(".")[0]
OUT = "/data/llm/run/host-%s.json" % NODE
SLURM_CG = "/sys/fs/cgroup/system.slice/slurmstepd.scope"
# Block devices whose I/O means the user is working (deadlock: /archive).
WATCH_MOUNTS = [m for m in os.environ.get("HOST_WATCH_MOUNTS", "/archive").split(",") if m]

# Thresholds, averaged over WINDOW seconds so a cron blip is not "busy".
WINDOW = 120
CPU_CORES_BUSY = 0.5          # non-slerm CPU, in cores
MEM_BUSY_MB = 1536            # non-slerm RAM above the idle OS baseline
OS_BASELINE_MB = 1024
IO_BUSY_MBPS = 2.0            # on the watched mounts
SESSION_IDLE_BUSY = 30 * 60   # an interactive login used within this long
RESERVE_MB = 2048             # always left for the OS and the user


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def cpu_total_usec():
    f = read("/proc/stat").split("\n")[0].split()[1:]
    ticks = sum(int(x) for x in f[:8]) - int(f[3]) - int(f[4])     # minus idle, iowait
    return ticks * 1e6 / os.sysconf("SC_CLK_TCK")


def slurm_cpu_usec():
    for line in read(os.path.join(SLURM_CG, "cpu.stat")).splitlines():
        if line.startswith("usage_usec"):
            return float(line.split()[1])
    return 0.0


def meminfo():
    m = {}
    for line in read("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) // 1024
    return m


def slurm_mem_mb():
    try:
        return int(read(os.path.join(SLURM_CG, "memory.current"), "0")) // (1 << 20)
    except ValueError:
        return 0


def mount_devices():
    devs = {}
    for line in read("/proc/self/mountinfo").splitlines():
        p = line.split()
        if p[4] in WATCH_MOUNTS:
            major, minor = p[2].split(":")
            devs[(major, minor)] = p[4]
    return devs


def disk_sectors(devs):
    out = {}
    for line in read("/proc/diskstats").splitlines():
        p = line.split()
        key = (p[0], p[1])
        if key in devs:
            out[devs[key]] = int(p[5]) + int(p[9])      # sectors read + written
    return out


def sessions(now):
    """Interactive logins (a TTY), with how long since they were last used."""
    found = []
    try:
        ids = subprocess.run(["loginctl", "list-sessions", "--no-legend"],
                             capture_output=True, text=True, timeout=10).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return found
    for line in ids:
        if not line.strip():
            continue
        sid = line.split()[0]
        props = {}
        try:
            r = subprocess.run(["loginctl", "show-session", sid, "-p", "Name", "-p", "TTY",
                                "-p", "Remote", "-p", "RemoteHost", "-p", "Type"],
                               capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for kv in r.splitlines():
            k, _, v = kv.partition("=")
            props[k] = v
        tty = props.get("TTY", "")
        if not tty:
            continue            # scripted ssh (the orchestrator, Claude) has no TTY
        try:
            idle = now - os.stat("/dev/" + tty).st_atime
        except OSError:
            idle = 0
        found.append({"user": props.get("Name"), "tty": tty,
                      "from": props.get("RemoteHost") or "console", "idle_s": int(idle)})
    return found


def main():
    devs = mount_devices()
    hist = []       # (t, total_cpu_usec, slurm_cpu_usec, {mount: sectors})
    while True:
        now = time.time()
        hist.append((now, cpu_total_usec(), slurm_cpu_usec(), disk_sectors(devs)))
        hist = [h for h in hist if h[0] >= now - WINDOW]
        t0, tot0, sl0, d0 = hist[0]
        dt = max(1.0, now - t0)
        other_cores = max(0.0, (hist[-1][1] - tot0) - (hist[-1][2] - sl0)) / 1e6 / dt
        io = {m: (hist[-1][3].get(m, 0) - d0.get(m, 0)) * 512 / 1e6 / dt for m in devs.values()}

        mi = meminfo()
        used = mi["MemTotal"] - mi["MemAvailable"]
        slurm_mb = slurm_mem_mb()
        other_mb = max(0, used - slurm_mb)
        sess = sessions(now)

        reasons = []
        if other_cores >= CPU_CORES_BUSY:
            reasons.append("%.1f CPU cores in use outside slerm" % other_cores)
        if other_mb - OS_BASELINE_MB >= MEM_BUSY_MB:
            reasons.append("%.1f GB RAM in use outside slerm" % (other_mb / 1024))
        for m, mbps in io.items():
            if mbps >= IO_BUSY_MBPS:
                reasons.append("%s I/O at %.0f MB/s" % (m, mbps))
        for s in sess:
            if s["idle_s"] < SESSION_IDLE_BUSY:
                reasons.append("%s logged in on %s from %s" % (s["user"], s["tty"], s["from"]))

        doc = {"node": NODE, "updated": int(now), "window_s": int(dt),
               "busy": bool(reasons), "reasons": reasons,
               "other_cpu_cores": round(other_cores, 2), "cpus": os.cpu_count(),
               "other_mem_mb": other_mb, "slurm_mem_mb": slurm_mb,
               "mem_total_mb": mi["MemTotal"],
               "free_for_slerm_mb": max(0, mi["MemTotal"] - other_mb - RESERVE_MB),
               "io_mbps": {m: round(v, 2) for m, v in io.items()},
               "sessions": sess}
        tmp = OUT + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(doc, f, indent=1)
            os.replace(tmp, OUT)
        except OSError as e:
            print("host-publish: cannot write %s: %s" % (OUT, e), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
