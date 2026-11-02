"""Thin wrappers around the Slurm CLI, using its native --json output.

Everything here runs as the panel's own user (slerm), so no privilege
plumbing is involved: sbatch/squeue/scancel/sacct all just work.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time

import config


class SlurmError(RuntimeError):
    pass


def _run(args, timeout=None):
    timeout = timeout or config.SLURM_TIMEOUT
    try:
        p = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/home/slerm"},
        )
    except FileNotFoundError:
        raise SlurmError("%s not found on this host" % args[0])
    except subprocess.TimeoutExpired:
        raise SlurmError("%s timed out after %ss" % (args[0], timeout))
    if p.returncode != 0:
        raise SlurmError((p.stderr or p.stdout or "").strip() or
                         "%s exited %d" % (args[0], p.returncode))
    return p.stdout


def _json(args, timeout=None):
    out = _run(args, timeout)
    try:
        return json.loads(out)
    except ValueError as exc:
        raise SlurmError("could not parse JSON from %s: %s" % (args[0], exc))


def num(field, default=None):
    """Unwrap Slurm's {set, infinite, number} envelope."""
    if isinstance(field, dict):
        if not field.get("set", False):
            return default
        if field.get("infinite", False):
            return None
        return field.get("number", default)
    if field is None:
        return default
    return field


def _state_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("current", []) or []
    if value:
        return [value]
    return []


def _parse_gres(text):
    """'gpu:turing:1' / 'gpu:turing:0(IDX:N/A)' -> [{type, model, count}]"""
    out = []
    if not text or text in ("N/A", "(null)"):
        return out
    for chunk in text.split(","):
        chunk = chunk.split("(")[0].strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) == 3:
            out.append({"type": parts[0], "model": parts[1], "count": _int(parts[2])})
        elif len(parts) == 2:
            out.append({"type": parts[0], "model": None, "count": _int(parts[1])})
    return out


def _int(text, default=0):
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _fresh_json(path, max_age):
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if time.time() - (doc.get("updated") or 0) <= max_age else None


# A node the orchestrator drained this long after slurmd started is on its way
# down, not coming back up: a boot is resumed within a minute or two.
STOPPING_AFTER_S = 600
ORCH_DRAIN = "slerm-orch"


def _power_phase(name, alloc_cpus, cpus, slurmd_uptime, states=None, reason=None):
    """Lifecycle of a node the slerm orchestrator manages, which Slurm alone
    cannot see: sleeping (powered off), waking, booting (powered on, slurmd not
    back yet), initialising (back in Slurm, not yet released by the
    orchestrator), stopping (drained by the orchestrator to power off), down
    (drained or failed by someone other than the orchestrator), idle / busy.
    deadlock also reports work the user does on it outside slerm.
    None when the orchestrator is not publishing (the card falls back to the
    raw Slurm state)."""
    orch = _fresh_json("/data/llm/run/orchestrator.json", 120)
    if not orch:
        return None
    dev = (orch.get("devices") or {}).get(name)
    if dev is None:
        return None
    out = {"phase": None, "detail": None, "outside": None,
           "can_wake": bool(dev.get("startable"))}
    queue = orch.get("queue") or {}
    states = set(states or [])
    ours = (reason or "").startswith(ORCH_DRAIN)
    if states & {"DRAIN", "DRAINED", "DRAINING", "FAIL", "FAILING"} and not ours:
        # The orchestrator reports such a node as not up, which would otherwise
        # read as asleep (or booting, for a running VM).
        out.update(phase="down", detail=reason or "drained outside slerm")
    elif not dev.get("up") and not dev.get("booting"):
        why = dev.get("reason") or "powered off"
        waking = bool(queue) and dev.get("startable")
        out.update(phase="waking" if waking else "sleeping",
                   detail=("powering on for " + ", ".join(queue)) if waking else why)
    elif dev.get("booting"):
        out.update(phase="booting", detail="powered on; waiting for slurmd")
    elif dev.get("drained_by_us") and (slurmd_uptime is None or slurmd_uptime >= STOPPING_AFTER_S):
        why = (reason or "")[len(ORCH_DRAIN):].lstrip(": ") or "orchestrator"
        out.update(phase="stopping", detail=why + "; powering off")
    elif dev.get("drained_by_us") or (slurmd_uptime is not None and slurmd_uptime < 90):
        out.update(phase="initialising", detail="back in Slurm; being released")
    elif alloc_cpus:
        out.update(phase="busy", detail="%d/%d cpus allocated" % (alloc_cpus, cpus))
    else:
        out.update(phase="idle")
    if name == "slerm-node05":
        host = _fresh_json("/data/llm/run/host-deadlock.json", 120)
        if host and host.get("busy"):
            out["outside"] = "; ".join(host.get("reasons") or [])
    return out


def nodes():
    """Per-node state and health, from `scontrol show node --json`."""
    data = _json(["scontrol", "show", "node", "--json"])
    now = time.time()
    live = gpu_live()
    out = []
    for n in data.get("nodes", []):
        cpus = n.get("cpus", 0) or 0
        alloc_cpus = n.get("alloc_cpus", 0) or 0
        real_mem = n.get("real_memory", 0) or 0
        alloc_mem = n.get("alloc_memory", 0) or 0
        free_mem = num(n.get("free_mem"))
        boot = num(n.get("boot_time"))
        slurmd = num(n.get("slurmd_start_time"))
        # cpu_load is reported x100 by Slurm
        raw_load = n.get("cpu_load")
        load = None
        if isinstance(raw_load, (int, float)):
            load = round(raw_load / 100.0, 2)
        states = _state_list(n.get("state"))
        gres = _parse_gres(n.get("gres", ""))
        gres_used = _parse_gres(n.get("gres_used", ""))
        used_by_key = {}
        for g in gres_used:
            used_by_key[(g["type"], g["model"])] = g["count"]
        for g in gres:
            g["used"] = used_by_key.get((g["type"], g["model"]), 0)
        out.append({
            "name": n.get("name"),
            "address": n.get("address"),
            "states": states,
            "state": states[0] if states else "UNKNOWN",
            "healthy": bool(states) and not (
                set(states) & {"DOWN", "DRAIN", "DRAINED", "DRAINING",
                               "FAIL", "FAILING", "NOT_RESPONDING",
                               "INVALID", "UNKNOWN", "ERROR"}),
            "partitions": n.get("partitions", []),
            "cpus": cpus,
            "alloc_cpus": alloc_cpus,
            "idle_cpus": n.get("alloc_idle_cpus", max(cpus - alloc_cpus, 0)),
            "cpu_pct": round(100.0 * alloc_cpus / cpus, 1) if cpus else 0.0,
            "cpu_load": load,
            "load_pct": round(100.0 * load / cpus, 1) if (load is not None and cpus) else None,
            "real_memory_mb": real_mem,
            "alloc_memory_mb": alloc_mem,
            "free_memory_mb": free_mem,
            "mem_pct": round(100.0 * alloc_mem / real_mem, 1) if real_mem else 0.0,
            "mem_used_pct": (round(100.0 * (real_mem - free_mem) / real_mem, 1)
                             if (real_mem and free_mem is not None) else None),
            "gres": gres,
            "boot_time": boot,
            "uptime_s": int(now - boot) if boot else None,
            "slurmd_start_time": slurmd,
            "slurmd_uptime_s": int(now - slurmd) if slurmd else None,
            "version": n.get("version"),
            "os": n.get("operating_system"),
            "arch": n.get("architecture"),
            "sockets": n.get("sockets"),
            "cores": n.get("cores"),
            "threads": n.get("threads"),
            "reason": n.get("reason") or None,
            # Live GPU telemetry, published for free by a running server job.
            # Absent when no job holds the card -- the panel falls back to the
            # on-demand probe button in that case.
            "gpu_live": live.get(n.get("name")),
            "reason_by": n.get("reason_set_by_user") or None,
            "power": _power_phase(n.get("name"), alloc_cpus, cpus,
                                  int(now - slurmd) if slurmd else None,
                                  states, n.get("reason")),
            "tmp_disk_mb": n.get("temporary_disk"),
            "weight": n.get("weight"),
        })
    out.sort(key=lambda x: x["name"] or "")
    return out


def partitions():
    """Partition rollup, derived from the node list plus `sinfo --json`."""
    data = _json(["sinfo", "--json"])
    agg = {}
    for entry in data.get("sinfo", []):
        part = (entry.get("partition") or {}).get("name")
        if not part:
            continue
        rec = agg.setdefault(part, {
            "name": part, "nodes": 0, "cpus_total": 0, "cpus_alloc": 0,
            "cpus_idle": 0, "states": {},
        })
        counts = entry.get("nodes", {}) or {}
        rec["nodes"] += counts.get("total", 0) or 0
        cpus = entry.get("cpus", {}) or {}
        rec["cpus_total"] += cpus.get("total", 0) or 0
        rec["cpus_alloc"] += cpus.get("allocated", 0) or 0
        rec["cpus_idle"] += cpus.get("idle", 0) or 0
        for st in _state_list((entry.get("node") or {}).get("state")):
            rec["states"][st] = rec["states"].get(st, 0) + (counts.get("total", 0) or 0)
    return sorted(agg.values(), key=lambda x: x["name"])


def queue():
    """Everything currently pending or running."""
    data = _json(["squeue", "--json"])
    out = []
    for j in data.get("jobs", []):
        states = _state_list(j.get("job_state"))
        array_task = j.get("array_task_id")
        out.append({
            "job_id": j.get("job_id"),
            "array_job_id": num(j.get("array_job_id")) or None,
            "array_task_id": num(array_task),
            "array_task_string": j.get("array_task_string") or None,
            "name": j.get("name"),
            "user": j.get("user_name"),
            "partition": j.get("partition"),
            "state": states[0] if states else "UNKNOWN",
            "states": states,
            "reason": j.get("state_reason") or None,
            "nodes": j.get("nodes") or None,
            "node_count": num(j.get("node_count")),
            "cpus": num(j.get("cpus")),
            "submit_time": num(j.get("submit_time")),
            "start_time": num(j.get("start_time")),
            "time_limit_min": num(j.get("time_limit")),
        })
    return out


def history(days=None, name=None, limit=400):
    """Finished jobs from the accounting DB."""
    days = days or config.HISTORY_DAYS
    args = ["sacct", "-X", "--allusers", "--starttime=now-%ddays" % days, "--json"]
    data = _json(args, timeout=45)
    out = []
    for j in data.get("jobs", []):
        if name and j.get("name") != name:
            continue
        t = j.get("time", {}) or {}
        state = j.get("state", {}) or {}
        arr = j.get("array", {}) or {}
        out.append({
            "job_id": j.get("job_id"),
            "array_job_id": arr.get("job_id") or None,
            "array_task_id": num(arr.get("task_id")),
            "name": j.get("name"),
            "user": j.get("user"),
            "partition": j.get("partition"),
            "state": (_state_list(state.get("current")) or ["UNKNOWN"])[0],
            "reason": state.get("reason") or None,
            "nodes": j.get("nodes"),
            "submit": num(t.get("submission")),
            "start": num(t.get("start")),
            "end": num(t.get("end")),
            "elapsed": t.get("elapsed"),
            "exit_code": num(((j.get("exit_code") or {}).get("return_code")) or {}),
            "stdout": j.get("stdout_expanded"),
            "stderr": j.get("stderr_expanded"),
            "submit_line": j.get("submit_line"),
        })
    out.sort(key=lambda x: (x["submit"] or 0), reverse=True)
    return out[:limit]


_SDIAG_KEYS = {
    "Jobs submitted": "jobs_submitted",
    "Jobs started": "jobs_started",
    "Jobs completed": "jobs_completed",
    "Jobs canceled": "jobs_canceled",
    "Jobs failed": "jobs_failed",
    "Jobs pending": "jobs_pending",
    "Jobs running": "jobs_running",
    "Last cycle": "sched_last_cycle_us",
    "Mean cycle": "sched_mean_cycle_us",
    "Total cycles": "sched_total_cycles",
    "Agent queue size": "agent_queue_size",
    "Server thread count": "server_threads",
}


def diag():
    """Scheduler internals from sdiag (text only -- no --json for this one)."""
    try:
        text = _run(["sdiag"])
    except SlurmError as exc:
        return {"error": str(exc)}
    out = {}
    section = "main"
    for line in text.splitlines():
        if line.startswith("Backfilling stats"):
            section = "backfill"
            continue
        if line.startswith("Main schedule statistics"):
            section = "sched"
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z ()/]*?)\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        label, value = m.group(1).strip(), m.group(2).strip()
        key = _SDIAG_KEYS.get(label)
        if not key:
            continue
        if section == "backfill" and key.startswith("sched_"):
            key = key.replace("sched_", "bf_")
        try:
            out[key] = int(value.split()[0])
        except (ValueError, IndexError):
            out[key] = value
    return out


def submit(sbatch_args, cwd=None):
    """Run sbatch and return the job id it printed."""
    args = ["sbatch"] + list(sbatch_args)
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=cwd,
                           env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/home/slerm"})
    except subprocess.TimeoutExpired:
        raise SlurmError("sbatch timed out")
    if p.returncode != 0:
        raise SlurmError((p.stderr or p.stdout or "").strip() or "sbatch failed")
    m = re.search(r"(\d+)", p.stdout)
    if not m:
        raise SlurmError("could not read a job id from sbatch: %r" % p.stdout)
    return int(m.group(1)), " ".join(args)


def cancel(job_id):
    _run(["scancel", str(job_id)])
    return True


GPU_LIVE_DIR = "/data/llm/run"
GPU_LIVE_MAX_AGE = 30.0


def gpu_live():
    """Free, continuous GPU telemetry published by the running llama.cpp jobs.

    gpu_probe below is the fallback: accurate, but it costs a real job slot, so
    it stays on-demand. When a server job is already sitting on the card it
    publishes nvidia-smi to NFS every few seconds and this just reads it: no
    slot, no ssh. Nothing deletes those files when a job dies, so anything
    older than GPU_LIVE_MAX_AGE is treated as absent rather than shown as live.
    """
    out = {}
    try:
        names = os.listdir(GPU_LIVE_DIR)
    except OSError:
        return out
    now = time.time()
    for fn in names:
        if not (fn.startswith("gpu-") and fn.endswith(".json")):
            continue
        try:
            with open(os.path.join(GPU_LIVE_DIR, fn)) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        ts = doc.get("ts")
        if not isinstance(ts, (int, float)) or now - ts > GPU_LIVE_MAX_AGE:
            continue
        host = doc.get("host")
        if host:
            doc["age_s"] = round(now - ts, 1)
            out[host] = doc
    return out


def gpu_probe(node, timeout=None):
    """Ask a GPU node about itself via a one-minute srun.

    Nodes are not SSH-reachable from the head, so this is the only way to get
    real GPU telemetry. It costs a job slot, so it is on-demand, never polled.
    """
    query = "name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
    args = ["srun", "--partition=gpu", "--nodelist=%s" % node, "--gres=gpu:1",
            "--ntasks=1", "--cpus-per-task=1", "--mem=512M", "--time=00:01:00",
            "--job-name=gpu-probe",
            "nvidia-smi", "--query-gpu=%s" % query, "--format=csv,noheader,nounits"]
    out = _run(args, timeout=timeout or config.SRUN_PROBE_TIMEOUT)
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({
            "name": parts[0],
            "temp_c": _int(parts[1], None),
            "util_pct": _int(parts[2], None),
            "mem_used_mb": _int(parts[3], None),
            "mem_total_mb": _int(parts[4], None),
            "power_w": float(parts[5]) if len(parts) > 5 and parts[5].replace(".", "").isdigit() else None,
        })
    return gpus


def head_stats():
    """Local health of the head node itself."""
    out = {"hostname": None, "load": None, "mem": None, "mounts": []}
    try:
        with open("/proc/sys/kernel/hostname") as fh:
            out["hostname"] = fh.read().strip()
    except OSError:
        pass
    try:
        with open("/proc/loadavg") as fh:
            parts = fh.read().split()
        out["load"] = [float(parts[0]), float(parts[1]), float(parts[2])]
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/uptime") as fh:
            out["uptime_s"] = int(float(fh.read().split()[0]))
    except (OSError, ValueError, IndexError):
        pass
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = int(v.strip().split()[0])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        out["mem"] = {
            "total_mb": total // 1024,
            "available_mb": avail // 1024,
            "used_mb": (total - avail) // 1024,
            "used_pct": round(100.0 * (total - avail) / total, 1) if total else 0,
            "swap_total_mb": info.get("SwapTotal", 0) // 1024,
            "swap_free_mb": info.get("SwapFree", 0) // 1024,
        }
    except (OSError, ValueError):
        pass
    try:
        out["cpus"] = len([1 for line in open("/proc/cpuinfo") if line.startswith("processor")])
    except OSError:
        out["cpus"] = None
    for path in config.WATCHED_MOUNTS:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        out["mounts"].append({
            "path": path,
            "total_gb": round(usage.total / 1e9, 1),
            "used_gb": round(usage.used / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
            "used_pct": round(100.0 * usage.used / usage.total, 1) if usage.total else 0,
        })
    return out


def parse_array_spec(spec):
    """Count tasks in an array task string like '5-217%32', '1,3,7' or '0-9:2'."""
    if not spec:
        return 0
    spec = str(spec).split("%")[0]
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, _, step_s = part.partition(":")
            step = max(_int(step_s, 1), 1)
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo, hi = _int(lo_s), _int(hi_s)
            if hi >= lo:
                total += (hi - lo) // step + 1
        else:
            total += 1
    return total


_DONE_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
_job_status_memo = {}
_job_status_lock = threading.Lock()
JOB_STATUS_TTL = 2.0


def job_status(job_id):
    """Return the current/terminal state for one non-array job.

    ``squeue`` owns live state and ``sacct`` owns terminal state. Accounting
    can lag briefly after a job leaves the queue, so an unknown job returns
    ``None`` and callers simply retry on their next poll.
    """
    try:
        job_id = int(job_id)
    except (TypeError, ValueError):
        raise SlurmError("invalid job id %r" % job_id)
    if job_id < 1:
        raise SlurmError("invalid job id %r" % job_id)

    now = time.time()
    with _job_status_lock:
        hit = _job_status_memo.get(job_id)
        if hit and hit[0] > now:
            return hit[1]

    def remember(result):
        with _job_status_lock:
            _job_status_memo[job_id] = (time.time() + JOB_STATUS_TTL, result)
        return result

    try:
        live = _json(["squeue", "--json", "-j", str(job_id)])
    except SlurmError:
        live = {"jobs": []}
    for job in live.get("jobs", []):
        if _int(job.get("job_id"), -1) != job_id:
            continue
        state = (_state_list(job.get("job_state")) or ["UNKNOWN"])[0]
        state = str(state).split("+", 1)[0]
        return remember({"job_id": job_id, "state": state,
                         "terminal": state in _DONE_STATES,
                         "success": state == "COMPLETED",
                         "reason": job.get("state_reason") or None,
                         "exit_code": None, "end": None})

    try:
        done = _json(["sacct", "-X", "-j", str(job_id), "--json"],
                     timeout=45)
    except SlurmError:
        done = {"jobs": []}
    for job in done.get("jobs", []):
        if _int(job.get("job_id"), -1) != job_id:
            continue
        state_info = job.get("state") or {}
        state = (_state_list(state_info.get("current")) or ["UNKNOWN"])[0]
        state = str(state).split("+", 1)[0]
        if state == "UNKNOWN":
            return remember(None)
        code = num(((job.get("exit_code") or {}).get("return_code")) or {})
        ended = num(((job.get("time") or {}).get("end")) or {})
        return remember({"job_id": job_id, "state": state,
                         "terminal": state in _DONE_STATES,
                         "success": state == "COMPLETED",
                         "reason": state_info.get("reason") or None,
                         "exit_code": code, "end": ended})
    return remember(None)


_progress_memo = {}
_progress_lock = threading.Lock()
PROGRESS_TTL = 2.0


def array_progress(job_id):
    """Live task breakdown for one (array) job, merging squeue and sacct.

    Memoised briefly: the UI polls once a second and a page can show many
    runs at once, but each call is a squeue plus a sacct against a job that
    may have hundreds of tasks.

    squeue knows about tasks that have not been created yet (they show up as a
    single row with a task *range*); sacct knows about everything that has
    already finished.
    """
    now = time.time()
    with _progress_lock:
        hit = _progress_memo.get(job_id)
        if hit and hit[0] > now:
            return hit[1]

    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0,
              "cancelled": 0, "other": 0}
    tasks = []
    seen = set()

    try:
        live = _json(["squeue", "--json", "-j", str(job_id)])
    except SlurmError:
        live = {"jobs": []}
    for j in live.get("jobs", []):
        states = _state_list(j.get("job_state"))
        state = states[0] if states else "UNKNOWN"
        task_id = num(j.get("array_task_id"))
        spec = j.get("array_task_string")
        n = parse_array_spec(spec) if (task_id is None and spec) else 1
        if state == "PENDING":
            counts["pending"] += n
        elif state in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
            counts["running"] += n
        else:
            counts["other"] += n
        if task_id is not None:
            seen.add(task_id)
            tasks.append({"task": task_id, "state": state,
                          "node": j.get("nodes") or None, "elapsed": None,
                          "reason": j.get("state_reason") or None})

    try:
        done = _json(["sacct", "-X", "-j", str(job_id), "--json"], timeout=45)
    except SlurmError:
        done = {"jobs": []}
    for j in done.get("jobs", []):
        arr = j.get("array", {}) or {}
        task_id = num(arr.get("task_id"))
        state = (_state_list((j.get("state") or {}).get("current")) or ["UNKNOWN"])[0]
        if task_id is None:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        if state == "COMPLETED":
            counts["completed"] += 1
        elif state in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"):
            counts["failed"] += 1
        elif state in ("CANCELLED", "REVOKED", "DEADLINE", "PREEMPTED"):
            counts["cancelled"] += 1
        elif state == "PENDING":
            counts["pending"] += 1
        elif state in ("RUNNING", "COMPLETING"):
            counts["running"] += 1
        else:
            counts["other"] += 1
        t = j.get("time", {}) or {}
        tasks.append({"task": task_id, "state": state, "node": j.get("nodes") or None,
                      "elapsed": t.get("elapsed"), "reason": None})

    tasks.sort(key=lambda x: x["task"])
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    counts["done"] = counts["completed"] + counts["failed"] + counts["cancelled"]
    counts["active"] = counts["pending"] + counts["running"]
    result = {"counts": counts, "tasks": tasks}
    with _progress_lock:
        _progress_memo[job_id] = (time.time() + PROGRESS_TTL, result)
        if len(_progress_memo) > 256:
            for stale in [k for k, v in _progress_memo.items() if v[0] < now]:
                _progress_memo.pop(stale, None)
    return result
