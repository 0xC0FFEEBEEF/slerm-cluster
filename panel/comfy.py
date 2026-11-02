"""ComfyUI on slerm, for the AI tab (2026-10-06).

ComfyUI is served at cryoai:8188 by comfy-gate: the UI comes from a CPU-only ComfyUI on cryoai, and prompts run on
an on-demand Slurm job (`comfyui`, one P100 on slerm-node04, /data/comfyui/comfyui-job.sh) that the gate starts and
that exits after an idle spell. The head cannot reach the worker itself (it listens on node04's loopback and the
head cannot ssh to compute nodes), so this joins three things it can read:

- the gate's GET /gate/status over the LAN (worker up/starting, queue, prompts held, live step progress),
- the Slurm job (squeue), and
- the worker's own /data/comfyui/run/worker.json (state and idle countdown, every 10 s) plus the shared GPU feed.
"""
import json
import time
import urllib.error
import urllib.request

import slurmapi

GATE = "http://cryoai:8188"
OPEN_URL = "http://cryoai:8188/"
JOB_NAME = "comfyui"
WORKER_FILE = "/data/comfyui/run/worker.json"
WORKER_MAX_AGE = 45


def _gate_status():
    try:
        with urllib.request.urlopen(GATE + "/gate/status", timeout=2) as r:
            return json.load(r)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _worker_file():
    try:
        with open(WORKER_FILE) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if time.time() - (doc.get("updated") or 0) > WORKER_MAX_AGE:
        return None
    return doc


def _job():
    try:
        jobs = [j for j in slurmapi.queue() if j.get("name") == JOB_NAME]
    except Exception:                                            # noqa: BLE001 - squeue hiccup: show no job
        return None
    running = [j for j in jobs if j.get("state") == "RUNNING"]
    return (running or jobs or [None])[0]


def summary():
    gate = _gate_status()
    job = _job()
    wfile = _worker_file()
    w = (gate or {}).get("worker") or {}
    queue = (gate or {}).get("queue") or {}
    progress = (gate or {}).get("progress")
    held = (gate or {}).get("held") or 0

    if job and job.get("state") == "PENDING":
        phase = "waiting"               # for a GPU (the LLM router yields once idle) or a node boot
    elif w.get("up") and (queue.get("running") or queue.get("pending")):
        phase = "generating"
    elif w.get("up"):
        phase = "ready"
    elif job and job.get("state") == "RUNNING":
        phase = "starting"
    elif held or w.get("starting_since"):
        phase = "waiting"
    else:
        phase = "off"

    gpu = None
    node = (job or {}).get("nodes") or (wfile or {}).get("node")
    if node and phase in ("starting", "ready", "generating"):
        feeds = slurmapi.gpu_live()
        gpu = feeds.get(node) or feeds.get(str(node).replace("slerm-", ""))

    idle_left = None
    if wfile and phase == "ready" and wfile.get("idle_limit"):
        idle_left = max(0, int(wfile["idle_limit"]) - int(wfile.get("idle_s") or 0))

    return {
        "phase": phase,
        "open_url": OPEN_URL,
        "gate_reachable": gate is not None,
        "job": job and {k: job.get(k) for k in ("job_id", "state", "reason", "nodes", "start_time",
                                               "submit_time", "partition")},
        "worker_up_since": w.get("up_since"),
        "starting_since": w.get("starting_since"),
        "start_note": w.get("start_note") or "",
        "queue": queue or None,
        "held": held,
        "progress": progress,
        "last_done": (gate or {}).get("last_done"),
        "sessions": (gate or {}).get("sessions"),
        "idle_left_s": idle_left,
        "idle_limit_s": (wfile or {}).get("idle_limit"),
        "gpu": gpu,
        "node": node,
    }


def stop():
    """Cancel the worker job (frees the card at once instead of after the idle spell)."""
    job = _job()
    if not job:
        return {"stopped": None}
    slurmapi.cancel(job["job_id"])
    return {"stopped": job["job_id"]}
