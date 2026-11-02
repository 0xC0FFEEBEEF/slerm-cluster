"""Cluster inference accounting: what the GPU nodes were asked to generate.

This is about the CLUSTER's models, not the panel's. Every client of the
llama.cpp servers running in the `llm` partition is counted the same way --
Open WebUI, the delegate harness, a curl from a laptop, the panel's own Wazuh
triage if it is pointed here. The panel is a bystander that reads two things
those servers already produce:

  * **the server log** (`llmlog.py`), which is the only per-request record
    anywhere: prompt tokens, generated tokens, timings and which model served
    it, one row per completed call;
  * **`/metrics`**, cumulative counters per model instance, sampled as deltas
    so they survive the model swaps that reset them.

Neither costs a job slot, neither needs SSH to a node, and neither perturbs the
server. Nothing here holds a credential, and the only write is `unload()`,
which a person asks for from the AI tab's server card.

**A log is only ever tailed forward.** llama.cpp's log clock is relative to the
stream that wrote the line and every model instance has its own zero, so the
file carries no shared wall time -- the router's clock was measured running
601 s behind its own Slurm job start, and a child's 8771 s behind the router's.
Rather than reconstruct timestamps by guesswork, a request is stamped with the
moment the panel read it, which is exact to the poll interval. The cost is that
whatever a log already contained when the panel first saw it is skipped, and
the UI says so.
"""
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import config
import db
import ggufmeta
import llmlog
import slurmapi

WINDOWS = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800,
           "30d": 2592000, "90d": 7776000, "all": None}
DEFAULT_WINDOW = "24h"

HOUR = 3600
_MAX_BUCKETS = 24 * 400

_parsers = {}                      # log path -> a live llmlog.Parser
_poll_lock = threading.Lock()
_state = {"last_poll": None, "last_error": None, "started": None,
          "polls": 0, "requests": 0}
_servers_cache = {"ts": 0.0, "data": None}
SERVERS_TTL = 5.0


# --- discovering the servers -------------------------------------------------

# --- the pooled devices ------------------------------------------------------
# A router job's own node holds one card; the rest of its compute is other
# nodes reached over ggml-rpc, each held open by its own Slurm job. Showing
# only the local card understates what is actually serving the model -- with
# gpt-oss-20b resident, node04's 980 Ti carries 5966 of its 6144 MB and would
# not appear at all.

_NODE_IPS = {}


def _node_ip(node):
    """Resolved once per node: this is called on every poll."""
    if node not in _NODE_IPS:
        try:
            _NODE_IPS[node] = socket.gethostbyname(node)
        except OSError:
            _NODE_IPS[node] = None
    return _NODE_IPS[node]


_RPC_LINE = re.compile(r"^rpc devices:\s*(.+)$", re.M)


def _rpc_devices(log_path):
    """The endpoints this router actually pooled, from its own log.

    router-rpc.sh prints them once before exec, so only the head of the file is
    read. None means the line is not there -- an older job, or a plain
    single-card server that pools nothing -- and the caller must not then claim
    a match it cannot prove.
    """
    if not log_path:
        return None
    try:
        with open(log_path) as fh:
            head = fh.read(65536)
    except OSError:
        return None
    found = _RPC_LINE.findall(head)
    if not found:
        return None
    return set(part.strip().split(":")[0]
               for part in found[-1].split(",") if part.strip())


def _pool(jobs, live, log_path):
    """Worker jobs lending this router a device, with their GPU feed.

    Discovered from the queue the same way the server is, then matched against
    the endpoint list the router logged. Without that line the pool is reported
    as unconfirmed rather than presented as fact -- two routers running at once
    would otherwise each claim every worker.
    """
    wanted = _rpc_devices(log_path)
    names = getattr(config, "LLM_WORKER_JOB_NAMES", ())
    cpu_names = getattr(config, "LLM_WORKER_CPU_NAMES", ())
    out = []
    for job in jobs:
        if job.get("name") not in names or job.get("state") != "RUNNING":
            continue
        node = _single_node(job)
        if not node:
            continue
        if wanted is not None and node not in wanted \
                and _node_ip(node) not in wanted:
            continue
        feed = live.get(node)
        # gpu_live() already drops feeds by age, but nothing deletes the files,
        # so a FRESH file written by a different job on that node is still the
        # wrong reading to attribute to this worker.
        if feed and str(feed.get("job_id")) != str(job.get("job_id")):
            feed = None
        out.append({"job_id": job.get("job_id"), "name": job.get("name"),
                    "node": node, "partition": job.get("partition"),
                    "kind": "cpu" if job.get("name") in cpu_names else "gpu",
                    "gpu": feed})
    return out, wanted is not None


def _job_matches(job):
    names = getattr(config, "LLM_SERVER_JOB_NAMES", ())
    return job.get("name") in names and job.get("state") == "RUNNING"


def _single_node(job):
    """Only a one-node server has an address this can talk to."""
    node = (job.get("nodes") or "").strip()
    if not node or "," in node or "[" in node:
        return None
    return node


def servers(refresh=False):
    """Every running llama.cpp server job, with its live state.

    Discovered from Slurm rather than pinned, because these jobs move between
    GPU nodes on every requeue.
    """
    now = time.time()
    if not refresh and _servers_cache["data"] is not None \
            and now - _servers_cache["ts"] < SERVERS_TTL:
        return _servers_cache["data"]
    out = []
    try:
        jobs = slurmapi.queue()
    except Exception as exc:                                    # noqa: BLE001
        jobs = []
        out.append({"name": "slurm", "ok": False,
                    "error": "could not read the queue: %s" % exc})
    try:
        live = slurmapi.gpu_live()
    except Exception:                                           # noqa: BLE001
        live = {}
    strata = []
    for job in jobs:
        if _is_strata_job(job):
            # Kept apart (strata_servers()): the Wazuh triage and the
            # load/unload buttons pick from this list and only speak llama.cpp.
            strata.append(_strata_record(job, live))
            continue
        if not _job_matches(job):
            continue
        node = _single_node(job)
        record = {
            "job_id": job.get("job_id"), "name": job.get("name"),
            "node": node, "partition": job.get("partition"),
            "started": job.get("start_time"),
            "url": ("http://%s:%d" % (node, config.LLM_SERVER_PORT)
                    if node else None),
            "log": _log_path(job.get("job_id")),
            "ok": False, "error": None, "models": [], "loaded": None,
            "metrics": None,
            "gpu": live.get(node) if node else None,
        }
        record["pool"], record["pool_confirmed"] = _pool(
            jobs, live, record["log"])
        if record["url"]:
            _probe(record)
        else:
            record["error"] = "job is on %r, not a single node" % job.get("nodes")
        out.append(record)
    _servers_cache["ts"] = time.time()
    _servers_cache["data"] = out
    _servers_cache["strata"] = strata
    return out


def strata_servers(refresh=False):
    """The running Strata job (STRATA-* models), discovered with servers()."""
    servers(refresh)
    return _servers_cache.get("strata") or []


# --- Strata ------------------------------------------------------------------
# 2026-10-17 (user: "is the AI tab configured to watch for strata?"): the
# STRATA-* models are served by Strata (/data/llm/strata), one Slurm job
# llm-strata (jobs/strata.sh) on node04's two P100s, submitted by slerm-orch
# instead of the llama.cpp router. It speaks none of the router's API, so it
# has its own discovery, progress and request log here:
#
#   * start-up (8-12 min) from the job's own log, strata-<job>.out, whose
#     "=== <time> ..." lines mark each step, and the server's log
#     strata-server-<job>.log for what the engine is doing. Strata opens its
#     port only once the engine is up, so until then the logs are all there is.
#     It counts as loaded once the ready marker is written: the gateway holds
#     requests until then.
#   * once up, its JSON /metrics: totals, what it is doing now, and the last
#     requests, each with its own start time (node04's clock, in sync with the
#     head), so a request is stored with its real time rather than the poll's.

STRATA_JOB_NAMES = ("llm-strata",)
STRATA_REGISTRY = "/data/llm/etc/strata-models.json"
STRATA_READY_MARKER = "/data/llm/run/strata-ready.json"
STRATA_TIMEOUT = 3
_STRATA_STEP = re.compile(r"^=== (\S+) (.*)$")
_STRATA_HEADER = re.compile(r"^job (\S+) on (\S+): (\S+)")
_STRATA_AFTER = re.compile(r"after (\d+)s")
_strata_hist = {"ts": 0.0, "data": {}}
# step -> what the card says; in the order jobs/strata.sh runs them
STRATA_PHASES = {
    "preparing": "preparing the node",
    "port": "waiting for port %s to free up",
    "warming": "reading the experts into RAM",
    "engine": "starting the engine",
    "table": "warming the n-gram table (requests wait for this)",
    "ready": "ready",
    "restarting": "engine restarting",
    "failed": "did not become ready",
    "exited": "server exited",
}


def _is_strata_job(job):
    return job.get("name") in STRATA_JOB_NAMES and \
        job.get("state") in ("RUNNING", "PENDING", "CONFIGURING")


def _strata_port():
    try:
        with open(STRATA_REGISTRY) as f:
            return int(json.load(f).get("port", 18100))
    except (OSError, ValueError, TypeError):
        return 18100


def _strata_registry_entry(model):
    try:
        with open(STRATA_REGISTRY) as f:
            return (json.load(f).get("models") or {}).get(model) or {}
    except (OSError, ValueError, TypeError):
        return {}


def _read_lines(path, limit=256 * 1024):
    """The last `limit` bytes of a small log, as lines."""
    if not path:
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def _iso_ts(text):
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _strata_steps(lines):
    """What a strata-<job>.out says: the model, each step and when it began."""
    out = {"model": None, "phase": "preparing", "since": None, "phase_since": None,
           "detail": None, "port": None, "ready_after_s": None, "marker_after_s": None,
           "rc": None}
    for line in lines:
        if line.startswith("page-warm:"):
            out["detail"] = line[len("page-warm:"):].strip()
            continue
        if line.startswith("port ") and "busy" in line:
            out["phase"], out["port"] = "port", line.split()[1]
            continue
        m = _STRATA_STEP.match(line)
        if not m:
            continue
        ts, what = _iso_ts(m.group(1)), m.group(2)
        head = _STRATA_HEADER.match(what)
        after = _STRATA_AFTER.search(what)
        step = None
        if head:
            out["model"], out["since"] = head.group(3), ts
            step = "preparing"
        elif what.startswith("warming before start"):
            step = "warming"
        elif what.startswith("starting the server"):
            step = "engine"
        elif what.startswith("READY after"):
            step = "table"
            out["ready_after_s"] = int(after.group(1)) if after else None
        elif what.startswith("ready marker written"):
            step = "ready"
            out["marker_after_s"] = int(after.group(1)) if after else None
        elif what.startswith("server did not become ready"):
            step = "failed"
        elif what.startswith("server exited"):
            step = "exited"
            rc = re.search(r"\((-?\d+)\)", what)
            out["rc"] = int(rc.group(1)) if rc else None
        if step:
            if step != out["phase"] or out["phase_since"] is None:
                out["detail"] = None
            out["phase"], out["phase_since"] = step, ts
    return out


def _strata_reason(text):
    """The short why of Strata's "the engine stopped unexpectedly ..." line."""
    if "Its log line:" in text:
        text = text.split("Its log line:", 1)[1].strip()
        if text.startswith("strata serve:"):
            text = text[len("strata serve:"):].strip()
        text = text.split(" - ", 1)[0]
    else:
        text = text.split(". ", 1)[0]
    return text[:160]


def _strata_engine(path):
    """What the server log says about the engine: its latest step ("filling
    the GPU's expert cache ..."), the seconds of its "still starting (N s)"
    heartbeat for the current (re)start, and why it last restarted. Request
    progress lines and indented asides are skipped."""
    out = {"step": None, "starting_s": None, "restart_reason": None}
    for line in _read_lines(path):
        if not line.strip() or line[:1].isspace():
            continue
        text = line[len("[strata] "):].strip() if line.startswith("[strata] ") else line.strip()
        beat = re.search(r"still starting \((\d+) s\)", text)
        if beat:
            out["starting_s"] = int(beat.group(1))
            continue
        if text.startswith(("reading the prompt", "thinking", "answering", "done:")):
            continue
        if text.startswith("the engine stopped unexpectedly"):
            out["restart_reason"] = _strata_reason(text)
            continue
        if text.startswith("the engine had stopped") or text.startswith("starting the engine"):
            out["starting_s"] = 0
        out["step"] = text
    return out


def _strata_history():
    """model -> the seconds its finished starts took (newest last), from the
    logs of earlier Strata jobs. Re-read once a minute at most."""
    now = time.time()
    if now - _strata_hist["ts"] < 60:
        return _strata_hist["data"]
    directory = getattr(config, "LLM_LOG_DIR", "/data/llm/logs")
    data = {}
    try:
        names = sorted(n for n in os.listdir(directory)
                       if n.startswith("strata-") and n.endswith(".out"))
    except OSError:
        names = []
    for name in names:
        steps = _strata_steps(_read_lines(os.path.join(directory, name), 32 * 1024))
        took = steps["marker_after_s"]
        if steps["model"] and took:
            data.setdefault(steps["model"], []).append(took)
    _strata_hist.update(ts=now, data=data)
    return data


def _strata_marker_job():
    try:
        with open(STRATA_READY_MARKER) as f:
            return str(json.load(f).get("job"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _strata_record(job, live):
    node = _single_node(job)
    job_id = job.get("job_id")
    port = _strata_port()
    directory = getattr(config, "LLM_LOG_DIR", "/data/llm/logs")
    record = {
        "job_id": job_id, "name": job.get("name"), "engine": "strata",
        "node": node, "partition": job.get("partition"),
        "started": job.get("start_time"), "job_state": job.get("state"),
        "url": "http://%s:%d" % (node, port) if node else None, "port": port,
        "log": _log_path(job_id),
        "server_log": os.path.join(directory, "strata-server-%s.log" % job_id),
        "ok": False, "error": None, "models": [], "loaded": None,
        "metrics": None, "gpu": live.get(node) if node else None,
        "pool": [], "pool_confirmed": True,
    }
    steps = _strata_steps(_read_lines(record["log"]))
    orch = (_orch_state().get("strata") or {})
    model = steps["model"] or (orch.get("model") if str(orch.get("job")) == str(job_id) else None)
    record["model"] = model
    if job.get("state") != "RUNNING":
        steps["phase"] = "pending"
    # The orchestrator's own test when it is fresh (marker of THIS job + /health
    # loaded), otherwise the marker alone.
    if str(orch.get("job")) == str(job_id) and "ready" in orch:
        ready = bool(orch.get("ready"))
    else:
        ready = steps["phase"] == "ready" and _strata_marker_job() == str(job_id)
    engine = _strata_engine(record["server_log"]) if steps["phase"] in ("engine", "ready") else {}
    if steps["phase"] == "ready" and not ready:
        # Up once, not loaded now: Strata restarts its engine after a crash or
        # a hang its watchdog caught (the job log still says ready).
        steps["phase"] = "restarting"
    now = time.time()
    since = steps["since"] or record["started"]
    took = (_strata_history().get(model) or [])
    expected = took[-1] if took else None
    start = {
        "phase": steps["phase"],
        "text": (STRATA_PHASES.get(steps["phase"], "waiting for Slurm") % steps["port"]
                 if steps["phase"] == "port" else
                 STRATA_PHASES.get(steps["phase"], "waiting for Slurm to start the job")),
        "detail": steps["detail"],
        "since": since,
        "elapsed_s": round(now - since) if since else None,
        "phase_elapsed_s": round(now - steps["phase_since"]) if steps["phase_since"] else None,
        "expected_s": expected, "starts_seen": len(took),
        "ready_after_s": steps["ready_after_s"], "marker_after_s": steps["marker_after_s"],
    }
    if steps["phase"] in ("engine", "restarting"):
        start["detail"] = engine.get("step")
    if steps["phase"] == "restarting":
        # The heartbeat restarts at 0 with the engine; the job-start estimate means nothing here.
        start["phase_elapsed_s"] = engine.get("starting_s")
        start["reason"] = engine.get("restart_reason")
        start["expected_s"] = None
    elif expected and since and not ready:
        start["pct"] = round(min(99.0, 100.0 * (now - since) / expected), 1)
    record["start"] = start
    if steps["phase"] in ("failed", "exited"):
        record["error"] = "%s (see %s)" % (start["text"], record["server_log"])
    else:
        _strata_refresh(record)
    if ready and record["ok"]:
        record["loaded"] = model
    entry = _strata_registry_entry(model)
    if model:
        record["models"].append({
            "id": model, "state": "loaded" if record["loaded"] else "loading",
            "loaded": bool(record["loaded"]),
            "ctx": ((record.get("engine_info") or {}).get("max_context")
                    or entry.get("ctx")),
            "port": port, "params": None, "size_bytes": None, "quant": None,
            "path": None, "engine": "strata"})
    return record


_STRATA_SERVING = ("table", "ready", "restarting")     # the port is open from READY on


def _strata_refresh(record):
    """Probe a server whose port should be open; one that is still starting,
    or restarting its engine, is not an error when it does not answer."""
    phase = (record.get("start") or {}).get("phase")
    if phase in _STRATA_SERVING and record.get("url"):
        _strata_probe(record)
        if not record["ok"] and phase == "restarting":
            record["ok"], record["error"] = True, None
    else:
        record["ok"] = True


def _strata_get(record, path):
    req = urllib.request.Request(record["url"] + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=STRATA_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _strata_probe(record):
    """Read Strata's JSON /metrics into the router-shaped fields the AI tab
    draws, plus its own (expert cache hits, MTP draft acceptance)."""
    try:
        d = _strata_get(record, "/metrics")
    except Exception as exc:                                    # noqa: BLE001
        record["ok"] = False
        record["error"] = "unreachable: %s" % exc
        return
    record["ok"] = True
    record["error"] = None
    t = d.get("totals") or {}
    lv = d.get("live") or {}
    reqs = d.get("requests") or []
    record["engine_info"] = d.get("engine") or {}
    record["strata_live"] = lv
    record["strata_requests"] = reqs[:15]                       # newest first
    prompt_s = (t.get("prompt_ms") or 0) / 1000.0
    gen_s = (t.get("decode_ms") or 0) / 1000.0
    read = max(0, (t.get("prompt_tokens") or 0) - (t.get("reused") or 0))
    out = t.get("output_tokens") or 0
    if "running" in lv:
        running = lv.get("running") or 0
    else:
        running = 1 if lv.get("state") in ("reading", "generating") else 0
    offered, accepted = t.get("drafts_offered") or 0, t.get("drafts_accepted") or 0
    record["metrics"] = {
        "requests": t.get("requests") or 0,
        "prompt_tokens": read, "cached_tokens": t.get("reused") or 0,
        "generated_tokens": out,
        "prompt_seconds": round(prompt_s, 1), "generated_seconds": round(gen_s, 1),
        "prompt_tok_s": round(read / prompt_s, 1) if prompt_s else 0,
        "generated_tok_s": round(out / gen_s, 1) if gen_s else 0,
        "generated_tok_s_recent": lv.get("tok_s"),
        "requests_processing": running,
        "requests_deferred": lv.get("queued") or 0,
        "draft_accept": round(accepted / offered, 3) if offered else None,
        "hit_rate": next((r.get("hit_rate") for r in reqs if r.get("hit_rate") is not None), None),
    }


def _strata_live(record):
    """What the Strata server is doing now, in the shape live() returns."""
    out = {"job_id": record.get("job_id"), "name": record.get("name"),
           "node": record.get("node"), "ok": record.get("ok"),
           "engine": "strata", "loaded": record.get("loaded"),
           "loading": [], "slots": [], "queued": None, "processing": None}
    st = record.get("start") or {}
    if not record.get("loaded") and record.get("model"):
        out["loading"].append(dict(st, model=record["model"], engine="strata",
                                   error=record.get("error")))
    m = record.get("metrics")
    if m:
        out["queued"] = m.get("requests_deferred")
        out["processing"] = m.get("requests_processing")
    lv = record.get("strata_live") or {}
    ctx = (record.get("engine_info") or {}).get("max_context")
    if lv.get("state") == "reading":
        total = lv.get("prompt_total") or lv.get("prompt_tokens") or 0
        done = lv.get("prompt_read") or 0
        rate = lv.get("prefill_tok_s_mean")
        slot = {"id": 0, "task": None, "engine": "strata", "n_ctx": ctx,
                "phase": "prompt", "prompt_tokens": total, "prompt_done": done,
                "prompt_cached": 0, "decoded": 0, "max_new": None,
                "prompt_tok_s": rate, "gen_tok_s": None}
        if rate and total > done:
            slot["eta_s"] = round((total - done) / rate)
        out["slots"].append(slot)
    elif lv.get("state") == "generating":
        made = lv.get("generated") or 0
        cap = lv.get("max_tokens")
        rate = lv.get("tok_s")
        slot = {"id": 0, "task": None, "engine": "strata", "n_ctx": ctx,
                "phase": "generating", "step": lv.get("phase"),
                "prompt_tokens": lv.get("prompt_tokens") or 0, "prompt_done": None,
                "prompt_cached": 0, "decoded": made, "max_new": cap,
                "prompt_tok_s": None, "gen_tok_s": rate}
        if cap and rate:
            slot["eta_s"] = round(max(0, cap - made) / rate)
        out["slots"].append(slot)
    if record.get("loaded") and record.get("engine_info"):
        out["monitor"] = _strata_monitor(record, lv)
    return out


# Strata monitor (2026-10-21, user: "can i have these kind of analytics in the AI tab", showing Strata's own Monitor
# page): what that page shows, from the same /metrics, plus node04's published feed (cards, CPU, RAM, disk). Sent on
# the 1.5 s live poll; the browser keeps the history for the sparklines.
_STRATA_ENGINE_KEYS = ("version", "model", "max_context", "kv", "kv_resident", "expert_slots", "expert_cache_mib",
                       "expert_slots_primary", "expert_cache_primary_mib", "vram_free_mib", "spec", "mtp_max",
                       "pcie_frac", "pool_workers", "conversation_cache_mib", "conversation_cache_slots",
                       "conversation_cache_min_tokens")
_STRATA_REQUEST_KEYS = ("time", "duration_s", "finish", "prompt_tokens", "reused", "prompt_read", "output_tokens",
                        "prompt_ms", "decode_ms", "decode_tok_s", "hit_rate", "pcie_share", "ram_blobs",
                        "file_blobs", "file_mb", "drafts_offered", "drafts_accepted")


def _strata_monitor(record, lv):
    eng = record.get("engine_info") or {}
    return {
        "live": lv,
        "engine": {k: eng.get(k) for k in _STRATA_ENGINE_KEYS if k in eng},
        "requests": [{k: r.get(k) for k in _STRATA_REQUEST_KEYS}
                     for r in record.get("strata_requests") or []],
        "feed": record.get("gpu"),
    }


def _strata_ingest(server):
    """Store the Strata server's finished requests. Each has its own start
    time, so (job, port, start in ms) identifies it and a re-read is a no-op."""
    if not server.get("url") or (server.get("start") or {}).get("phase") not in _STRATA_SERVING:
        return 0
    try:
        d = _strata_get(server, "/metrics?requests=all")
    except Exception:                                           # noqa: BLE001
        return 0
    rows = []
    for r in d.get("requests") or []:
        began = r.get("time")
        if not isinstance(began, (int, float)):
            continue
        took = r.get("duration_s") or 0
        read = r.get("prompt_read")
        if read is None:
            read = max(0, (r.get("prompt_tokens") or 0) - (r.get("reused") or 0))
        made = r.get("output_tokens") or 0
        prompt_ms = r.get("prompt_ms") or 0
        decode_ms = r.get("decode_ms") or 0
        rate = r.get("decode_tok_s")
        if rate is None and decode_ms and made:
            rate = round(made / (decode_ms / 1000.0), 2)
        rows.append({
            "ts": began + took, "job_id": server.get("job_id"),
            "node": server.get("node"), "port": server.get("port"),
            "model": server.get("model"), "task_id": int(round(began * 1000)),
            "prompt_tokens": read, "generated_tokens": made,
            "total_tokens": read + made,
            "context_tokens": (r.get("prompt_total") or r.get("prompt_tokens") or 0) + made,
            "prompt_ms": prompt_ms, "eval_ms": decode_ms, "total_ms": took * 1000.0,
            "prompt_tok_s": round(read / (prompt_ms / 1000.0), 2) if prompt_ms else 0,
            "generated_tok_s": rate or 0, "truncated": 0})
    return db.insert_llm_requests(rows)


def _log_path(job_id):
    """`/data/llm/logs/<anything>-<jobid>.out`, whichever job script wrote it."""
    if job_id is None:
        return None
    directory = getattr(config, "LLM_LOG_DIR", "/data/llm/logs")
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    suffix = "-%s.out" % job_id
    for name in names:
        if name.endswith(suffix):
            return os.path.join(directory, name)
    return None


def _probe(record):
    try:
        listing = _get_json(record["url"] + "/v1/models")
    except Exception as exc:                                    # noqa: BLE001
        record["error"] = "unreachable: %s" % exc
        return
    record["ok"] = True
    for entry in listing.get("data") or []:
        status = entry.get("status") or {}
        state = status.get("value") if isinstance(status, dict) else None
        meta = entry.get("meta") or {}
        model = {"id": entry.get("id"), "state": state,
                 "loaded": state == "loaded",
                 "ctx": meta.get("n_ctx") or _arg(status, "--ctx-size", int),
                 "port": _arg(status, "--port", int),
                 "params": meta.get("n_params"),
                 "size_bytes": meta.get("size"),
                 "quant": meta.get("ftype")}
        model["path"] = _arg(status, "--model")
        # A managed router points at its node-local link farm, which does not
        # exist here on the head; the same name under models/servable does.
        if model["path"] and model["path"].startswith("/var/cache/slerm-llm/servable/"):
            model["path"] = "/data/llm/models/servable/" + os.path.basename(model["path"])
        if not model["size_bytes"]:
            # Only a resident model has router meta; read the rest off disk.
            disk = ggufmeta.lookup(model["path"])
            if disk:
                model["size_bytes"] = disk["size"]
                model["params"] = model["params"] or disk["n_params"]
                model["quant"] = model["quant"] or disk["ftype"]
        record["models"].append(model)
        if model["loaded"]:
            record["loaded"] = model["id"]
    record["loading"] = _track_loads(record)
    if record["loaded"]:
        try:
            record["metrics"] = metrics(record["url"], record["loaded"])
        except Exception as exc:                                # noqa: BLE001
            record["metrics_error"] = str(exc)


def _arg(status, flag, cast=str):
    """Read one flag out of the argv llama.cpp reports for a model instance."""
    args = (status or {}).get("args") if isinstance(status, dict) else None
    if not isinstance(args, list):
        return None
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            try:
                return cast(args[i + 1])
            except (TypeError, ValueError):
                return None
    return None


_METRICS = {
    "llamacpp:prompt_tokens_total": "prompt_tokens",
    "llamacpp:prompt_tokens_cached_total": "cached_tokens",
    "llamacpp:tokens_predicted_total": "generated_tokens",
    "llamacpp:prompt_seconds_total": "prompt_seconds",
    "llamacpp:tokens_predicted_seconds_total": "generated_seconds",
    "llamacpp:n_decode_total": "decodes",
    "llamacpp:n_tokens_max": "longest_sequence",
    "llamacpp:requests_processing": "requests_processing",
    "llamacpp:requests_deferred": "requests_deferred",
    "llamacpp:kv_cache_usage_ratio": "kv_cache_ratio",
    "llamacpp:kv_cache_tokens": "kv_cache_tokens",
    # These two are NOT throughput. llama.cpp computes them from a bucket it
    # RESETS on every scrape, so any poll following another with no traffic
    # between reads 0 -- measured: two requests a second apart, both 0, while
    # the cumulative counters sat at 540 and 226. The rates below are derived
    # from the totals instead; these are kept as "since the last scrape".
    "llamacpp:prompt_tokens_seconds": "prompt_tok_s_recent",
    "llamacpp:predicted_tokens_seconds": "generated_tok_s_recent",
}


def metrics(url, model):
    """Cumulative counters for one model instance.

    The router serves one model at a time but exposes metrics per model, so the
    alias must be named or it answers HTTP 400.
    """
    # autoload=false (2026-10-17): asking about a model the router just unloaded made it LOAD that model again, so
    # the idle unload never stuck and node04 never reached its idle power-off (17 unload/reload rounds in job 4590).
    text = _get_text("%s/metrics?model=%s&autoload=false" % (url, urllib.parse.quote(str(model))))
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        key = _METRICS.get(parts[0])
        if not key:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        out[key] = int(value) if value.is_integer() else round(value, 2)
    for tokens, seconds, rate in (
            ("prompt_tokens", "prompt_seconds", "prompt_tok_s"),
            ("generated_tokens", "generated_seconds", "generated_tok_s")):
        spent = out.get(seconds) or 0
        out[rate] = round((out.get(tokens) or 0) / spent, 1) if spent else 0
    return out


# --- live progress ----------------------------------------------------------
# What a server is doing THIS second, for the progress bars on the AI tab: a
# model being read in, a prompt being evaluated, tokens being generated.
#
# Model loading has no progress counter anywhere -- the router only reports
# progress for downloads -- so it is an ESTIMATE: time since the panel first
# saw the model in "loading", against the GGUF's size at the NFS read rate
# (21.7 GiB took 8 min = ~46 MB/s, 2026-09-25). A load the panel watches from
# start to finish replaces that rate with a measured one.
#
# Prompt evaluation is exact but coarse: /slots updates
# n_prompt_tokens_processed once per batch (2048 tokens), so the bar moves in
# steps -- ~65 s apart on gemma4-26b-a4b at 31 tok/s.

_DEFAULT_LOAD_BPS = 46e6
_loads = {}             # (job_id, model) -> {"since", "exact", "bytes"}
_load_rate = {"bps": _DEFAULT_LOAD_BPS, "measured": None}
_last_seen = {}         # job_id -> (ts, set of models loading at that probe)
_tasks = {}             # (job_id, model, id_task) -> rate samples
_live_lock = threading.Lock()
_live_cache = {"ts": 0.0, "data": None}
LIVE_TTL = 1.0
LIVE_TIMEOUT = 3


def _gguf_bytes(path):
    """Size of a model on disk, counting every shard of a split GGUF."""
    if not path:
        return None
    try:
        m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", path)
        if not m:
            return os.path.getsize(path)
        return sum(os.path.getsize("%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(2)))
                   for i in range(1, int(m.group(2)) + 1))
    except OSError:
        return None


def _track_loads(record):
    """Remember when each model entered "loading", and learn the load rate."""
    now = time.time()
    job = record.get("job_id")
    loading = {m["id"] for m in record["models"] if m.get("state") == "loading"}
    prev_ts, prev_loading = _last_seen.get(job, (None, None))
    # "exact" = the previous probe of this job, recently, saw it NOT loading,
    # so the start is known to within one probe interval.
    recent = prev_ts is not None and now - prev_ts < 3 * config.LLM_POLL_SECONDS
    for model in record["models"]:
        key = (job, model["id"])
        if model["id"] in loading and key not in _loads:
            _loads[key] = {"since": now,
                           "exact": recent and model["id"] not in prev_loading,
                           "bytes": _gguf_bytes(model.get("path"))}
        elif model["id"] not in loading and key in _loads:
            done = _loads.pop(key)
            took = now - done["since"]
            if model.get("loaded") and done["exact"] and done["bytes"] and took > 20:
                _load_rate["bps"] = done["bytes"] / took
                _load_rate["measured"] = {"model": model["id"], "seconds": round(took),
                                          "at": now}
    _last_seen[job] = (now, loading)
    out = []
    for model_id in sorted(loading):
        info = _loads[(job, model_id)]
        elapsed = now - info["since"]
        total = info["bytes"]
        expected = total / _load_rate["bps"] if total else None
        out.append({
            "model": model_id, "since": info["since"],
            "elapsed_s": round(elapsed), "exact_start": info["exact"],
            "bytes": total,
            "expected_s": round(expected) if expected else None,
            # Never claim 100% of a guess: hold at 99 until the router says loaded.
            "pct": round(min(99.0, 100.0 * elapsed / expected), 1) if expected else None,
            "rate_mb_s": round(_load_rate["bps"] / 1e6, 1),
            "rate_measured": _load_rate["measured"] is not None,
        })
    return out


def _slot_rates(key, slot, now):
    """Prompt and generation rates for one task, from changes the panel saw.

    Measured between the first and latest CHANGE of each counter, not from
    first sight: processed advances in batch-sized jumps, and timing from
    first sight would count the wait before the first jump as zero throughput.
    """
    t = _tasks.get(key)
    if t is None:
        t = _tasks[key] = {"p": [], "d": []}
    for field, name in (("n_prompt_tokens_processed", "p"), ("n_decoded", "d")):
        value = slot.get(field) or 0
        seq = t[name]
        if not seq or seq[-1][1] != value:
            seq.append((now, value))
            if len(seq) > 2:
                seq[1:-1] = []      # keep the first change and the latest
    rates = {}
    for name in ("p", "d"):
        seq = t[name]
        if len(seq) >= 3 or (len(seq) == 2 and seq[0][1] > 0):
            (t0, v0), (t1, v1) = seq[0], seq[-1]
            if t1 > t0 and v1 > v0:
                rates[name] = (v1 - v0) / (t1 - t0)
    return rates.get("p"), rates.get("d")


def _live_server(record, now):
    out = {"job_id": record.get("job_id"), "name": record.get("name"),
           "node": record.get("node"), "ok": record.get("ok"),
           "loaded": record.get("loaded"),
           "loading": record.get("loading") or [], "slots": [],
           "queued": None, "processing": None}
    model = record.get("loaded")
    if not model:
        return out
    q = urllib.parse.quote(str(model))
    m = record.get("metrics")
    if m:
        out["queued"] = m.get("requests_deferred")
        out["processing"] = m.get("requests_processing")
    try:
        slots = json.loads(_request("%s/slots?model=%s&autoload=false" % (record["url"], q),
                                    timeout=LIVE_TIMEOUT))
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = "slots: %s" % exc
        return out
    for slot in slots if isinstance(slots, list) else []:
        if not slot.get("is_processing"):
            continue
        nt = (slot.get("next_token") or [{}])[0]
        key = (record.get("job_id"), model, slot.get("id_task"))
        p_rate, d_rate = _slot_rates(key, slot, now)
        total = slot.get("n_prompt_tokens") or 0
        done = slot.get("n_prompt_tokens_processed") or 0
        cached = slot.get("n_prompt_tokens_cache") or 0
        decoded = nt.get("n_decoded") or 0
        remain = nt.get("n_remain")
        prompting = total > 0 and done < total and decoded == 0
        p_rate = p_rate or (m and m.get("prompt_tok_s")) or None
        d_rate = d_rate or (m and m.get("generated_tok_s")) or None
        entry = {"id": slot.get("id"), "task": slot.get("id_task"),
                 "n_ctx": slot.get("n_ctx"),
                 "phase": "prompt" if prompting else "generating",
                 "prompt_tokens": total, "prompt_done": done,
                 "prompt_cached": cached, "decoded": decoded,
                 "max_new": (decoded + remain) if isinstance(remain, int) and remain >= 0 else None,
                 "prompt_tok_s": round(p_rate, 1) if p_rate else None,
                 "gen_tok_s": round(d_rate, 1) if d_rate else None}
        if prompting and p_rate:
            entry["eta_s"] = round((total - done) / p_rate)
        elif not prompting and entry["max_new"] and d_rate:
            entry["eta_s"] = round((entry["max_new"] - decoded) / d_rate)
        out["slots"].append(entry)
    return out


def live():
    """Progress of whatever every server is doing now. Cheap, cached ~1 s.

    One fetch at a time: a server that is slow to answer /slots mid-batch gets
    the previous answer again, not a pile of concurrent requests.
    """
    now = time.time()
    if _live_cache["data"] is not None and now - _live_cache["ts"] < LIVE_TTL:
        return _live_cache["data"]
    if not _live_lock.acquire(False):
        return _live_cache["data"] or {"now": now, "servers": []}
    try:
        out = []
        # Discovery stays on the 5 s cache (it runs squeue); only the router
        # itself is asked again, on a copy so the cached record is untouched.
        for cached in servers():
            if not cached.get("url"):
                continue
            record = dict(cached, ok=False, error=None, models=[],
                          loaded=None, metrics=None)
            _probe(record)
            out.append(_live_server(record, now))
        for cached in strata_servers():
            record = dict(cached)
            if not record.get("error"):
                _strata_refresh(record)
            out.append(_strata_live(record))
        live_tasks = {(s["job_id"], s["loaded"], sl["task"])
                      for s in out for sl in s["slots"]}
        for key in [k for k in _tasks if k not in live_tasks]:
            del _tasks[key]
        data = {"now": now, "servers": out,
                "load_rate": {"mb_s": round(_load_rate["bps"] / 1e6, 1),
                              "measured": _load_rate["measured"]}}
        _live_cache["ts"] = time.time()
        _live_cache["data"] = data
        return data
    finally:
        _live_lock.release()


# --- control -----------------------------------------------------------------

class UnloadError(Exception):
    pass


def unload(job_id, model):
    """Ask a router to drop its resident model, freeing the card(s).

    Both the job and the model must match what is resident right now, so a
    click on a stale card cannot evict whatever was swapped in since. Anything
    decoding on that model is cut off; the caller warns about that.
    """
    target = None
    for server in servers(refresh=True):
        if str(server.get("job_id")) == str(job_id):
            target = server
            break
    if target is None or not target.get("url"):
        raise UnloadError("server job %s is not running" % job_id)
    if not target.get("loaded"):
        raise UnloadError("nothing is loaded on %s" % target.get("node"))
    if target["loaded"] != model:
        raise UnloadError("%s is loaded now, not %s" % (target["loaded"], model))
    _refuse_if_locked(model, "unload")
    return _router_unload(target, model)


def stop_loading(job_id, model):
    """Abandon a model that is still being read in.

    The router's /models/unload force-kills an instance in "loading" (the same
    call as unload(); llama.cpp special-cases the state). Refused unless that
    model is loading on that job right now, so a stale button cannot evict a
    model that finished loading since. Requests waiting for the model fail.
    """
    target = None
    for server in servers(refresh=True):
        if str(server.get("job_id")) == str(job_id):
            target = server
            break
    if target is None or not target.get("url"):
        raise UnloadError("server job %s is not running" % job_id)
    state = next((m.get("state") for m in target.get("models") or []
                  if m.get("id") == model), None)
    if state != "loading":
        raise UnloadError("%s is %s now, not loading" % (model, state or "unknown"))
    _refuse_if_locked(model, "stop")
    result = _router_unload(target, model)
    _loads.pop((target["job_id"], model), None)
    return result


def load_model(job_id, model):
    """Ask a router to load a model now rather than on its first request.

    With --models-max 1 the router first evicts whatever is resident (waiting
    up to its 10 s force-kill), which cuts off anything running on it -- the
    caller warns about that. Refused unless the model is unloaded on that job.
    """
    target = None
    for server in servers(refresh=True):
        if str(server.get("job_id")) == str(job_id):
            target = server
            break
    if target is None or not target.get("url"):
        raise UnloadError("server job %s is not running" % job_id)
    state = next((m.get("state") for m in target.get("models") or []
                  if m.get("id") == model), None)
    if state is None:
        raise UnloadError("%s is not a model job %s can serve" % (model, job_id))
    if state != "unloaded":
        raise UnloadError("%s is %s already" % (model, state))
    _refuse_if_locked(model, "load")
    result = _router_call(target, "/models/load", model, timeout=45)
    result["evicted"] = target.get("loaded")
    return result


def _router_unload(target, model):
    return _router_call(target, "/models/unload", model)


def _router_call(target, path, model, timeout=None):
    body = json.dumps({"model": model}).encode()
    req = urllib.request.Request(target["url"] + path, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(
                req, timeout=timeout or getattr(config, "LLM_SERVER_TIMEOUT", 6)) as resp:
            answer = json.loads(resp.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:                                       # noqa: BLE001
            pass
        raise UnloadError("router refused: %s" % detail)
    except Exception as exc:                                    # noqa: BLE001
        raise UnloadError("router unreachable: %s" % exc)
    _servers_cache["data"] = None
    _live_cache["data"] = None
    return {"job_id": target["job_id"], "node": target["node"],
            "model": model, "router": answer}


# --- sampling ----------------------------------------------------------------

def poll():
    """One sampling pass over every running server. Never raises."""
    if not _poll_lock.acquire(False):
        return {"skipped": "a poll is already running"}
    try:
        found = 0
        for server in servers(refresh=True):
            if server.get("job_id") is None:
                continue
            found += _tail(server)
        for server in strata_servers():
            found += _strata_ingest(server)
        _state["last_poll"] = time.time()
        _state["polls"] += 1
        _state["requests"] += found
        _state["last_error"] = None
        return {"requests": found}
    except Exception as exc:                                    # noqa: BLE001
        _state["last_error"] = "%s: %s" % (type(exc).__name__, exc)
        return {"error": _state["last_error"]}
    finally:
        _poll_lock.release()


def _tail(server):
    """Read whatever has been appended to one server log since last time."""
    path = server.get("log")
    if not path:
        return 0
    try:
        stat = os.stat(path)
    except OSError:
        return 0
    cursor = db.get_log_cursor(path) or {}
    offset = cursor.get("offset") or 0
    inode = cursor.get("inode")
    fresh = inode != stat.st_ino
    if fresh or offset > stat.st_size:
        # A new file under a reused name, or one that was truncated. Start at
        # the end: what a log already held when the panel first saw it has no
        # trustworthy timestamps, so it is skipped rather than invented.
        _parsers.pop(path, None)
        offset = stat.st_size
        db.save_log_cursor(path, stat.st_ino, offset, server.get("job_id"))
        return 0
    if offset == stat.st_size:
        # Nothing new, but a parser may be holding a request whose closing
        # lines were split across the previous read. An empty chunk releases
        # it, so it is reported one poll late at worst rather than waiting for
        # the server to be used again.
        parser = _parsers.get(path)
        if parser is None or not parser.done:
            return 0
        held, _events = parser.feed("")
        for record in held:
            record.update({"ts": time.time(), "job_id": server.get("job_id"),
                           "node": server.get("node")})
            record.setdefault("model", server.get("loaded"))
        return db.insert_llm_requests(held)
    limit = getattr(config, "LLM_LOG_MAX_READ", 4 * 1024 * 1024)
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(limit)
    except OSError:
        return 0
    parser = _parsers.get(path)
    if parser is None:
        parser = _parsers[path] = llmlog.Parser(_ports_from(server))
    else:
        parser.port_model.update(_ports_from(server))
    requests, events = parser.feed(chunk.decode("utf-8", "replace"))
    now = time.time()
    for record in requests:
        record.update({"ts": now, "job_id": server.get("job_id"),
                       "node": server.get("node")})
        record.setdefault("model", server.get("loaded"))
    for event in events:
        event.update({"ts": now, "job_id": server.get("job_id"),
                      "node": server.get("node")})
    stored = db.insert_llm_requests(requests)
    db.insert_llm_events(events)
    db.save_log_cursor(path, stat.st_ino, offset + len(chunk),
                       server.get("job_id"))
    if requests and stored < len(requests):
        # Either a genuine re-read (harmless) or a row the database refused.
        # Either way, count what LANDED, never what was parsed: reporting the
        # parsed figure is how a silently rejected insert stays invisible.
        _state["rejected"] = _state.get("rejected", 0) + len(requests) - stored
    return stored


def _ports_from(server):
    """Port -> alias, straight from the server's own model listing."""
    out = {}
    for model in server.get("models") or []:
        if model.get("port") and model.get("id"):
            out[int(model["port"])] = model["id"]
    return out


def start():
    """Background sampler. The tab is a view of what this collects."""
    interval = getattr(config, "LLM_POLL_SECONDS", 20)
    if not interval:
        return
    _state["started"] = time.time()

    def loop():
        while True:
            poll()
            time.sleep(interval)

    threading.Thread(target=loop, name="llm-usage", daemon=True).start()


def prune():
    try:
        return db.prune_llm(getattr(config, "LLM_KEEP_DAYS", 0))
    except Exception:                                           # noqa: BLE001
        return 0


# --- reading -----------------------------------------------------------------

def window_seconds(spec):
    return WINDOWS.get(spec, WINDOWS[DEFAULT_WINDOW])


def _since(spec):
    seconds = window_seconds(spec)
    return None if seconds is None else time.time() - seconds


def _where(since, extra=None, args=None):
    clauses = []
    values = list(args or [])
    if since is not None:
        clauses.append("ts >= ?")
        values.insert(0, since)
    if extra:
        clauses.extend(extra)
    return ((" WHERE " + " AND ".join(clauses)) if clauses else "",
            tuple(values))


_AGG = ("COUNT(*) AS calls,"
        " SUM(prompt_tokens) AS prompt_tokens,"
        " SUM(generated_tokens) AS generated_tokens,"
        " SUM(total_tokens) AS total_tokens,"
        " SUM(truncated) AS truncated,"
        " SUM(prompt_ms) AS prompt_ms,"
        " SUM(eval_ms) AS eval_ms,"
        " SUM(total_ms) AS total_ms,"
        " MAX(context_tokens) AS longest_context,"
        " MIN(ts) AS first_ts, MAX(ts) AS last_ts")

_COUNTERS = ("calls", "prompt_tokens", "generated_tokens", "total_tokens",
             "truncated", "prompt_ms", "eval_ms", "total_ms",
             "longest_context", "models")


def _finish(row):
    """Derived rates. SUM() over no rows is NULL, so counters are coerced.

    Only the counters: a NULL in a label column means something else and is
    left alone.
    """
    out = dict(row or {})
    for key in _COUNTERS:
        if key in out:
            out[key] = out[key] or 0
    calls = out.get("calls") or 0
    eval_s = (out.pop("eval_ms", 0) or 0) / 1000.0
    prompt_s = (out.pop("prompt_ms", 0) or 0) / 1000.0
    total_s = (out.pop("total_ms", 0) or 0) / 1000.0
    out["busy_s"] = round(total_s, 1)
    out["avg_s"] = round(total_s / calls, 1) if calls else 0
    # Tokens over the time actually spent on that phase, which is the number
    # that predicts how long the next call takes.
    out["generated_tok_s"] = (round((out.get("generated_tokens") or 0) / eval_s, 1)
                              if eval_s > 0 else 0)
    out["prompt_tok_s"] = (round((out.get("prompt_tokens") or 0) / prompt_s, 1)
                           if prompt_s > 0 else 0)
    out["avg_prompt_tokens"] = (round((out.get("prompt_tokens") or 0) / calls)
                                if calls else 0)
    out["avg_generated_tokens"] = (round((out.get("generated_tokens") or 0) / calls)
                                   if calls else 0)
    return out


def totals(since):
    where, args = _where(since)
    row = db.one("SELECT %s, COUNT(DISTINCT model) AS models"
                 " FROM llm_requests%s" % (_AGG, where), args)
    return _finish(row or {})


def by_model(since, limit=40):
    where, args = _where(since)
    rows = db.query(
        "SELECT model, %s FROM llm_requests%s GROUP BY model"
        " ORDER BY total_tokens DESC, calls DESC LIMIT ?" % (_AGG, where),
        args + (limit,))
    return [_finish(r) for r in rows]


def by_node(since, limit=20):
    where, args = _where(since)
    rows = db.query(
        "SELECT node, model, %s FROM llm_requests%s GROUP BY node, model"
        " ORDER BY total_tokens DESC LIMIT ?" % (_AGG, where),
        args + (limit,))
    return [_finish(r) for r in rows]


def by_size(since):
    """How the calls are distributed by generated length.

    A mean hides the shape completely here: the same model serves a 1-token
    title generation and a 2,500-token code answer, and only one of those is
    why the GPU is busy.
    """
    buckets = [(0, 64, "under 64"), (64, 256, "64-256"), (256, 1024, "256-1k"),
               (1024, 4096, "1k-4k"), (4096, None, "over 4k")]
    where, args = _where(since)
    rows = db.query("SELECT generated_tokens, total_tokens, total_ms"
                    " FROM llm_requests%s" % where, args)
    out = []
    for low, high, label in buckets:
        hit = [r for r in rows
               if (r["generated_tokens"] or 0) >= low
               and (high is None or (r["generated_tokens"] or 0) < high)]
        out.append({"bucket": label, "calls": len(hit),
                    "generated_tokens": sum(r["generated_tokens"] or 0 for r in hit),
                    "total_tokens": sum(r["total_tokens"] or 0 for r in hit),
                    "busy_s": round(sum(r["total_ms"] or 0 for r in hit) / 1000.0, 1)})
    return out


def series(since, tz_name=None):
    """Tokens per bucket, in the panel's display zone.

    Aggregated in SQL at hour granularity -- a bounded number of rows whatever
    the window -- then folded into local days here, so a DST day really is 23
    or 25 hours rather than a nominal 24.
    """
    where, args = _where(since)
    hours = db.query(
        "SELECT CAST(ts / %d AS INTEGER) AS bucket, COUNT(*) AS calls,"
        " SUM(prompt_tokens) AS prompt_tokens,"
        " SUM(generated_tokens) AS generated_tokens,"
        " SUM(total_tokens) AS total_tokens"
        " FROM llm_requests%s GROUP BY bucket ORDER BY bucket LIMIT %d"
        % (HOUR, where, _MAX_BUCKETS), args)
    if not hours:
        return {"bucket": "hour", "points": []}
    span = (hours[-1]["bucket"] - hours[0]["bucket"]) * HOUR
    if span <= 3 * 86400:
        return {"bucket": "hour",
                "points": [{"t": h["bucket"] * HOUR, "calls": h["calls"],
                            "prompt_tokens": h["prompt_tokens"] or 0,
                            "generated_tokens": h["generated_tokens"] or 0,
                            "total_tokens": h["total_tokens"] or 0}
                           for h in hours]}
    tz = _zone(tz_name or config.DISPLAY_TZ)
    days = {}
    for h in hours:
        start = datetime.fromtimestamp(h["bucket"] * HOUR, tz)
        key = start.date().isoformat()
        point = days.setdefault(key, {
            "t": int(datetime(start.year, start.month, start.day,
                              tzinfo=tz).timestamp()),
            "day": key, "calls": 0, "prompt_tokens": 0,
            "generated_tokens": 0, "total_tokens": 0})
        for field in ("calls", "prompt_tokens", "generated_tokens",
                      "total_tokens"):
            point[field] += h[field] or 0
    return {"bucket": "day", "points": [days[k] for k in sorted(days)]}


def _zone(name):
    try:
        return ZoneInfo(name)
    except Exception:                                           # noqa: BLE001
        return ZoneInfo("UTC")


def requests(since=None, limit=50, model=None):
    extra, args = [], []
    if model:
        extra.append("model = ?")
        args.append(model)
    where, values = _where(since, extra, args)
    try:
        limit = min(max(1, int(limit)), 1000)
    except (TypeError, ValueError):
        limit = 50
    rows = db.query("SELECT * FROM llm_requests%s ORDER BY ts DESC, id DESC"
                    " LIMIT ?" % where, values + (limit,))
    for row in rows:
        row["total_s"] = round((row.pop("total_ms", 0) or 0) / 1000.0, 1)
        row["prompt_s"] = round((row.pop("prompt_ms", 0) or 0) / 1000.0, 1)
        row["eval_s"] = round((row.pop("eval_ms", 0) or 0) / 1000.0, 1)
        row["truncated"] = bool(row.get("truncated"))
    return rows


def events(since=None, limit=30):
    where, args = _where(since)
    return db.query("SELECT * FROM llm_model_events%s ORDER BY ts DESC LIMIT ?"
                    % where, args + (limit,))


def collecting_since():
    """When this panel started watching, so an empty tab explains itself."""
    row = db.one("SELECT MIN(updated) AS first FROM llm_log_cursors")
    return (row or {}).get("first")


def overview(window=DEFAULT_WINDOW):
    since = _since(window)
    live = servers() + strata_servers()
    return {
        "window": window if window in WINDOWS else DEFAULT_WINDOW,
        "windows": sorted(WINDOWS, key=lambda k: (WINDOWS[k] is None,
                                                  WINDOWS[k] or 0)),
        "since": since,
        "now": time.time(),
        "collecting_since": collecting_since(),
        "keep_days": getattr(config, "LLM_KEEP_DAYS", 0),
        "sampler": {"interval_s": getattr(config, "LLM_POLL_SECONDS", 20),
                    "last_poll": _state["last_poll"],
                    "polls": _state["polls"],
                    "observed": _state["requests"],
                    "error": _state["last_error"]},
        "totals": totals(since),
        "by_model": by_model(since),
        "by_node": by_node(since),
        "by_size": by_size(since),
        "series": series(since),
        "requests": requests(since, limit=40),
        "events": events(since, limit=20),
        "servers": live,
        "catalogue": _catalogue(live),
        "orchestrator": orchestrator_summary(),
    }


def _catalogue(live):
    """Every model the cluster can serve, with what it has actually done.

    A model that is configured but never called is worth seeing: it is the
    difference between "we host thirteen models" and "we use two".
    """
    used = {}
    for row in db.query("SELECT model, COUNT(*) AS calls,"
                        " SUM(total_tokens) AS total_tokens,"
                        " MAX(ts) AS last_ts FROM llm_requests"
                        " GROUP BY model"):
        used[row["model"]] = row
    out = []
    seen = set()
    for server in live:
        for model in server.get("models") or []:
            key = model.get("id")
            if not key or key in seen:
                continue
            seen.add(key)
            stat = used.get(key) or {}
            entry = dict(model)
            entry.update({"calls": stat.get("calls") or 0,
                          "total_tokens": stat.get("total_tokens") or 0,
                          "last_ts": stat.get("last_ts"),
                          "node": server.get("node"),
                          "job_id": server.get("job_id"),
                          "engine": server.get("engine") or "llama.cpp"})
            out.append(entry)
    # What the orchestrator says about each model: whether it can run on the
    # devices that are reachable right now (the user's VMs, deadlock's own
    # work, batch jobs and the UPS all take devices away), and why not.
    orch = _orch_state()
    avail = orch.get("models") or {}
    cached = _cached_models(orch)
    staged = {s.get("model"): s for s in orch.get("staged") or []}
    for entry in out:
        entry["avail"] = avail.get(entry.get("id"))
        entry["cached"] = entry.get("id") in cached
    for key, a in sorted(avail.items()):
        if key in seen:
            continue
        # Not on the running router (its device set cannot hold it, or no
        # router is up): a card anyway, so the user sees it and why.
        seen.add(key)
        stat = used.get(key) or {}
        out.append({"id": key, "state": "cold" if a.get("reachable") else "unavailable",
                    "loaded": False, "avail": a, "cached": key in cached,
                    "calls": stat.get("calls") or 0,
                    "total_tokens": stat.get("total_tokens") or 0,
                    "last_ts": stat.get("last_ts")})
    for key, stat in used.items():
        if key and key not in seen:
            # Served by a job that has since ended; its history still counts.
            out.append({"id": key, "state": "gone", "loaded": False,
                        "calls": stat.get("calls") or 0,
                        "total_tokens": stat.get("total_tokens") or 0,
                        "last_ts": stat.get("last_ts")})
    for entry in out:
        entry["staged"] = staged.get(entry.get("id"))
    out.sort(key=lambda m: (-(m.get("total_tokens") or 0), m.get("id") or ""))
    return out


def _orch_state():
    path = getattr(config, "LLM_ORCH_STATE", None)
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, TypeError, ValueError):
        return {}
    # A stale file means the orchestrator is down: say nothing rather than
    # grey out cards on old information.
    if time.time() - (state.get("updated") or 0) > 120:
        return {}
    return state


def _cached_models(orch):
    """Models with a local copy on the router's node (bin/model-cache.py)."""
    node = ((orch.get("router") or {}).get("node")) or ""
    try:
        with open("/data/llm/run/cache-%s.json" % node) as f:
            return set((json.load(f).get("cached") or {}).keys())
    except (OSError, ValueError):
        return set()


def orchestrator_summary():
    o = _orch_state()
    if not o:
        return None
    out = {k: o.get(k) for k in ("idle_s", "inflight", "queue", "ups", "devices",
                                  "power_control", "gateway", "policy", "router",
                                  "strata", "staged", "stage_max", "lock", "lock_nodes")}
    # An orchestrator older than the model lock publishes no "lock" key at all:
    # the UI hides the button rather than offer one that answers 404.
    out["lock_supported"] = "lock" in o
    return out


# --- model lock (2026-10-17) ---------------------------------------------------
# Held by slerm-orch (POST /lock, /unlock): while a model is locked it is never
# unloaded or idled out, other models are refused, and the nodes it runs on stay
# on. The router-direct calls below (Load, Unload, Stop loading) bypass the
# orchestrator, so they check the lock themselves.

def current_lock():
    return (_orch_state() or {}).get("lock") or None


def _refuse_if_locked(model, verb):
    lock = current_lock()
    if not lock:
        return
    if verb in ("unload", "stop") and model == lock.get("model"):
        raise UnloadError("%s is locked: unlock it first" % model)
    if verb == "load" and model != lock.get("model"):
        raise UnloadError("slerm is locked to %s: unlock it first" % lock.get("model"))


def lock_model(model=None):
    body = {"model": model} if model else {}
    return _orch_call("/lock", body)


def unlock_model():
    return _orch_call("/unlock", {})


def bring_up(model):
    """Ask the orchestrator to make MODEL servable (returns immediately)."""
    return _orch_post("/ensure", model)


def stage(model):
    """Copy MODEL onto the nodes it will load on, ahead of use (returns at once)."""
    return _orch_post("/stage", model)


def unstage(model):
    return _orch_post("/unstage", model)


def _orch_post(path, model):
    return _orch_call(path, {"model": model})


def _orch_call(path, body):
    url = getattr(config, "LLM_ORCH_CONTROL", "http://127.0.0.1:8091") + path
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read()).get("error")
        except ValueError:
            msg = str(exc)
        raise UnloadError(msg or str(exc))
    except OSError as exc:
        raise UnloadError("orchestrator unreachable: %s" % exc)


# --- http --------------------------------------------------------------------

def _request(url, timeout=None):
    timeout = timeout or getattr(config, "LLM_SERVER_TIMEOUT", 6)
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _get_text(url):
    return _request(url)


def _get_json(url):
    return json.loads(_request(url))
