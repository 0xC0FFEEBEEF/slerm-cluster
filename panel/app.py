"""Slerm Panel -- internal control UI for the cluster and its projects."""
import fcntl
import functools
import os
import sys
import threading
import time

from flask import Flask, jsonify, request, send_file, render_template
from werkzeug.exceptions import RequestEntityTooLarge

import config
import db
import gifbuild
import comfy
import llmusage
import projects
import runs
import scheduler
import slurmapi
import wazuh
import wazuhapi
from projects.base import ProjectError

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
# Reject oversized multipart bodies before Werkzeug spools them. The extra
# MiB allows for multipart headers while _save_upload enforces the exact file
# byte limit on the uploaded stream itself.
app.config["MAX_CONTENT_LENGTH"] = int(config.WAZUH_UPLOAD_MAX_BYTES) + 1024 * 1024

_memo = {}
_memo_lock = threading.Lock()


def cached(seconds):
    """Tiny TTL memo so several open browser tabs don't hammer slurmctld."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            with _memo_lock:
                hit = _memo.get(key)
                if hit and hit[0] > now:
                    return hit[1]
            value = fn(*args, **kwargs)
            with _memo_lock:
                _memo[key] = (now + seconds, value)
            return value
        return wrapper
    return decorator


def api_error(exc, status=400):
    return jsonify({"error": str(exc), "type": type(exc).__name__}), status


@app.errorhandler(ProjectError)
def _handle_project_error(exc):
    return api_error(exc, 400)


@app.errorhandler(RequestEntityTooLarge)
def _handle_request_too_large(exc):
    return api_error(exc, 413)


@app.errorhandler(slurmapi.SlurmError)
def _handle_slurm_error(exc):
    return api_error(exc, 502)


@app.errorhandler(wazuhapi.WazuhError)
def _handle_wazuh_api_error(exc):
    return api_error(exc, 502)


@app.errorhandler(scheduler.ScheduleError)
def _handle_schedule_error(exc):
    return api_error(exc, 400)


# --- page -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": time.time(),
                    "scheduler": scheduler.status()})


# --- cluster ----------------------------------------------------------------

@cached(1)
def _nodes():
    return slurmapi.nodes()


@cached(5)
def _partitions():
    return slurmapi.partitions()


@cached(1)
def _queue():
    return slurmapi.queue()


@cached(15)
def _history(days):
    return slurmapi.history(days=days)


@cached(10)
def _diag():
    return slurmapi.diag()


@app.route("/api/cluster")
def api_cluster():
    """Everything the cluster tab needs, in one round trip."""
    nodes = _nodes()
    return jsonify({
        "nodes": nodes,
        "partitions": _partitions(),
        "queue": _queue(),
        "head": slurmapi.head_stats(),
        "diag": _diag(),
        "probes": db.list_probes(),
        # idle clock + gateway chip; None while the orchestrator is not publishing
        "orch": llmusage.orchestrator_summary(),
        "summary": {
            "nodes_total": len(nodes),
            "nodes_healthy": sum(1 for n in nodes if n["healthy"]),
            "cpus_total": sum(n["cpus"] for n in nodes),
            "cpus_alloc": sum(n["alloc_cpus"] for n in nodes),
            "gpus_total": sum(g["count"] for n in nodes for g in n["gres"]
                              if g["type"] == "gpu"),
            "gpus_used": sum(g.get("used", 0) for n in nodes for g in n["gres"]
                             if g["type"] == "gpu"),
        },
        "time": time.time(),
    })


@app.route("/api/cluster/nodes")
def api_nodes():
    return jsonify({"nodes": _nodes()})


@app.route("/api/cluster/queue")
def api_queue():
    return jsonify({"queue": _queue()})


@app.route("/api/cluster/history")
def api_history():
    days = int(request.args.get("days", config.HISTORY_DAYS))
    return jsonify({"jobs": _history(min(days, 30))})


@app.route("/api/cluster/gpu-probe", methods=["POST"])
def api_gpu_probe():
    body = request.get_json(silent=True) or {}
    wanted = body.get("nodes")
    if not wanted:
        wanted = [n["name"] for n in _nodes()
                  if any(g["type"] == "gpu" for g in n["gres"])]
    for node in wanted:
        threading.Thread(target=_probe_one, args=(node,),
                         name="probe-%s" % node, daemon=True).start()
    return jsonify({"probing": wanted})


def _probe_one(node):
    try:
        gpus = slurmapi.gpu_probe(node)
        db.save_probe(node, True, data=gpus)
    except slurmapi.SlurmError as exc:
        db.save_probe(node, False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        db.save_probe(node, False, error="%s: %s" % (type(exc).__name__, exc))


@app.route("/api/cluster/probes")
def api_probes():
    return jsonify({"probes": db.list_probes()})


@app.route("/api/jobs/<int:job_id>/cancel", methods=["POST"])
def api_cancel_job(job_id):
    slurmapi.cancel(job_id)
    return jsonify({"cancelled": job_id})


# --- projects ---------------------------------------------------------------

@app.route("/api/projects")
def api_projects():
    return jsonify({"projects": [p.as_dict() for p in projects.REGISTRY],
                    "display_tz": config.DISPLAY_TZ,
                    "poll": config.POLL})


@app.route("/api/p/<pid>/status")
def api_project_status(pid):
    project = projects.get(pid)
    recent = db.list_runs(pid, limit=15)
    for run in recent:
        run["progress"] = None
    return jsonify({
        "project": project.as_dict(),
        "status": project.status(),
        "runs": recent,
        "active": db.active_runs(pid),
    })


@app.route("/api/p/<pid>/resolve-day")
def api_resolve_day(pid):
    project = projects.get(pid)
    if not project.day_info:
        return jsonify({"error": "this project has no day resolver"}), 400
    return jsonify(project.day_info(request.args.get("date"),
                                    request.args.get("mode", "central")))


@app.route("/api/p/<pid>/run", methods=["POST"])
def api_project_run(pid):
    body = request.get_json(silent=True) or {}
    params = body.get("params") or {}
    options = body.get("options") or {}
    run_id = runs.start(pid, params, options, source="manual")
    return jsonify({"run_id": run_id})


@app.route("/api/runs")
def api_runs():
    pid = request.args.get("project")
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify({"runs": db.list_runs(pid, limit=limit)})


@app.route("/api/runs/<int:run_id>")
def api_run(run_id):
    run = db.get_run(run_id)
    if not run:
        return jsonify({"error": "no such run"}), 404
    return jsonify(runs.progress(run))


@app.route("/api/runs/<int:run_id>/cancel", methods=["POST"])
def api_run_cancel(run_id):
    runs.cancel(run_id)
    return jsonify({"cancelled": run_id})


@app.route("/api/p/<pid>/logs/<int:job_id>")
def api_logs(pid, job_id):
    task = request.args.get("task")
    stream = request.args.get("stream", "out")
    lines = min(int(request.args.get("lines", 200)), 2000)
    return jsonify(runs.tail_log(pid, job_id, task=task, lines=lines, stream=stream))


# --- Wazuh ------------------------------------------------------------------

_WAZUH_DEFAULT_PROFILES = ("nbdevlab", "remote")


@cached(10)
def _wazuh_profile_health():
    """Include the two designed profiles even before credentials are installed."""
    names = set(_WAZUH_DEFAULT_PROFILES)
    names.update(wazuhapi.profile_ids())
    return [wazuhapi.health(name) for name in sorted(names)]


@app.route("/api/wazuh/profiles")
def api_wazuh_profiles():
    # health() exposes the configured URL and username for diagnosis, but the
    # password never leaves wazuhapi's process-local profile object.
    return jsonify({"profiles": _wazuh_profile_health()})


@app.route("/api/wazuh/<profile>/indices")
def api_wazuh_indices(profile):
    return jsonify({"indices": wazuhapi.indices(
        profile, pattern=request.args.get("pattern"))})


@app.route("/api/wazuh/llm")
def api_wazuh_llm():
    import wazuhllm
    return jsonify(wazuhllm.health())


@app.route("/api/wazuh/llm/settings")
def api_wazuh_llm_settings():
    import wazuhllm
    return jsonify(wazuhllm.describe())


@app.route("/api/wazuh/llm/settings", methods=["POST"])
def api_wazuh_llm_settings_save():
    import wazuhllm
    body = request.get_json(silent=True) or {}
    try:
        saved = wazuhllm.save_settings(body)
    except wazuhllm.LLMError as exc:
        # a rejected value is the user's mistake, not a server fault
        return jsonify({"error": str(exc)}), 400
    return jsonify({"settings": saved})


@app.route("/api/wazuh/reports/<int:report_id>/triage")
def api_wazuh_triage(report_id):
    return jsonify(wazuh.read_triage(report_id) or {"state": "none"})


@app.route("/api/wazuh/reports/<int:report_id>/triage/cancel", methods=["POST"])
def api_wazuh_triage_cancel(report_id):
    return jsonify(wazuh.cancel_triage(report_id))


@app.route("/api/wazuh/reports/<int:report_id>/triage", methods=["POST"])
def api_wazuh_triage_start(report_id):
    body = request.get_json(silent=True) or {}
    return jsonify(wazuh.start_triage(report_id, force=bool(body.get("force"))))


# --- AI: cluster inference usage ---------------------------------------------
# Rollups over the request ledger the sampler fills, plus a live probe of each
# running server. Memoised for less than one poll interval so several open tabs
# do not re-run the same grouped SELECTs and re-probe the same servers.

@cached(3)
def _ai_overview(window):
    return llmusage.overview(window)


@cached(3)
def _ai_comfy():
    return comfy.summary()


@app.route("/api/ai/overview")
def api_ai_overview():
    data = dict(_ai_overview(request.args.get("window") or llmusage.DEFAULT_WINDOW))
    data["comfy"] = _ai_comfy()
    return jsonify(data)


@app.route("/api/ai/comfy")
def api_ai_comfy():
    return jsonify(comfy.summary())


@app.route("/api/ai/comfy/stop", methods=["POST"])
def api_ai_comfy_stop():
    """Cancel the on-demand ComfyUI worker job, freeing its card now."""
    result = comfy.stop()
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_comfy"]:
            del _memo[key]
    app.logger.info("comfyui worker stop: %s", result)
    return jsonify(result)


@app.route("/api/ai/requests")
def api_ai_requests():
    window = request.args.get("window") or llmusage.DEFAULT_WINDOW
    seconds = llmusage.window_seconds(window)
    since = None if seconds is None else time.time() - seconds
    rows = llmusage.requests(since, limit=request.args.get("limit", 200),
                             model=request.args.get("model") or None)
    return jsonify({"window": window, "requests": rows, "count": len(rows)})


@app.route("/api/ai/live")
def api_ai_live():
    """Progress bars: model loading, prompt evaluation, generation."""
    return jsonify(llmusage.live())


# Live model I/O: the opencode sandbox's llm-tap.py streams a copy of its router traffic to the head, where
# /data/llm/run/live-opencode.json is rewritten atomically. The router itself exposes no prompt or output text
# (/slots has counts only), so this is the only source of raw input/output.
LIVE_IO_PATH = "/data/llm/run/live-opencode.json"


def _live_io_doc():
    import json
    try:
        with open(LIVE_IO_PATH) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return {"exchanges": [], "missing": True}
    except ValueError:
        return {"exchanges": [], "error": "live file unreadable"}
    doc["age"] = max(0.0, time.time() - (doc.get("updated") or 0))
    return doc


@app.route("/api/ai/raw")
def api_ai_raw():
    """Streamed output of the latest exchanges; request bodies only on ?full=<sess>:<id>."""
    doc = _live_io_doc()
    full = request.args.get("full")
    if full:
        for e in doc.get("exchanges", []):
            if "%s:%s" % (e.get("sess"), e.get("id")) == full:
                return jsonify({"exchange": e})
        return jsonify({"error": "exchange no longer kept"}), 404
    for e in doc.get("exchanges", []):
        req = e.pop("request", None)
        if isinstance(req, dict):
            msgs = req.get("messages") or []
            e["request_summary"] = {
                "model": req.get("model"), "messages": len(msgs),
                "tools": len(req.get("tools") or []),
                "chars": sum(len(m.get("content") or "") if isinstance(m.get("content"), str)
                             else len(str(m.get("content") or "")) for m in msgs),
                "last_user": next((m.get("content") for m in reversed(msgs)
                                   if m.get("role") == "user" and isinstance(m.get("content"), str)), None),
                "reasoning_effort": req.get("reasoning_effort"),
                "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens"),
            }
            if e["request_summary"]["last_user"]:
                e["request_summary"]["last_user"] = e["request_summary"]["last_user"][-600:]
    return jsonify(doc)


@app.route("/api/ai/poll", methods=["POST"])
def api_ai_poll():
    """Sample now rather than waiting for the next tick."""
    return jsonify(llmusage.poll())


@app.route("/api/ai/load", methods=["POST"])
def api_ai_load():
    """Load a model on a router now, evicting whatever is resident."""
    body = request.get_json(silent=True) or {}
    if not body.get("job_id") or not body.get("model"):
        return jsonify({"error": "job_id and model are required"}), 400
    try:
        result = llmusage.load_model(body["job_id"], str(body["model"]))
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_overview"]:
            del _memo[key]
    app.logger.info("loading %s on job %s on %s (evicting %s)", result["model"],
                    result["job_id"], result["node"], result.get("evicted"))
    return jsonify(result)


@app.route("/api/ai/lock", methods=["POST"])
def api_ai_lock():
    """Lock the loaded model (slerm-orch): never unloaded or idled out, other
    models refused, its nodes kept on, until unlocked."""
    body = request.get_json(silent=True) or {}
    try:
        result = llmusage.lock_model(body.get("model"))
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    _forget_ai_overview()
    app.logger.info("model lock: %s", result.get("model"))
    return jsonify(result)


@app.route("/api/ai/unlock", methods=["POST"])
def api_ai_unlock():
    try:
        result = llmusage.unlock_model()
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    _forget_ai_overview()
    app.logger.info("model unlocked: %s", result.get("unlocked"))
    return jsonify(result)


def _forget_ai_overview():
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_overview"]:
            del _memo[key]


@app.route("/api/ai/bringup", methods=["POST"])
def api_ai_bringup():
    """Have the orchestrator bring up whatever MODEL needs (nodes, workers,
    a router on the right device set), then load it."""
    body = request.get_json(silent=True) or {}
    if not body.get("model"):
        return jsonify({"error": "model is required"}), 400
    try:
        result = llmusage.bring_up(str(body["model"]))
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_overview"]:
            del _memo[key]
    app.logger.info("bring-up requested for %s", body["model"])
    return jsonify(result)


@app.route("/api/ai/stage", methods=["POST"])
def api_ai_stage():
    """Stage or unstage MODEL: a local copy on every node it will load on."""
    body = request.get_json(silent=True) or {}
    if not body.get("model"):
        return jsonify({"error": "model is required"}), 400
    fn = llmusage.unstage if body.get("unstage") else llmusage.stage
    try:
        result = fn(str(body["model"]))
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_overview"]:
            del _memo[key]
    app.logger.info("%s %s", "unstaged" if body.get("unstage") else "staged", body["model"])
    return jsonify(result)


@app.route("/api/ai/stop-load", methods=["POST"])
def api_ai_stop_load():
    """Abandon a model that is still loading."""
    body = request.get_json(silent=True) or {}
    if not body.get("job_id") or not body.get("model"):
        return jsonify({"error": "job_id and model are required"}), 400
    try:
        result = llmusage.stop_loading(body["job_id"], str(body["model"]))
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_overview"]:
            del _memo[key]
    app.logger.info("stopped loading %s on job %s on %s", result["model"],
                    result["job_id"], result["node"])
    return jsonify(result)


@app.route("/api/ai/unload", methods=["POST"])
def api_ai_unload():
    """Evict the resident model from one router, freeing its GPUs."""
    body = request.get_json(silent=True) or {}
    if not body.get("job_id") or not body.get("model"):
        return jsonify({"error": "job_id and model are required"}), 400
    try:
        result = llmusage.unload(body["job_id"], str(body["model"]))
    except llmusage.UnloadError as exc:
        return jsonify({"error": str(exc)}), 409
    with _memo_lock:
        for key in [k for k in _memo if k[0] == "_ai_overview"]:
            del _memo[key]
    app.logger.info("unloaded %s from job %s on %s", result["model"],
                    result["job_id"], result["node"])
    return jsonify(result)


@app.route("/api/wazuh/fetches/<int:fetch_id>/cancel", methods=["POST"])
def api_wazuh_fetch_cancel(fetch_id):
    return jsonify({"fetch": wazuh.cancel_fetch(fetch_id)})


@app.route("/api/wazuh/<profile>/fetch", methods=["POST"])
def api_wazuh_fetch(profile):
    body = request.get_json(silent=True) or {}
    fetch_id = wazuh.start_fetch(
        profile, body.get("start"), body.get("end"),
        pattern=body.get("pattern"), filter_query=body.get("query"))
    return jsonify({"fetch_id": fetch_id,
                    "fetch": wazuh.get_fetch(fetch_id)}), 202


@app.route("/api/wazuh/import", methods=["POST"])
def api_wazuh_import():
    upload = request.files.get("file")
    if upload is None and request.files:
        upload = next(iter(request.files.values()))
    report_id = wazuh.start_import(upload)
    return jsonify({"report_id": report_id,
                    "report": wazuh.get_report(report_id)}), 202


def _wazuh_report_rows(profile=None, source=None, limit=100):
    rows = wazuh.list_reports(profile=profile, source=source, limit=limit)
    # The fetch table is intentionally separate (and contains no credentials).
    # Embed its live byte/alert counters so one report list drives the complete
    # snapshot lifecycle without a second browser poll.
    for row in rows:
        fetch = (wazuh.get_fetch(row["fetch_id"])
                 if row.get("fetch_id") else None)
        row["fetch"] = fetch
        row["bytes"] = fetch.get("bytes", 0) if fetch else None
        if fetch and row.get("state") == "fetching":
            row["alerts"] = fetch.get("alerts", row.get("alerts", 0))
    return rows


@app.route("/api/wazuh/reports")
def api_wazuh_reports():
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    except ValueError:
        raise wazuh.WazuhError("bad report limit")
    return jsonify({"reports": _wazuh_report_rows(
        profile=request.args.get("profile") or None,
        source=request.args.get("source") or None, limit=limit)})


@app.route("/api/wazuh/reports/<int:report_id>")
def api_wazuh_report(report_id):
    row = wazuh.get_report(report_id)
    if not row:
        return jsonify({"error": "no such Wazuh report"}), 404
    data = wazuh.read_report(report_id) if row.get("state") == "ready" else None
    return jsonify({"report": row, "data": data})


@app.route("/api/wazuh/reports/<int:report_id>/evidence")
def api_wazuh_evidence(report_id):
    return jsonify(wazuh.read_evidence(report_id,
                                       request.args.get("finding")))


# --- frames and GIF ---------------------------------------------------------

@app.route("/api/p/<pid>/prod/<prod>/frames")
def api_frames(pid, prod):
    product = projects.get(pid).product(prod)
    group = request.args.get("group")
    return jsonify({"frames": product.frames_in_group(group),
                    "groups": product.groups(),
                    "out_dir": product.out_dir})


@app.route("/api/p/<pid>/prod/<prod>/frame/<path:name>")
def api_frame(pid, prod, name):
    product = projects.get(pid).product(prod)
    resp = send_file(product.frame_path(name), mimetype="image/png",
                     conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


_assembling = {}


@app.route("/api/p/<pid>/prod/<prod>/assemble", methods=["POST"])
def api_assemble(pid, prod):
    project = projects.get(pid)
    product = project.product(prod)
    if not product.assemble:
        return jsonify({"error": "this product has no assembly step"}), 400
    body = request.get_json(silent=True) or {}
    group = body.get("group")
    token = "%s/%s/%s" % (pid, prod, group)
    if _assembling.get(token):
        return jsonify({"state": "running", "group": group})

    def work():
        _assembling[token] = True
        try:
            path = product.assemble(project, product, group)
            _assembling[token] = False
            app.logger.info("assembled %s", path)
        except Exception as exc:  # noqa: BLE001
            _assembling[token] = False
            app.logger.warning("assemble failed: %s", exc)
            _assembling[token + "!err"] = str(exc)

    _assembling.pop(token + "!err", None)
    threading.Thread(target=work, name="assemble-%s" % prod, daemon=True).start()
    return jsonify({"state": "started", "group": group})


@app.route("/api/p/<pid>/prod/<prod>/assemble/status")
def api_assemble_status(pid, prod):
    group = request.args.get("group")
    token = "%s/%s/%s" % (pid, prod, group)
    return jsonify({"running": bool(_assembling.get(token)),
                    "error": _assembling.get(token + "!err")})


@app.route("/api/p/<pid>/prod/<prod>/gif/status")
def api_gif_status(pid, prod):
    product = projects.get(pid).product(prod)
    return jsonify(gifbuild.describe(product, request.args.to_dict()))


@app.route("/api/p/<pid>/prod/<prod>/gif/build", methods=["POST"])
def api_gif_build(pid, prod):
    product = projects.get(pid).product(prod)
    body = request.get_json(silent=True) or {}
    key = gifbuild.build(product, body, force=bool(body.get("force")))
    return jsonify(dict(gifbuild.describe(product, body), key=key))


@app.route("/api/p/<pid>/prod/<prod>/gif")
def api_gif(pid, prod):
    product = projects.get(pid).product(prod)
    info = gifbuild.describe(product, request.args.to_dict())
    if not info["ready"]:
        return jsonify({"error": "not built yet", "info": info}), 404
    download = request.args.get("download") in ("1", "true", "yes")
    resp = send_file(gifbuild.cache_path(product, info["key"]),
                     mimetype="image/gif", conditional=True,
                     as_attachment=download,
                     download_name="%s-%s-%s.gif" % (pid, product.id, info["key"]))
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


# --- schedules --------------------------------------------------------------

@app.route("/api/schedules")
def api_schedules():
    pid = request.args.get("project")
    out = []
    for sched in db.list_schedules(pid):
        sched["description"] = scheduler.describe(sched["kind"], sched["expr"])
        out.append(sched)
    return jsonify({"schedules": out, "scheduler": scheduler.status()})


@app.route("/api/schedules", methods=["POST"])
def api_schedule_create():
    body = request.get_json(silent=True) or {}
    pid = body.get("project")
    projects.get(pid)                       # validates the project id
    kind = body.get("kind", "cron")
    expr = str(body.get("expr", "")).strip()
    scheduler.validate(kind, expr)
    sched_id = db.create_schedule(
        pid, body.get("label") or "", kind, expr,
        body.get("params") or {}, body.get("options") or {},
        enabled=bool(body.get("enabled", True)),
        next_fire=scheduler.next_fire(kind, expr))
    return jsonify({"schedule": db.get_schedule(sched_id)}), 201


@app.route("/api/schedules/<int:sched_id>", methods=["PATCH"])
def api_schedule_update(sched_id):
    sched = db.get_schedule(sched_id)
    if not sched:
        return jsonify({"error": "no such schedule"}), 404
    body = request.get_json(silent=True) or {}
    fields = {}
    kind = body.get("kind", sched["kind"])
    expr = str(body.get("expr", sched["expr"])).strip()
    if "kind" in body or "expr" in body:
        scheduler.validate(kind, expr)
        fields["kind"] = kind
        fields["expr"] = expr
        fields["next_fire"] = scheduler.next_fire(kind, expr)
    for key in ("label",):
        if key in body:
            fields[key] = body[key]
    for key in ("params", "options"):
        if key in body:
            fields[key] = body[key]
    if "enabled" in body:
        fields["enabled"] = 1 if body["enabled"] else 0
        if body["enabled"] and "next_fire" not in fields:
            fields["next_fire"] = scheduler.next_fire(kind, expr)
    db.update_schedule(sched_id, **fields)
    return jsonify({"schedule": db.get_schedule(sched_id)})


@app.route("/api/schedules/<int:sched_id>", methods=["DELETE"])
def api_schedule_delete(sched_id):
    db.delete_schedule(sched_id)
    return jsonify({"deleted": sched_id})


@app.route("/api/schedules/<int:sched_id>/run", methods=["POST"])
def api_schedule_run(sched_id):
    sched = db.get_schedule(sched_id)
    if not sched:
        return jsonify({"error": "no such schedule"}), 404
    run_id = scheduler.fire(sched, force=True)
    return jsonify({"run_id": run_id})


_instance_lock = None


def claim_single_instance():
    """Refuse to start when another panel already owns this data root.

    create_app() reconciles orphaned work on the way up, so a second start --
    including one that is only going to fail the port bind a moment later --
    used to reach the triage sidecars first. Taking the lock before anything is
    reconciled means the loser exits having touched nothing.
    """
    global _instance_lock
    path = os.path.join(config.WAZUH_DIR, ".panel.lock")
    fh = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o644), "r+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        holder = fh.read().strip()
        fh.close()
        sys.stderr.write(
            "slerm-panel already holds %s%s -- refusing to start a second "
            "instance (%s)\n"
            % (path, " (pid " + holder + ")" if holder else "", exc))
        raise SystemExit(1)
    fh.seek(0)
    fh.truncate()
    fh.write("%d\n" % os.getpid())
    fh.flush()
    _instance_lock = fh  # held open for the life of the process
    return fh


def create_app():
    claim_single_instance()
    db.init()
    try:
        stale = wazuh.reconcile_stale_fetches()
        if stale:
            app.logger.info("resolved %d orphaned wazuh fetch(es)", stale)
        stale_triage = wazuh.reconcile_stale_triage()
        if stale_triage:
            app.logger.info("resolved %d orphaned triage run(s)", stale_triage)
    except Exception as exc:  # noqa: BLE001 - never block startup on this
        app.logger.warning("wazuh fetch reconciliation failed: %s", exc)
    dropped = llmusage.prune()
    if dropped:
        app.logger.info("pruned %d llm row(s) past the retention window", dropped)
    llmusage.start()
    for project in projects.REGISTRY:
        try:
            project.ensure_dirs()
        except OSError:
            pass
    scheduler.start()
    return app


if __name__ == "__main__":
    from waitress import serve
    create_app()
    print("slerm-panel listening on %s:%d" % (config.BIND_HOST, config.BIND_PORT),
          flush=True)
    serve(app, host=config.BIND_HOST, port=config.BIND_PORT,
          threads=config.THREADS, ident="slerm-panel")
