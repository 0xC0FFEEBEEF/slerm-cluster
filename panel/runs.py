"""Turning a project + parameters into a Slurm array job, and following it."""
import datetime
import glob
import os
import re
import threading

import db
import projects
import slurmapi
from projects.base import ProjectError

_TOKEN_RE = re.compile(r"\{\{\s*date\s*:\s*([a-z0-9+\-]+)\s*\}\}", re.I)


def expand_tokens(params, now=None):
    """Expand {{date:yesterday}} / {{date:today}} / {{date:today-3}} to YYYY/MM/DD.

    Lets a recurring schedule follow the calendar instead of being pinned to
    one hard-coded day.
    """
    now = now or datetime.datetime.now()
    out = {}
    for key, value in params.items():
        if not isinstance(value, str):
            out[key] = value
            continue

        def repl(match):
            word = match.group(1).lower()
            if word == "today":
                delta = 0
            elif word == "yesterday":
                delta = 1
            else:
                m = re.match(r"^today([+-])(\d+)$", word)
                if not m:
                    return match.group(0)
                delta = int(m.group(2)) * (-1 if m.group(1) == "+" else 1)
            return (now - datetime.timedelta(days=delta)).strftime("%Y/%m/%d")

        out[key] = _TOKEN_RE.sub(repl, value)
    return out


def start(project_id, params, options=None, source="manual", schedule_id=None):
    """Create a run row and kick off the listing/submission in the background."""
    project = projects.get(project_id)
    params = expand_tokens(dict(params or {}))
    options = dict(options or {})
    # count against the product this run renders, not the project default
    frames_before = (len(project.product(options.get("product")).frames())
                     if project.products else None)
    run_id = db.create_run(project_id, params, options, source=source,
                           schedule_id=schedule_id, frames_before=frames_before)
    thread = threading.Thread(target=_worker, args=(run_id, project, params, options),
                             name="run-%d" % run_id, daemon=True)
    thread.start()
    return run_id


def _worker(run_id, project, params, options):
    try:
        product = (project.product(options.get("product"))
                   if project.products else None)
        target_label = product.label if product else project.label
        db.update_run(run_id, state="listing",
                      message="enumerating work for %s" % target_label)
        # a product whose work items are not the project's own (HRRR analysis
        # hours rather than radar scans) brings its own lister
        if product and product.make_list:
            list_path, count = product.make_list(project, product, params)
        else:
            if not project.make_list:
                raise ProjectError("project %r has no work lister" % project.id)
            list_path, count = project.make_list(project, params)
        db.update_run(run_id, list_path=list_path, item_count=count,
                      state="submitting", message="%d items listed" % count)

        args = project.sbatch_args(project, list_path, count, options)
        job_id, submit_line = slurmapi.submit(args, cwd=project.scripts_dir)
        db.update_run(run_id, job_id=job_id, submit_line=submit_line,
                      state="submitted", message="submitted as job %d" % job_id)
        if project.post_submit:
            project.post_submit(project, run_id, job_id, list_path, options)
    except (ProjectError, slurmapi.SlurmError) as exc:
        db.update_run(run_id, state="failed", message=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface anything else in the UI
        db.update_run(run_id, state="failed",
                      message="%s: %s" % (type(exc).__name__, exc))


def progress(run):
    """Live view of a run: task buckets plus how many frames it has produced."""
    project = projects.get(run["project"])
    info = {"run": run, "counts": None, "tasks": [], "frames_new": None,
            "frames_now": None}
    frames_now = (len(project.product(
        run.get("options", {}).get("product")).frames())
        if project.products else None)
    info["frames_now"] = frames_now
    if frames_now is not None and run.get("frames_before") is not None:
        info["frames_new"] = max(frames_now - run["frames_before"], 0)

    if not run.get("job_id"):
        return info

    data = slurmapi.array_progress(run["job_id"])
    info["counts"] = data["counts"]
    info["tasks"] = data["tasks"]

    # advance the stored state so history reads correctly without a poller
    counts = data["counts"]
    if run["state"] in ("submitted", "running"):
        summary = "%d/%d tasks done" % (counts["done"], counts["total"])
        if counts["active"] > 0:
            if counts["running"] > 0 and run["state"] != "running":
                db.update_run(run["id"], state="running", message=summary)
        elif counts["total"] > 0:
            if counts["failed"]:
                final = "failed"
            elif counts["cancelled"]:
                final = "cancelled"
            else:
                final = "completed"
            db.update_run(run["id"], state=final,
                          message="%d completed, %d failed, %d cancelled"
                                  % (counts["completed"], counts["failed"],
                                     counts["cancelled"]))
    return info


def cancel(run_id):
    run = db.get_run(run_id)
    if not run:
        raise ProjectError("no such run")
    if not run.get("job_id"):
        db.update_run(run_id, state="cancelled", message="cancelled before submission")
        return True
    slurmapi.cancel(run["job_id"])
    db.update_run(run_id, state="cancelled", message="cancelled by user")
    return True


def tail_log(project_id, job_id, task=None, lines=200, stream="out"):
    """Read the tail of an array task's stdout/stderr."""
    project = projects.get(project_id)
    suffix = "err" if stream == "err" else "out"
    if task is None:
        pattern = os.path.join(project.logs_dir, "%s_*.%s" % (job_id, suffix))
        matches = sorted(glob.glob(pattern))
        if not matches:
            pattern = os.path.join(project.logs_dir, "*%s*.%s" % (job_id, suffix))
            matches = sorted(glob.glob(pattern))
        if not matches:
            return {"path": None, "text": "", "missing": True}
        path = matches[-1]
    else:
        path = os.path.join(project.logs_dir, "%s_%s.%s" % (job_id, task, suffix))

    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(project.logs_dir) + os.sep):
        raise ProjectError("log outside the project log directory")
    if not os.path.isfile(real):
        return {"path": real, "text": "", "missing": True}
    try:
        with open(real, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            back = min(size, 256 * 1024)
            fh.seek(size - back)
            data = fh.read().decode("utf-8", "replace")
    except OSError as exc:
        return {"path": real, "text": "", "missing": True, "error": str(exc)}
    tail = data.splitlines()[-lines:]
    return {"path": real, "text": "\n".join(tail), "missing": False,
            "size": size}
