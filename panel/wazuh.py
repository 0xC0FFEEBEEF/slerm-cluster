"""Wazuh ingest orchestration: snapshot, normalise, shard, and report.

The indexer is fetched serially on the panel host because that is where the
credentials and the remote tunnel live.  Both indexer fetches and uploaded
exports end as one normalised JSONL snapshot; only then is work split into
byte ranges for the Slurm array.

Files used as completion signals are always written beside their destination
as ``.part`` files and installed with :func:`os.replace`.
"""
import importlib.util
import json
import os
import re
import shutil
import threading
import time
import uuid

import config
import db
import runs
import slurmapi
import wazuhapi
import wazuhllm
import wazuhweb
from projects.base import ProjectError


class WazuhError(ProjectError):
    """Bad ingest input or an unavailable/malformed local Wazuh artifact."""


_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_normaliser = None
_normaliser_path = None
_normaliser_lock = threading.Lock()


def _error_text(exc):
    text = str(exc).strip()
    if not text:
        text = type(exc).__name__
    return ("%s: %s" % (type(exc).__name__, text))[:1500]


def _id(value, label="id"):
    if isinstance(value, bool):
        raise WazuhError("bad %s" % label)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise WazuhError("bad %s" % label)
    if value <= 0:
        raise WazuhError("bad %s" % label)
    return value


def _profile_id(value):
    value = str(value or "")
    if not _PROFILE_ID.match(value) or value in (".", ".."):
        raise WazuhError("bad Wazuh profile id")
    return value


def _path(*parts):
    """Resolve a generated path and prove that it stays below WAZUH_DIR."""
    root = os.path.realpath(config.WAZUH_DIR)
    path = os.path.realpath(os.path.join(root, *[str(p) for p in parts]))
    if path != root and not path.startswith(root + os.sep):
        raise WazuhError("Wazuh path escapes %s" % root)
    return path


def _safe_existing_export(path):
    real = os.path.realpath(str(path))
    root = os.path.realpath(config.WAZUH_DIR)
    if not real.startswith(root + os.sep):
        raise WazuhError("export is outside %s" % root)
    if any(ch.isspace() for ch in real):
        # wazuh_array.sh intentionally uses a whitespace-delimited, three-field
        # list format. Generated paths never contain whitespace.
        raise WazuhError("export path contains whitespace")
    if not os.path.isfile(real):
        raise WazuhError("export does not exist")
    return real


def _unlink_part(path):
    if not path:
        return
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
    except OSError:
        pass


def _atomic_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    try:
        with open(part, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(part, path)
    except Exception:
        _unlink_part(part)
        raise


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, separators=(",", ":")))


def _report_path(report_id):
    return _path("reports", "%d.json" % _id(report_id, "report id"))


def _load_normaliser():
    """Load the deployed cluster normaliser from its configured script path."""
    global _normaliser, _normaliser_path
    path = os.path.realpath(os.path.join(config.WAZUH_SCRIPTS,
                                         "wazuh_normalise.py"))
    with _normaliser_lock:
        if _normaliser is not None and _normaliser_path == path:
            return _normaliser
        if not os.path.isfile(path):
            raise WazuhError("Wazuh normaliser is missing at %s" % path)
        spec = importlib.util.spec_from_file_location(
            "_slerm_panel_wazuh_normalise", path)
        if spec is None or spec.loader is None:
            raise WazuhError("cannot load Wazuh normaliser at %s" % path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise WazuhError("cannot load Wazuh normaliser: %s" % exc)
        _normaliser = module
        _normaliser_path = path
        return module


def plan_shards(export_path, report_id=None):
    """Atomically write the contiguous byte-range list for one snapshot.

    The normalised JSONL is deliberately *not* aligned to newlines here.
    ``wazuh_shard.py`` owns boundary handling and its exact-newline case is
    covered by the cluster self-test.

    Returns ``(list_path, shard_count)``.
    """
    export_path = _safe_existing_export(export_path)
    try:
        target = int(config.WAZUH_SHARD_BYTES)
        maximum = int(config.WAZUH_MAX_SHARDS)
    except (TypeError, ValueError):
        raise WazuhError("invalid Wazuh shard configuration")
    if target <= 0 or maximum <= 0:
        raise WazuhError("invalid Wazuh shard configuration")

    size = os.path.getsize(export_path)
    count = max(1, min(maximum, (size + target - 1) // target))
    if report_id is None:
        token = uuid.uuid4().hex
    else:
        token = str(_id(report_id, "report id"))
    list_path = _path("lists", token + ".txt")

    lines = []
    for index in range(count):
        start = (size * index) // count
        end = (size * (index + 1)) // count
        lines.append("%s %d %d\n" % (export_path, start, end))
    _atomic_text(list_path, "".join(lines))
    return list_path, count


def _prepare_analysis(report_id, export_path, meta):
    """Plan shards and hand one prepared report to ``runs.start``."""
    report_id = _id(report_id, "report id")
    list_path, count = plan_shards(export_path, report_id=report_id)
    shard_dir = _path("shards", str(report_id))
    os.makedirs(shard_dir, exist_ok=True)

    report = db.get_wazuh_report(report_id)
    if not report:
        raise WazuhError("no such Wazuh report")
    full_meta = {
        "source": report.get("source"),
        "profile": report.get("profile"),
        "fetch_id": report.get("fetch_id"),
        "filename": report.get("filename"),
        "truncated": bool(report.get("truncated")),
        "expected_shards": count,
    }
    full_meta.update(meta or {})
    # Reducer consumes this only after the numbered sidecars are merged. Its
    # loader deliberately excludes meta.json.
    _atomic_json(os.path.join(shard_dir, "meta.json"), full_meta)

    db.update_wazuh_report(
        report_id, state="queued",
        message="%d shard%s planned; waiting for Slurm" %
                (count, "" if count == 1 else "s"))
    try:
        run_id = runs.start("wazuh", params={"report": report_id}, options={})
    except Exception as exc:
        db.update_wazuh_report(report_id, state="failed",
                               message=_error_text(exc))
        raise
    # Do not write state here: the very small array can reach post_submit and
    # advance the report to "submitted" before runs.start returns.
    db.update_wazuh_report(report_id, run_id=run_id)
    return {"report_id": report_id, "run_id": run_id,
            "list_path": list_path, "shards": count}


class FetchCancelled(Exception):
    """Raised inside the fetch callback to unwind a fetch the user stopped."""


# Cancellation is in-process because the worker thread is: a panel restart
# ends every fetch anyway. The DB state is what the UI reads, so it is set
# first and the flag is only the wake-up.
_cancelled = set()
_active = set()          # fetch ids with a live worker thread in THIS process
_cancel_lock = threading.Lock()


def cancel_fetch(fetch_id):
    row = db.get_wazuh_fetch(fetch_id)
    if not row:
        raise WazuhIngestError("no such fetch %s" % fetch_id)
    if row.get("state") in ("done", "failed", "cancelled"):
        return row
    with _cancel_lock:
        _cancelled.add(int(fetch_id))
        live = int(fetch_id) in _active
    if not live:
        # No worker is running to notice the flag -- the thread died with a
        # panel restart, or the fetch had already stopped. Without this the row
        # sits at "cancelling" forever waiting for something that cannot come.
        _finalise_cancelled(fetch_id, "cancelled (no active worker)")
    else:
        db.update_wazuh_fetch(fetch_id, state="cancelling",
                              message="cancelling on request")
    return db.get_wazuh_fetch(fetch_id)


def _finalise_cancelled(fetch_id, message):
    db.update_wazuh_fetch(fetch_id, state="cancelled", message=message)
    for report in db.list_wazuh_reports(fetch_id=fetch_id, limit=50):
        if report.get("state") not in ("done", "failed", "cancelled"):
            db.update_wazuh_report(report["id"], state="cancelled",
                                   message=message)


def reconcile_stale_triage():
    """Resolve triage sidecars whose worker died with a previous process.

    Same orphan class as fetches: the thread does not survive a restart, so a
    sidecar left at "running" would keep the UI polling for a run that no
    longer exists.

    Only a sidecar whose OWNING PROCESS IS GONE is an orphan. A run still being
    written by a live panel is left alone -- claiming it here is what turned a
    working triage into "interrupted" mid-run.
    """
    import glob
    resolved = 0
    for path in glob.glob(_path("reports", "*.triage.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if data.get("state") != "running":
            continue
        if _triage_owner_alive(data.get("owner")):
            continue
        data["state"] = "interrupted"
        data["message"] = "interrupted: the panel restarted mid-triage"
        tmp = path + ".part"
        try:
            with open(tmp, "w") as fh:
                json.dump(data, fh, separators=(",", ":"))
            os.replace(tmp, path)
            resolved += 1
        except OSError:
            pass
    return resolved


def reconcile_stale_fetches():
    """Resolve fetches whose worker died with a previous panel process.

    Worker threads do not survive a restart, so anything still marked live at
    startup is orphaned by definition. Leaving those rows alone is what made
    two cancelled remote fetches sit at "cancelling" indefinitely.
    """
    resolved = 0
    for row in db.list_wazuh_fetches(limit=500):
        state = row.get("state")
        if state == "cancelling":
            _finalise_cancelled(row["id"], "cancelled (panel restarted)")
            resolved += 1
        elif state in ("queued", "fetching"):
            db.update_wazuh_fetch(
                row["id"], state="failed",
                message="interrupted: the panel restarted mid-fetch")
            for report in db.list_wazuh_reports(fetch_id=row["id"], limit=50):
                if report.get("state") not in ("done", "failed", "cancelled"):
                    db.update_wazuh_report(
                        report["id"], state="failed",
                        message="interrupted: the panel restarted mid-fetch")
            resolved += 1
    return resolved


def _is_cancelled(fetch_id):
    with _cancel_lock:
        return int(fetch_id) in _cancelled


def _clear_cancel(fetch_id):
    with _cancel_lock:
        _cancelled.discard(int(fetch_id))


def start_fetch(profile, start, end, pattern=None, filter_query=None):
    """Create fetch/report rows and fetch an indexer window in a daemon thread.

    Returns the integer fetch id immediately. The linked report is created
    before the thread starts, so it is discoverable through ``list_reports``
    throughout the fetch.
    """
    profile = _profile_id(profile)
    prof = wazuhapi.profile(profile)  # validates it before creating DB rows
    start = str(start or "").strip()
    end = str(end or "").strip()
    if not start or not end:
        raise WazuhError("start and end are required")
    if len(start) > 128 or len(end) > 128:
        raise WazuhError("start or end is too long")
    pattern = str(pattern or prof.get("index") or
                  config.WAZUH_INDEX_PATTERN).strip()
    if not pattern:
        raise WazuhError("index pattern is required")
    if len(pattern) > 512:
        raise WazuhError("index pattern is too long")
    query = str(filter_query or "").strip() or None
    if query and len(query) > 8192:
        raise WazuhError("query is too long")

    fetch_id = db.create_wazuh_fetch(
        profile, index_pattern=pattern, query=query, t_start=start, t_end=end,
        state="fetching", message="waiting to fetch alerts")
    report_id = None
    try:
        report_id = db.create_wazuh_report(
            "fetch", profile=profile, fetch_id=fetch_id, state="fetching",
            message="fetching alerts from %s" % profile)
        db.update_wazuh_report(report_id, report_path=_report_path(report_id))
    except Exception as exc:
        db.update_wazuh_fetch(fetch_id, state="failed",
                              message=_error_text(exc))
        raise

    thread = threading.Thread(
        target=_fetch_worker,
        args=(fetch_id, report_id, profile, start, end, pattern, query),
        name="wazuh-fetch-%d" % fetch_id, daemon=True)
    thread.start()
    return fetch_id


def _fetch_worker(fetch_id, report_id, profile, start, end, pattern, query):
    with _cancel_lock:
        _active.add(int(fetch_id))
    try:
        _run_fetch(fetch_id, report_id, profile, start, end, pattern, query)
    finally:
        with _cancel_lock:
            _active.discard(int(fetch_id))
        _clear_cancel(fetch_id)


def _run_fetch(fetch_id, report_id, profile, start, end, pattern, query):
    part = None
    snapshot_ready = False
    try:
        snapshot_dir = _path("snapshots", profile, str(fetch_id))
        os.makedirs(snapshot_dir, exist_ok=True)
        export_path = os.path.join(snapshot_dir, "export.jsonl")
        part = export_path + ".part"

        slices = wazuhapi.slice_window(start, end, config.WAZUH_FETCH_SLICES)
        # Sequential paging is a cursor and cannot be parallelised, but disjoint
        # TIME RANGES can be. Counter merging downstream is order-independent,
        # so the slice files are simply concatenated. Each slice owns its own
        # file, so a failure or cancellation never leaves two writers
        # interleaved in one snapshot.
        counters = {"alerts": 0, "bytes": 0}
        errors = []
        progress_lock = threading.Lock()
        parts = []

        def pull(index, window):
            begin, finish, exclusive = window
            slice_path = "%s.slice%d" % (part, index)
            parts.append((index, slice_path))
            try:
                with open(slice_path, "wb") as sfh:
                    def on_batch(alerts, offset, reported, reported_exact=True):
                        if _is_cancelled(fetch_id):
                            raise FetchCancelled()
                        for alert in alerts:
                            sfh.write(json.dumps(
                                alert, separators=(",", ":")).encode("utf-8") + b"\n")
                        sfh.flush()
                        with progress_lock:
                            counters["alerts"] += len(alerts)
                            counters["bytes"] += sfh.tell() - counters.get(
                                "seen%d" % index, 0)
                            counters["seen%d" % index] = sfh.tell()
                            alerts_now = counters["alerts"]
                        message = "{:,} alerts fetched across {} slices".format(
                            alerts_now, len(slices))
                        db.update_wazuh_fetch(fetch_id, alerts=alerts_now,
                                              bytes=counters["bytes"],
                                              message=message)
                        db.update_wazuh_report(report_id, alerts=alerts_now,
                                               message=message)

                    wazuhapi.fetch(profile, begin, finish, pattern=pattern,
                                   filter_query=query, on_batch=on_batch,
                                   end_exclusive=exclusive,
                                   should_stop=lambda: _is_cancelled(fetch_id))
                    sfh.flush()
                    os.fsync(sfh.fileno())
            except FetchCancelled:
                errors.append(FetchCancelled())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = []
        for index, window in enumerate(slices):
            thread = threading.Thread(target=pull, args=(index, window),
                                      name="wazuh-fetch-%s-%d" % (fetch_id, index),
                                      daemon=True)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        if any(isinstance(e, FetchCancelled) for e in errors):
            for _i, path in parts:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            raise FetchCancelled()
        if errors:
            for _i, path in parts:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            raise errors[0]

        with open(part, "wb") as fh:
            for _i, path in sorted(parts):
                with open(path, "rb") as sfh:
                    shutil.copyfileobj(sfh, fh, 1024 * 1024)
                os.unlink(path)
            fh.flush()
            os.fsync(fh.fileno())
        result = {"fetched": counters["alerts"], "reported": None,
                  "capped": False}

        os.replace(part, export_path)
        part = None
        snapshot_ready = True
        size = os.path.getsize(export_path)
        capped = bool(result.get("capped"))
        if capped:
            message = ("snapshot ready: %d of %s alerts (configured cap reached)"
                       % (result["fetched"], result.get("reported", "?")))
        else:
            message = "snapshot ready: %d alerts" % result["fetched"]
        db.update_wazuh_fetch(
            fetch_id, state="ready", message=message,
            alerts=result["fetched"], bytes=size, path=export_path)
        db.update_wazuh_report(
            report_id, alerts=result["fetched"], truncated=1 if capped else 0,
            message=message)
        _prepare_analysis(report_id, export_path, {
            "source": "fetch", "profile": profile, "fetch_id": fetch_id,
            "filename": None, "truncated": capped,
            "index": result.get("index"), "reported": result.get("reported"),
            "fetched": result.get("fetched"), "t_start": start,
            "t_end": end, "query": query,
        })
    except FetchCancelled:
        # must precede the BaseException handler below, which would otherwise
        # record a deliberate stop as a fetch failure
        db.update_wazuh_fetch(fetch_id, state="cancelled",
                              message="cancelled before completion")
        db.update_wazuh_report(report_id, state="cancelled",
                               message="fetch cancelled before completion")
        if part and os.path.exists(part):
            try:
                os.unlink(part)       # a partial snapshot is not a snapshot
            except OSError:
                pass
        return
    except BaseException as exc:  # normaliser-style SystemExit is made visible
        if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
            raise
        _unlink_part(part)
        message = _error_text(exc)
        if not snapshot_ready:
            db.update_wazuh_fetch(fetch_id, state="failed", message=message)
        db.update_wazuh_report(report_id, state="failed", message=message)


def _display_filename(file_storage):
    name = str(getattr(file_storage, "filename", "") or "upload")
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = name.replace("\x00", "").strip() or "upload"
    return name[:255]


def _save_upload(file_storage, report_id):
    """Persist a request-owned upload stream before the request can close it."""
    stream = getattr(file_storage, "stream", file_storage)
    if not hasattr(stream, "read"):
        raise WazuhError("no upload supplied")
    try:
        limit = int(config.WAZUH_UPLOAD_MAX_BYTES)
    except (TypeError, ValueError):
        raise WazuhError("invalid Wazuh upload limit")
    if limit <= 0:
        raise WazuhError("uploads are disabled")

    upload_dir = _path("uploads", str(_id(report_id, "report id")))
    os.makedirs(upload_dir, exist_ok=True)
    part = os.path.join(upload_dir, "raw.part")
    total = 0
    magic = b""
    try:
        with open(part, "wb") as out:
            while True:
                chunk = stream.read(min(1024 * 1024, limit - total + 1))
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if not magic:
                    magic = chunk[:2]
                total += len(chunk)
                if total > limit:
                    raise WazuhError(
                        "upload exceeds the %d-byte limit" % limit)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if total == 0:
            raise WazuhError("the uploaded file is empty")
        # The deployed normaliser detects gzip from the filename. Use the file
        # signature rather than trusting the browser-provided name.
        raw_path = os.path.join(upload_dir,
                                "raw.gz" if magic == b"\x1f\x8b" else "raw")
        os.replace(part, raw_path)
        return raw_path, total
    except Exception:
        _unlink_part(part)
        raise


def start_import(file_storage):
    """Persist an upload and normalise/analyse it in a daemon thread.

    The request-owned stream is saved synchronously; everything CPU-heavy runs
    after the integer report id has been returned.
    """
    if file_storage is None:
        raise WazuhError("no upload supplied")
    filename = _display_filename(file_storage)
    report_id = db.create_wazuh_report(
        "import", filename=filename, state="uploading",
        message="saving %s" % filename)
    db.update_wazuh_report(report_id, report_path=_report_path(report_id))
    try:
        raw_path, size = _save_upload(file_storage, report_id)
    except Exception as exc:
        db.update_wazuh_report(report_id, state="failed",
                               message=_error_text(exc))
        raise
    db.update_wazuh_report(
        report_id, state="normalising",
        message="saved %d bytes; normalising %s" % (size, filename))
    thread = threading.Thread(
        target=_import_worker, args=(report_id, raw_path, filename),
        name="wazuh-import-%d" % report_id, daemon=True)
    thread.start()
    return report_id


def _import_worker(report_id, raw_path, filename):
    part = None
    try:
        snapshot_dir = _path("snapshots", "import", str(report_id))
        os.makedirs(snapshot_dir, exist_ok=True)
        export_path = os.path.join(snapshot_dir, "export.jsonl")
        part = export_path + ".part"
        normaliser = _load_normaliser()
        try:
            info = normaliser.convert(raw_path, part)
        except SystemExit as exc:
            raise WazuhError(str(exc) or "normalisation failed")
        if not isinstance(info, dict):
            raise WazuhError("normaliser returned no result")
        os.replace(part, export_path)
        part = None
        rows = int(info.get("rows") or 0)
        truncated = bool(info.get("truncated"))
        if truncated:
            message = ("normalised exactly %d dashboard rows; the export was "
                       "probably truncated" % rows)
        else:
            message = "normalised %d alerts from %s" % (rows, filename)
        db.update_wazuh_report(
            report_id, alerts=rows, truncated=1 if truncated else 0,
            message=message)
        _prepare_analysis(report_id, export_path, {
            "source": "import", "profile": None, "fetch_id": None,
            "filename": filename, "truncated": truncated,
            "format": info.get("format"), "normalised_rows": rows,
            "assumed_zone": (config.DISPLAY_TZ
                             if info.get("format") == "csv" else None),
        })
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
            raise
        _unlink_part(part)
        db.update_wazuh_report(report_id, state="failed",
                               message=_error_text(exc))


def get_fetch(fetch_id):
    return db.get_wazuh_fetch(_id(fetch_id, "fetch id"))


def list_fetches(profile=None, state=None, limit=100):
    if profile is not None:
        profile = _profile_id(profile)
    return db.list_wazuh_fetches(profile=profile, state=state, limit=limit)


def _row_report_path(row):
    expected = _report_path(row["id"])
    configured = row.get("report_path")
    if configured and os.path.realpath(configured) != os.path.realpath(expected):
        raise WazuhError("report path does not match report id")
    return expected


def _read_report_file(row):
    path = _row_report_path(row)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise WazuhError("cannot read Wazuh report %d: %s" % (row["id"], exc))
    if not isinstance(data, dict):
        raise WazuhError("Wazuh report %d is not a JSON object" % row["id"])
    return data


_LIVE_RUN_STATES = ("listing", "submitting", "submitted", "running")
_REDUCER_REPORT_GRACE = 60


def _settle_linked_run(row, state, message):
    """Keep the generic run history consistent with the reducer outcome."""
    run_id = row.get("run_id")
    if not run_id:
        return
    run = db.get_run(run_id)
    if not run:
        return
    current = run.get("state")
    if state == "completed":
        # A valid atomic report is the final source of truth. It may appear
        # after an accounting/metadata visibility race marked the run failed.
        if current not in _LIVE_RUN_STATES + ("failed",):
            return
    elif state == "failed":
        # runs.progress() can mark the shard array completed before its
        # dependent reducer finishes. Reducer failure must override that state.
        if current in ("failed", "cancelled"):
            return
    else:
        return
    db.update_run(run_id, state=state, message=str(message)[:1500])


def _fail_report(row, message):
    message = (str(message).strip() or "Wazuh analysis failed")[:1500]
    if row.get("state") != "failed" or row.get("message") != message:
        db.update_wazuh_report(row["id"], state="failed", message=message)
        row = db.get_wazuh_report(row["id"])
    _settle_linked_run(row, "failed", message)
    return row


def _failure_marker(row):
    path = _row_report_path(row) + ".failed"
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            message = fh.read(1501).strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return "cannot read reducer failure marker: %s" % exc
    return message[:1500] or "Wazuh reducer failed"


def _terminal_reducer_failure(row):
    job_id = row.get("reduce_job_id")
    if not job_id:
        return None
    try:
        status = slurmapi.job_status(job_id)
    except slurmapi.SlurmError:
        return None
    if not status or not status.get("terminal"):
        return None
    state = status.get("state") or "UNKNOWN"
    if status.get("success"):
        ended = status.get("end")
        # Accounting may become visible slightly before an NFS directory
        # lookup sees the reducer's atomic rename. Give the report a full
        # minute to appear before calling a successful job inconsistent.
        if not ended or time.time() - float(ended) < _REDUCER_REPORT_GRACE:
            return None
        return ("reduce job %d completed without publishing a report"
                % job_id)
    detail = status.get("reason")
    if status.get("exit_code") not in (None, 0):
        detail = ((detail + "; ") if detail else "") + \
                 "exit %s" % status["exit_code"]
    return "reduce job %d ended %s%s" % (
        job_id, str(state).lower(), ": " + detail if detail else "")


def _reconcile_report(row):
    if not row:
        return None
    data = None
    path = _row_report_path(row)
    if os.path.isfile(path):
        # Reducer reports can be large. The atomic first appearance is the
        # transition signal; once reconciled, list polling must stay O(rows)
        # rather than reparsing every completed report every two seconds.
        if row.get("state") == "ready":
            _settle_linked_run(
                row, "completed", "Wazuh report %d is ready" % row["id"])
            return row
        try:
            data = _read_report_file(row)
        except WazuhError as exc:
            return _fail_report(row, str(exc))
        fields = {
            "state": "ready",
            "message": "%d findings (%d actionable)" %
                       (int(data.get("finding_count") or 0),
                        int(data.get("actionable_count") or 0)),
            "alerts": int(data.get("alerts") or 0),
            "findings": int(data.get("finding_count") or 0),
            "actionable": int(data.get("actionable_count") or 0),
        }
        if any(row.get(key) != value for key, value in fields.items()):
            db.update_wazuh_report(row["id"], **fields)
            row = db.get_wazuh_report(row["id"])
        _settle_linked_run(
            row, "completed", "Wazuh report %d is ready" % row["id"])
        # Kick triage off as soon as the report exists so the AI fields fill in
        # while it is being read, rather than waiting for someone to press a
        # button. Cheap and idempotent: start_triage returns early if a sidecar
        # already exists or a run is in flight.
        if wazuhllm.settings().get("auto"):
            try:
                start_triage(row["id"])
            except Exception as exc:  # noqa: BLE001 - never block a report
                pass
        return row

    # A reducer that starts and exits normally publishes this marker
    # atomically. A terminal Slurm state covers cancellation, timeout, and node
    # failure where the shell never gets a chance to write one.
    if row.get("state") == "failed":
        _settle_linked_run(row, "failed", row.get("message") or "failed")
        return row
    failure = _failure_marker(row)
    if failure:
        return _fail_report(row, failure)
    failure = _terminal_reducer_failure(row)
    if failure:
        return _fail_report(row, failure)

    # A run that failed while listing/submitting never installed the dependent
    # reduce. Once post_submit succeeds it advances the report beyond queued,
    # so an array with failed tasks is not prematurely marked failed here.
    if row.get("state") == "queued" and row.get("run_id"):
        run = db.get_run(row["run_id"])
        if run and run.get("state") in ("failed", "cancelled"):
            return _fail_report(row, run.get("message") or run["state"])
    return row


def get_report(report_id):
    row = db.get_wazuh_report(_id(report_id, "report id"))
    return _reconcile_report(row) if row else None


def list_reports(profile=None, fetch_id=None, state=None, source=None, limit=100):
    if profile is not None:
        profile = _profile_id(profile)
    if fetch_id is not None:
        fetch_id = _id(fetch_id, "fetch id")
    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        raise WazuhError("bad report limit")
    if source is not None:
        source = str(source).strip()
        if not source or len(source) > 64:
            raise WazuhError("bad report source")
    rows = db.list_wazuh_reports(profile=profile, fetch_id=fetch_id,
                                 state=None, source=source, limit=limit)
    rows = [_reconcile_report(row) for row in rows]
    if state is not None:
        rows = [row for row in rows if row.get("state") == state]
    return rows


def read_report(report_id):
    """Return one completed reducer document with persisted ingest metadata."""
    row = get_report(report_id)
    if not row:
        raise WazuhError("no such Wazuh report")
    data = _read_report_file(row)
    if data is None:
        raise WazuhError("Wazuh report %d is %s" %
                         (row["id"], row.get("state") or "not ready"))
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    meta.setdefault("source", row.get("source"))
    meta.setdefault("profile", row.get("profile"))
    meta.setdefault("fetch_id", row.get("fetch_id"))
    meta.setdefault("filename", row.get("filename"))
    meta.setdefault("truncated", bool(row.get("truncated")))
    if meta.get("format") == "csv" and not meta.get("assumed_zone"):
        meta["assumed_zone"] = config.DISPLAY_TZ
    return data


def read_evidence(report_id, finding):
    """Parse the bounded raw evidence samples attached to one finding."""
    try:
        finding = int(finding)
    except (TypeError, ValueError):
        raise WazuhError("finding must be an integer")
    report = read_report(report_id)
    findings = report.get("findings") or []
    if finding < 0 or finding >= len(findings):
        raise WazuhError("no such finding")
    selected = findings[finding]
    parsed = []
    raw = []
    for sample in selected.get("evidence") or []:
        if isinstance(sample, (dict, list)):
            parsed.append(sample)
            continue
        if not isinstance(sample, str):
            raw.append(str(sample))
            continue
        try:
            parsed.append(json.loads(sample))
        except ValueError:
            raw.append(sample)
    return {
        "report_id": _id(report_id, "report id"),
        "finding": finding,
        "detector": selected.get("detector"),
        "title": selected.get("title"),
        "evidence": parsed,
        "raw": raw,
    }


# --- optional LLM triage -----------------------------------------------------

_triage_running = set()
_triage_cancelled = set()


def cancel_triage(report_id):
    """Stop a triage run between findings.

    The in-flight model call is allowed to finish rather than being torn out
    from under urllib -- at ~90s that is the longest you wait, and it keeps
    the sidecar consistent.
    """
    with _cancel_lock:
        live = report_id in _triage_running
        if live:
            _triage_cancelled.add(report_id)
    data = read_triage(report_id) or {}
    if not live:
        # nothing running: resolve now rather than leaving it "running" forever
        if data.get("state") == "running":
            data["state"] = "cancelled"
            data["message"] = "cancelled (no active triage worker)"
            _write_triage(report_id, data)
        return {"state": data.get("state", "none")}
    data["message"] = "stopping after the current finding"
    _write_triage(report_id, data)
    return {"state": "cancelling"}


def _triage_context(report, ranked, finding):
    """History for one finding: estate scale, and the host's own record.

    Everything here comes from the report the cluster already produced, so it
    costs nothing extra and stays reproducible.
    """
    total = report.get("alerts") or 0
    ctx = {"estate": {
        "alerts_in_window": total,
        "agents_reporting": report.get("agent_count"),
        "distinct_rules": report.get("rule_count"),
    }}
    agent = finding.get("agent")
    if not agent:
        return ctx
    tops = report.get("top_agents") or []
    rec = None
    for i, a in enumerate(tops):
        if str(a.get("id")) == str(agent) or a.get("name") == agent:
            rec = (i, a)
            break
    history = {}
    if rec:
        rank, a = rec
        count = a.get("count") or 0
        history.update({
            "agent": a.get("name") or agent,
            "alerts_in_window": count,
            "share_of_estate_pct": round(100.0 * count / total, 2) if total else None,
            "rank_by_volume": rank + 1,
            "of_top_agents_listed": len(tops),
        })
    siblings = [f for f in ranked
                if f is not finding and f.get("agent") == agent]
    if siblings:
        history["other_findings_on_this_host"] = [
            {"detector": f.get("detector"), "title": f.get("title"),
             "score": f.get("score")} for f in siblings[:6]]
        history["other_finding_count"] = len(siblings)
    if history:
        ctx["host_history"] = history
    return ctx


def triage_path(report_id):
    return _path("reports", "%s.triage.json" % report_id)


def read_triage(report_id):
    try:
        with open(triage_path(report_id)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def start_triage(report_id, force=False):
    """Annotate the top findings in the background. Never blocks a report."""
    if not wazuhllm.configured():
        raise WazuhIngestError("LLM triage is not configured")
    report = read_report(report_id)
    if not report:
        raise WazuhIngestError("no such report %s" % report_id)
    with _cancel_lock:
        if report_id in _triage_running:
            return {"state": "running"}
        _triage_running.add(report_id)
    if not force and read_triage(report_id):
        with _cancel_lock:
            _triage_running.discard(report_id)
        return {"state": "done", "cached": True}
    threading.Thread(target=_triage_worker, args=(report_id, report),
                     name="wazuh-triage-%s" % report_id, daemon=True).start()
    return {"state": "started"}


# Errors that describe the ENDPOINT rather than the finding in hand: a model
# that will not load will not load for the next finding either. Two in a row
# ends the run instead of spending every finding on the same wall.
_FATAL_LLM = ("failed to load", "unreachable at", "no llm endpoint available",
              "http 401", "http 403", "http 404", "http 500", "http 502")
_TRIAGE_FATAL_STREAK = 2


def _fatal_llm(text):
    lowered = str(text).lower()
    return any(marker in lowered for marker in _FATAL_LLM)


def _last_triage_error(out):
    for item in reversed(list(out["items"].values())):
        if "error" in item:
            return item["error"]
    return "no error recorded"


def _triage_worker(report_id, report):
    findings = report.get("findings") or []
    # index into the ORIGINAL list so annotations can never reorder anything
    targets = [(i, f) for i, f in enumerate(findings) if not f.get("routine")]
    ranked = [f for _i, f in targets]
    _cfg = wazuhllm.settings()
    if not _cfg.get("all_findings"):
        targets = targets[:int(_cfg["top_n"])]
    out = {"report": report_id, "model": wazuhllm.settings()["model"] or None,
           "state": "running", "items": {}, "errors": 0,
           # the indices being triaged, so the UI can mark the ones still
           # pending rather than leaving them looking un-analysed
           "targets": [i for i, _f in targets],
           "total": len(targets), "done": 0}
    # Publish the running state BEFORE the first call. Each finding takes ~90s,
    # so without this the UI keeps showing the previous run's sidecar for a
    # minute and a half and looks stuck.
    _write_triage(report_id, out)
    fatal_streak = 0
    try:
        for index, finding in targets:
            with _cancel_lock:
                stop = report_id in _triage_cancelled
            if stop:
                out["state"] = "cancelled"
                out["message"] = "stopped after %d of %d findings" % (
                    out["done"], out["total"])
                out["current"] = None
                break
            # publish WHAT is being worked on before the call, so the UI can
            # name it for the ~90s the model is thinking
            out["current"] = {"index": index,
                              "title": finding.get("title"),
                              "detector": finding.get("detector")}
            _write_triage(report_id, out)
            try:
                enriched = dict(finding)
                reference = wazuhweb.reference_for(finding)
                if reference:
                    enriched["reference_material"] = reference
                out["items"][str(index)] = wazuhllm.triage_one(
                    enriched,
                    context=_triage_context(report, ranked, finding))
                if reference:
                    out["items"][str(index)]["references"] = reference
                fatal_streak = 0
            except wazuhllm.LLMError as exc:
                text = str(exc)
                out["errors"] += 1
                out["items"][str(index)] = {"error": text}
                fatal_streak = fatal_streak + 1 if _fatal_llm(text) else 0
                if fatal_streak >= _TRIAGE_FATAL_STREAK:
                    out["done"] += 1
                    out["current"] = None
                    out["state"] = "failed"
                    out["message"] = (
                        "stopped after %d of %d findings -- the endpoint is "
                        "failing every request: %s"
                        % (out["done"], out["total"], text))
                    _write_triage(report_id, out)
                    break
            out["done"] += 1
            out["current"] = None
            _write_triage(report_id, out)
        else:
            # A run in which nothing was triaged is not a completed run. This
            # is what let a dead endpoint look like a finished triage that
            # simply had no verdicts.
            if out["total"] and out["errors"] >= out["total"]:
                out["state"] = "failed"
                out["message"] = ("no finding could be triaged: %s"
                                  % _last_triage_error(out))
            else:
                out["state"] = "done"
    except Exception as exc:  # noqa: BLE001
        out["state"] = "failed"
        out["message"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        out["current"] = None
        _write_triage(report_id, out)
        with _cancel_lock:
            _triage_running.discard(report_id)
            _triage_cancelled.discard(report_id)


# A live sidecar records the process writing it, so reconciliation can tell
# "the worker died with a previous panel" from "another process started while
# this run is still going". Without that, a second `python app.py` -- even one
# about to die on the port bind -- marked a healthy in-flight run interrupted.
def _boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _proc_start(pid):
    """Process start time in clock ticks; makes a recycled PID detectable."""
    try:
        with open("/proc/%d/stat" % pid, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    # comm may contain spaces and parentheses, so fields are read after the
    # LAST ')': that leaves state (field 3) at index 0, starttime (22) at 19.
    tail = data[data.rfind(b")") + 2:].split()
    try:
        return int(tail[19])
    except (IndexError, ValueError):
        return None


def _triage_owner():
    pid = os.getpid()
    return {"pid": pid, "boot": _boot_id(), "start": _proc_start(pid)}


def _triage_owner_alive(owner):
    """True while the process that wrote this sidecar is still running."""
    if not isinstance(owner, dict):
        return False  # written before owners existed -- treat as orphaned
    if owner.get("boot") and owner["boot"] != _boot_id():
        return False  # the machine rebooted; nothing from before survived it
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return False
    start = _proc_start(pid)
    if start is None:
        return False  # no such process
    if owner.get("start") is not None and start != owner["start"]:
        return False  # the PID was recycled onto something else
    return True


def _write_triage(report_id, payload):
    path = triage_path(report_id)
    tmp = path + ".part"
    payload["owner"] = _triage_owner()
    with open(tmp, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
