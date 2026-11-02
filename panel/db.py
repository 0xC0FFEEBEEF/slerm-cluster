"""SQLite storage for runs, schedules, probes, and Wazuh analysis state."""
import json
import os
import sqlite3
import threading
import time

import config

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project      TEXT NOT NULL,
    params_json  TEXT NOT NULL DEFAULT '{}',
    options_json TEXT NOT NULL DEFAULT '{}',
    source       TEXT NOT NULL DEFAULT 'manual',
    schedule_id  INTEGER,
    state        TEXT NOT NULL DEFAULT 'listing',
    message      TEXT,
    list_path    TEXT,
    item_count   INTEGER,
    job_id       INTEGER,
    submit_line  TEXT,
    frames_before INTEGER,
    created      REAL NOT NULL,
    updated      REAL NOT NULL,
    finished     REAL
);
CREATE INDEX IF NOT EXISTS runs_project_idx ON runs(project, id DESC);

CREATE TABLE IF NOT EXISTS schedules (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project      TEXT NOT NULL,
    label        TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL DEFAULT 'cron',
    expr         TEXT NOT NULL,
    params_json  TEXT NOT NULL DEFAULT '{}',
    options_json TEXT NOT NULL DEFAULT '{}',
    enabled      INTEGER NOT NULL DEFAULT 1,
    last_fired   REAL,
    next_fire    REAL,
    last_run_id  INTEGER,
    last_error   TEXT,
    created      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS probes (
    node       TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    ok         INTEGER NOT NULL,
    data_json  TEXT,
    error      TEXT
);

CREATE TABLE IF NOT EXISTS wazuh_fetches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile       TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'queued',
    message       TEXT,
    index_pattern TEXT,
    query         TEXT,
    t_start       TEXT,
    t_end         TEXT,
    alerts        INTEGER NOT NULL DEFAULT 0,
    bytes         INTEGER NOT NULL DEFAULT 0,
    path          TEXT,
    created       REAL NOT NULL,
    updated       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS wazuh_fetches_profile_idx
    ON wazuh_fetches(profile, id DESC);
CREATE INDEX IF NOT EXISTS wazuh_fetches_state_idx
    ON wazuh_fetches(state, id DESC);

CREATE TABLE IF NOT EXISTS wazuh_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    profile     TEXT,
    fetch_id    INTEGER,
    run_id      INTEGER,
    reduce_job_id INTEGER,
    filename    TEXT,
    state       TEXT NOT NULL DEFAULT 'queued',
    message     TEXT,
    alerts      INTEGER NOT NULL DEFAULT 0,
    findings    INTEGER NOT NULL DEFAULT 0,
    actionable  INTEGER NOT NULL DEFAULT 0,
    truncated   INTEGER NOT NULL DEFAULT 0,
    report_path TEXT,
    created     REAL NOT NULL,
    updated     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS wazuh_reports_profile_idx
    ON wazuh_reports(profile, id DESC);
CREATE INDEX IF NOT EXISTS wazuh_reports_fetch_idx
    ON wazuh_reports(fetch_id, id DESC);
CREATE INDEX IF NOT EXISTS wazuh_reports_state_idx
    ON wazuh_reports(state, id DESC);

-- One completed inference request served by the cluster, whoever asked for it.
-- `ts` is when the panel OBSERVED the request finish, not a time read out of
-- the log: llama.cpp's log clock is per-stream and has no shared zero, so it
-- cannot be turned into wall time. See llmlog.py.
CREATE TABLE IF NOT EXISTS llm_requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL NOT NULL,
    job_id           INTEGER,
    node             TEXT,
    port             INTEGER,
    model            TEXT,
    task_id          INTEGER,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    generated_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens     INTEGER NOT NULL DEFAULT 0,
    context_tokens   INTEGER NOT NULL DEFAULT 0,
    prompt_ms        REAL NOT NULL DEFAULT 0,
    eval_ms          REAL NOT NULL DEFAULT 0,
    total_ms         REAL NOT NULL DEFAULT 0,
    prompt_tok_s     REAL NOT NULL DEFAULT 0,
    generated_tok_s  REAL NOT NULL DEFAULT 0,
    truncated        INTEGER NOT NULL DEFAULT 0,
    UNIQUE(job_id, port, task_id)
);
CREATE INDEX IF NOT EXISTS llm_requests_ts_idx ON llm_requests(ts DESC);
CREATE INDEX IF NOT EXISTS llm_requests_model_idx ON llm_requests(model, ts DESC);

-- A model being loaded, evicted, or giving up its slot. --models-max 1 means
-- these happen constantly and explain both the latency spikes and the
-- /metrics counters going back to zero.
CREATE TABLE IF NOT EXISTS llm_model_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    job_id  INTEGER,
    node    TEXT,
    model   TEXT,
    event   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_model_events_ts_idx ON llm_model_events(ts DESC);

-- Where the tail of each server log has been read to. Keyed by path; the inode
-- is stored so a job restarting into the same filename is detected as a new
-- file rather than seeking past the end of a short one.
CREATE TABLE IF NOT EXISTS llm_log_cursors (
    path    TEXT PRIMARY KEY,
    inode   INTEGER,
    offset  INTEGER NOT NULL DEFAULT 0,
    job_id  INTEGER,
    updated REAL NOT NULL
);

-- Last /metrics reading per model instance, so the next poll can store the
-- delta. These counters restart at zero on every model load.
CREATE TABLE IF NOT EXISTS llm_counters (
    key       TEXT PRIMARY KEY,
    ts        REAL NOT NULL,
    data_json TEXT NOT NULL
);
"""


def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init():
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    with _lock, connect() as conn:
        conn.executescript(SCHEMA)
        # CREATE TABLE IF NOT EXISTS does not add columns to an existing panel
        # database. Keep this migration beside the schema so older installs
        # gain durable tracking of the dependent reducer job on restart.
        columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(wazuh_reports)").fetchall()}
        if "reduce_job_id" not in columns:
            conn.execute(
                "ALTER TABLE wazuh_reports ADD COLUMN reduce_job_id INTEGER")


def query(sql, args=()):
    with _lock, connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def one(sql, args=()):
    rows = query(sql, args)
    return rows[0] if rows else None


def execute(sql, args=()):
    with _lock, connect() as conn:
        cur = conn.execute(sql, args)
        return cur.lastrowid


# --- runs -------------------------------------------------------------------

def create_run(project, params, options, source="manual", schedule_id=None,
               frames_before=None):
    now = time.time()
    return execute(
        "INSERT INTO runs (project, params_json, options_json, source, schedule_id,"
        " state, frames_before, created, updated) VALUES (?,?,?,?,?,?,?,?,?)",
        (project, json.dumps(params), json.dumps(options), source, schedule_id,
         "listing", frames_before, now, now))


def update_run(run_id, **fields):
    if not fields:
        return
    fields["updated"] = time.time()
    if fields.get("state") in ("completed", "failed", "cancelled"):
        fields.setdefault("finished", time.time())
    cols = ", ".join("%s = ?" % k for k in fields)
    execute("UPDATE runs SET %s WHERE id = ?" % cols,
            tuple(fields.values()) + (run_id,))


def get_run(run_id):
    row = one("SELECT * FROM runs WHERE id = ?", (run_id,))
    return _hydrate_run(row) if row else None


def list_runs(project=None, limit=50):
    if project:
        rows = query("SELECT * FROM runs WHERE project = ? ORDER BY id DESC LIMIT ?",
                     (project, limit))
    else:
        rows = query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    return [_hydrate_run(r) for r in rows]


def active_runs(project=None):
    live = ("listing", "submitting", "submitted", "running")
    marks = ",".join("?" * len(live))
    if project:
        rows = query("SELECT * FROM runs WHERE project = ? AND state IN (%s)" % marks,
                     (project,) + live)
    else:
        rows = query("SELECT * FROM runs WHERE state IN (%s)" % marks, live)
    return [_hydrate_run(r) for r in rows]


def _hydrate_run(row):
    row = dict(row)
    row["params"] = json.loads(row.pop("params_json", "{}") or "{}")
    row["options"] = json.loads(row.pop("options_json", "{}") or "{}")
    return row


# --- schedules --------------------------------------------------------------

def create_schedule(project, label, kind, expr, params, options, enabled=True,
                    next_fire=None):
    return execute(
        "INSERT INTO schedules (project, label, kind, expr, params_json,"
        " options_json, enabled, next_fire, created) VALUES (?,?,?,?,?,?,?,?,?)",
        (project, label, kind, expr, json.dumps(params), json.dumps(options),
         1 if enabled else 0, next_fire, time.time()))


def update_schedule(sched_id, **fields):
    if not fields:
        return
    for key in ("params", "options"):
        if key in fields:
            fields[key + "_json"] = json.dumps(fields.pop(key))
    cols = ", ".join("%s = ?" % k for k in fields)
    execute("UPDATE schedules SET %s WHERE id = ?" % cols,
            tuple(fields.values()) + (sched_id,))


def delete_schedule(sched_id):
    execute("DELETE FROM schedules WHERE id = ?", (sched_id,))


def get_schedule(sched_id):
    row = one("SELECT * FROM schedules WHERE id = ?", (sched_id,))
    return _hydrate_schedule(row) if row else None


def list_schedules(project=None):
    if project:
        rows = query("SELECT * FROM schedules WHERE project = ? ORDER BY id", (project,))
    else:
        rows = query("SELECT * FROM schedules ORDER BY id")
    return [_hydrate_schedule(r) for r in rows]


def _hydrate_schedule(row):
    row = dict(row)
    row["params"] = json.loads(row.pop("params_json", "{}") or "{}")
    row["options"] = json.loads(row.pop("options_json", "{}") or "{}")
    row["enabled"] = bool(row["enabled"])
    return row


# --- probes -----------------------------------------------------------------

def save_probe(node, ok, data=None, error=None):
    execute("INSERT INTO probes (node, ts, ok, data_json, error) VALUES (?,?,?,?,?)"
            " ON CONFLICT(node) DO UPDATE SET ts=excluded.ts, ok=excluded.ok,"
            " data_json=excluded.data_json, error=excluded.error",
            (node, time.time(), 1 if ok else 0,
             json.dumps(data) if data is not None else None, error))


def list_probes():
    out = {}
    for row in query("SELECT * FROM probes"):
        out[row["node"]] = {
            "ts": row["ts"],
            "ok": bool(row["ok"]),
            "gpus": json.loads(row["data_json"]) if row["data_json"] else None,
            "error": row["error"],
        }
    return out


# --- Wazuh fetches ----------------------------------------------------------

_WAZUH_FETCH_FIELDS = (
    "profile", "state", "message", "index_pattern", "query", "t_start",
    "t_end", "alerts", "bytes", "path",
)
_WAZUH_REPORT_FIELDS = (
    "source", "profile", "fetch_id", "run_id", "reduce_job_id", "filename",
    "state", "message", "alerts", "findings", "actionable", "truncated",
    "report_path",
)


def _check_fields(fields, allowed):
    unknown = sorted(set(fields) - set(allowed))
    if unknown:
        raise ValueError("unknown database field(s): %s" % ", ".join(unknown))


def _insert_record(table, columns, values):
    _check_fields(values, columns)
    now = time.time()
    names = [name for name in columns if name in values]
    names.extend(("created", "updated"))
    marks = ",".join("?" for _ in names)
    sql = "INSERT INTO %s (%s) VALUES (%s)" % (
        table, ", ".join(names), marks)
    args = [values[name] for name in names[:-2]] + [now, now]
    return execute(sql, tuple(args))


def _update_record(table, row_id, fields, allowed):
    if not fields:
        return
    _check_fields(fields, allowed)
    fields = dict(fields)
    fields["updated"] = time.time()
    cols = ", ".join("%s = ?" % key for key in fields)
    execute("UPDATE %s SET %s WHERE id = ?" % (table, cols),
            tuple(fields.values()) + (row_id,))


def _row_limit(limit):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be a positive integer")
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    return min(limit, 1000)


def create_wazuh_fetch(profile, index_pattern=None, query=None, t_start=None,
                       t_end=None, **fields):
    """Create a persisted indexer fetch and return its integer id."""
    if not profile:
        raise ValueError("profile is required")
    values = {
        "profile": profile,
        "index_pattern": index_pattern,
        "query": query,
        "t_start": t_start,
        "t_end": t_end,
        "state": "queued",
        "alerts": 0,
        "bytes": 0,
    }
    values.update(fields)
    return _insert_record("wazuh_fetches", _WAZUH_FETCH_FIELDS, values)


def update_wazuh_fetch(fetch_id, **fields):
    _update_record("wazuh_fetches", fetch_id, fields, _WAZUH_FETCH_FIELDS)


def get_wazuh_fetch(fetch_id):
    return one("SELECT * FROM wazuh_fetches WHERE id = ?", (fetch_id,))


def list_wazuh_fetches(profile=None, state=None, limit=100):
    clauses = []
    args = []
    if profile is not None:
        clauses.append("profile = ?")
        args.append(profile)
    if state is not None:
        clauses.append("state = ?")
        args.append(state)
    sql = "SELECT * FROM wazuh_fetches"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(_row_limit(limit))
    return query(sql, tuple(args))


# --- Wazuh reports ----------------------------------------------------------

def create_wazuh_report(source, profile=None, fetch_id=None, filename=None,
                        **fields):
    """Create a report record for a fetch or uploaded export."""
    if not source:
        raise ValueError("source is required")
    values = {
        "source": source,
        "profile": profile,
        "fetch_id": fetch_id,
        "filename": filename,
        "state": "queued",
        "alerts": 0,
        "findings": 0,
        "actionable": 0,
        "truncated": 0,
    }
    values.update(fields)
    values["truncated"] = 1 if values.get("truncated") else 0
    return _insert_record("wazuh_reports", _WAZUH_REPORT_FIELDS, values)


def update_wazuh_report(report_id, **fields):
    if "truncated" in fields:
        fields["truncated"] = 1 if fields["truncated"] else 0
    _update_record("wazuh_reports", report_id, fields, _WAZUH_REPORT_FIELDS)


def _hydrate_wazuh_report(row):
    if not row:
        return None
    row = dict(row)
    row["truncated"] = bool(row.get("truncated"))
    return row


def get_wazuh_report(report_id):
    return _hydrate_wazuh_report(
        one("SELECT * FROM wazuh_reports WHERE id = ?", (report_id,)))


def list_wazuh_reports(profile=None, fetch_id=None, state=None, source=None,
                       limit=100):
    clauses = []
    args = []
    if profile is not None:
        clauses.append("profile = ?")
        args.append(profile)
    if fetch_id is not None:
        clauses.append("fetch_id = ?")
        args.append(fetch_id)
    if state is not None:
        clauses.append("state = ?")
        args.append(state)
    if source is not None:
        clauses.append("source = ?")
        args.append(source)
    sql = "SELECT * FROM wazuh_reports"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(_row_limit(limit))
    return [_hydrate_wazuh_report(row) for row in query(sql, tuple(args))]


# --- cluster inference accounting -------------------------------------------

_LLM_REQUEST_FIELDS = (
    "ts", "job_id", "node", "port", "model", "task_id", "prompt_tokens",
    "generated_tokens", "total_tokens", "context_tokens", "prompt_ms",
    "eval_ms", "total_ms", "prompt_tok_s", "generated_tok_s", "truncated",
)


_LLM_REQUEST_NUMERIC = (
    "prompt_tokens", "generated_tokens", "total_tokens", "context_tokens",
    "prompt_ms", "eval_ms", "total_ms", "prompt_tok_s", "generated_tok_s",
    "truncated",
)


def insert_llm_requests(rows):
    """Append observed requests, ignoring any already recorded.

    A server log is tailed from a stored offset, but a restarted panel or a
    re-read file can present the same request twice. The UNIQUE key makes that
    a no-op rather than a double count.

    Every counter is coerced to 0 first. `INSERT OR IGNORE` ignores ANY
    constraint violation, not just the uniqueness one it is here for, so a
    single None in a NOT NULL counter silently drops the whole row -- which is
    exactly what happened to the first request this ever parsed: a call with no
    `release` line left `context_tokens` unset and the insert reported success
    while storing nothing. A column DEFAULT does not help, because passing NULL
    explicitly bypasses it.
    """
    if not rows:
        return 0
    names = list(_LLM_REQUEST_FIELDS)
    sql = ("INSERT OR IGNORE INTO llm_requests (%s) VALUES (%s)"
           % (", ".join(names), ",".join("?" for _ in names)))
    values = []
    for row in rows:
        clean = dict(row)
        for name in _LLM_REQUEST_NUMERIC:
            if clean.get(name) is None:
                clean[name] = 0
        values.append(tuple(clean.get(name) for name in names))
    with _lock, connect() as conn:
        cur = conn.executemany(sql, values)
        return cur.rowcount


def insert_llm_events(rows):
    if not rows:
        return 0
    with _lock, connect() as conn:
        cur = conn.executemany(
            "INSERT INTO llm_model_events (ts, job_id, node, model, event)"
            " VALUES (?,?,?,?,?)",
            [(r.get("ts"), r.get("job_id"), r.get("node"), r.get("model"),
              r.get("event")) for r in rows])
        return cur.rowcount


def get_log_cursor(path):
    return one("SELECT * FROM llm_log_cursors WHERE path = ?", (path,))


def save_log_cursor(path, inode, offset, job_id):
    execute("INSERT INTO llm_log_cursors (path, inode, offset, job_id, updated)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(path) DO UPDATE SET inode=excluded.inode,"
            " offset=excluded.offset, job_id=excluded.job_id,"
            " updated=excluded.updated",
            (path, inode, offset, job_id, time.time()))


def get_llm_counter(key):
    row = one("SELECT * FROM llm_counters WHERE key = ?", (key,))
    if not row:
        return None
    try:
        return {"ts": row["ts"], "data": json.loads(row["data_json"])}
    except ValueError:
        return None


def save_llm_counter(key, data):
    execute("INSERT INTO llm_counters (key, ts, data_json) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET ts=excluded.ts,"
            " data_json=excluded.data_json",
            (key, time.time(), json.dumps(data)))


def prune_llm(keep_days):
    """Drop request and event rows older than keep_days. 0 keeps everything."""
    if not keep_days:
        return 0
    cutoff = time.time() - float(keep_days) * 86400.0
    with _lock, connect() as conn:
        dropped = conn.execute(
            "DELETE FROM llm_requests WHERE ts < ?", (cutoff,)).rowcount
        dropped += conn.execute(
            "DELETE FROM llm_model_events WHERE ts < ?", (cutoff,)).rowcount
        return dropped
