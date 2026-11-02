#!/usr/bin/env python3
"""Receive the opencode sandbox's live LLM traffic (from llm-tap.py over ssh stdin) and keep
/data/llm/run/live-opencode.json current for the panel's "Live model I/O" view.

Keeps the last KEEP exchanges; only the newest KEEP_REQ keep their full request body. Writes atomically
(tmp + rename) at most every 0.5 s, so the panel never reads a half-written file.
"""
import json, os, select, sys, time

OUT = os.environ.get("LIVE_TAP_OUT", "/data/llm/run/live-opencode.json")
KEEP, KEEP_REQ = 6, 3
ex = {}                       # (sess, id) -> exchange
dirty, last = False, 0.0


def write():
    global dirty, last
    keys = sorted(ex, key=lambda k: ex[k].get("t0", 0))[-KEEP:]
    for k in list(ex):
        if k not in keys:
            del ex[k]
    for k in keys[:-KEEP_REQ]:
        ex[k].pop("request", None)
    doc = {"updated": time.time(), "source": "opencode sandbox (llm-tap)",
           "exchanges": [ex[k] for k in reversed(keys)]}
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, OUT)
    dirty, last = False, time.time()


def handle(line):
    global dirty
    try:
        rec = json.loads(line)
    except Exception:
        return
    key = (rec.get("sess"), rec.get("id"))
    e = ex.setdefault(key, {"sess": key[0], "id": key[1], "t0": rec.get("t", time.time())})
    if rec.get("type") == "req":
        e["t0"], e["path"], e["request"] = rec.get("t"), rec.get("path"), rec.get("request")
    else:
        for k in ("reasoning", "content", "tool_calls", "finish", "usage", "timings", "status", "done"):
            if k in rec:
                e[k] = rec[k]
        e["t"] = rec.get("t")
    dirty = True


buf = b""
while True:
    ready, _, _ = select.select([sys.stdin.buffer], [], [], 0.5)
    if ready:
        chunk = os.read(sys.stdin.fileno(), 1 << 20)
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            handle(line)
    if dirty and time.time() - last >= 0.5:
        write()
if dirty:
    write()
