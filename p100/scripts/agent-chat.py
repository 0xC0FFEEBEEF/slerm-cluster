#!/usr/bin/env python3
"""Print the latest Open WebUI chat (Computer Agent run) with token-like strings masked. Read-only."""
import json
import re
import sqlite3
import sys

c = sqlite3.connect("file:/home/slerm/.open-webui/webui.db?mode=ro", uri=True)
cid, title, upd, blob = c.execute("select id, title, updated_at, chat from chat order by updated_at desc limit 1").fetchone()
d = json.loads(blob)
msgs = d.get("messages") or list((d.get("history") or {}).get("messages", {}).values())


def clean(s):
    s = re.sub(r"([?&](?:t|token|key|s)=)[^&\s\"')]+", r"\1<masked>", s)
    return re.sub(r"\b[0-9a-fA-F]{24,}\b", "<hex>", s)


limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
print("chat:", title, "| messages:", len(msgs))
for m in msgs:
    role, model = m.get("role"), m.get("model", "")
    text = clean(m.get("content") or "")
    print("=== %s %s (%d chars)" % (role, model, len(text)))
    print(text[:limit])
