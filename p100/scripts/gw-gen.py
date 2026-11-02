#!/usr/bin/env python3
"""Generation speed through the gateway with a LONG answer (~450 tokens), thinking
off, so tok/s is not dominated by the first few tokens. 2026-10-04.

    gw-gen.py MODEL [MODEL...]
"""
import json, sys, time, urllib.request

GW = "http://slerm-head:8090/v1/chat/completions"

for model in sys.argv[1:]:
    body = {"model": model, "max_tokens": 450, "temperature": 0.7, "stream": False,
            "messages": [{"role": "user", "content": "Write a 400-word short story about a lighthouse keeper."}],
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
    t0 = time.time()
    try:
        req = urllib.request.Request(GW, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3000) as r:
            d = json.load(r)
        t = d.get("timings", {})
        print("%-28s gen %4d tok @ %5.1f tok/s  (wall %ds)" % (model, t.get("predicted_n", 0),
              t.get("predicted_per_second", 0), time.time() - t0), flush=True)
    except Exception as e:                                       # noqa: BLE001
        print("%-28s FAILED %s: %s" % (model, type(e).__name__, e), flush=True)
