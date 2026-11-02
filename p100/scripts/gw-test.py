#!/usr/bin/env python3
"""Acceptance test of slerm models THROUGH THE GATEWAY (2026-10-04, P100 rework).

For each model: one chat request with ~2k tokens of real C++ and a question whose
answer can be checked, thinking off where the template allows it. Reports wall
time (includes any router restart / model load), llama.cpp timings, the answer,
and per-GPU VRAM on slerm-node04 right after. A successful load also proves the
preset's full ctx-size fits: llama.cpp allocates the whole KV cache at load.

    gw-test.py MODEL [MODEL...]          results appended to gw-test.jsonl
"""
import json, subprocess, sys, time, urllib.request

GW = "http://slerm-head:8090/v1/chat/completions"
SRC = "/data/llm/src/llama.cpp/src/llama-chat.cpp"   # first function: trim
OUT = "/home/slerm/slerm-dynamic/p100/gw-test.jsonl"


def source():
    txt = subprocess.run(["ssh", "-o", "BatchMode=yes", "slerm-head", "head -c 7000 " + SRC],
                         capture_output=True, text=True, timeout=30).stdout
    return txt


def vram():
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "slerm-node04",
                        "nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    return [line.strip() for line in r.stdout.strip().splitlines()]


def run(model, code):
    body = {"model": model, "max_tokens": 400, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": code + "\n\nIn the C++ above, what is the name of the "
                          "first function defined? Reply with just the function name."}],
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
    t0 = time.time()
    req = urllib.request.Request(GW, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3000) as r:
            d = json.load(r)
    except Exception as e:                                       # noqa: BLE001
        body = getattr(e, "read", lambda: b"")()[:300]
        return {"model": model, "ok": False, "error": "%s: %s %s" % (type(e).__name__, e, body), "wall_s": round(time.time() - t0)}
    m = d["choices"][0]["message"]
    t = d.get("timings", {})
    return {"model": model, "ok": True, "wall_s": round(time.time() - t0),
            "prompt_n": t.get("prompt_n"), "prompt_tps": round(t.get("prompt_per_second", 0), 1),
            "gen_n": t.get("predicted_n"), "gen_tps": round(t.get("predicted_per_second", 0), 1),
            "answer": (m.get("content") or "").strip()[:80], "reasoning_chars": len(m.get("reasoning_content") or "")}


if __name__ == "__main__":
    code = source()
    for model in sys.argv[1:]:
        res = run(model, code)
        res["vram_mib_temp"] = vram()
        res["at"] = time.strftime("%H:%M:%S")
        print(json.dumps(res), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
