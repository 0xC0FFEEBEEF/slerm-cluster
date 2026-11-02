#!/usr/bin/env python3
"""Client half of jobs/p100-loads.sh (2026-10-04). Python 3.9, stdlib only.

  p100-loads.py longctx PORT      gpt-oss-20b at ~20k / ~45k / ~60k prompt tokens
  p100-loads.py parallel PORT     qwen3-14b: 1 request, then 4 at once
  p100-loads.py vision PORT       generated PNG -> what shapes / colours
"""
import base64, glob, json, struct, sys, threading, time, urllib.request, zlib

SRC = "/data/llm/src/llama.cpp/src"


def post(port, path, body, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(port, content, max_tokens, **extra):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "cache_prompt": False}
    body.update(extra)
    t0 = time.time()
    d = post(port, "/v1/chat/completions", body)
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or ""), d.get("timings", {}), time.time() - t0


def fmt(t):
    return "prompt %5d tok @ %6.1f tok/s | gen %4d tok @ %5.1f tok/s" % (
        t.get("prompt_n", 0), t.get("prompt_per_second", 0), t.get("predicted_n", 0), t.get("predicted_per_second", 0))


def source_text(port, target_tokens):
    """Real C++ from the llama.cpp tree, trimmed to about target_tokens."""
    files = sorted(glob.glob(SRC + "/*.cpp"), key=lambda p: -len(open(p, errors="replace").read()))
    text = ""
    for p in files:
        text += "\n// ===== %s =====\n" % p.rsplit("/", 1)[-1] + open(p, errors="replace").read()
        if len(text) > target_tokens * 5:
            break
    lo, hi = 0, len(text)
    while hi - lo > 2000:                       # bisect on the server's own tokenizer
        mid = (lo + hi) // 2
        n = len(post(port, "/tokenize", {"content": text[:mid]})["tokens"])
        lo, hi = (mid, hi) if n < target_tokens else (lo, mid)
    return text[:lo]


def longctx(port):
    q = "\n\nAbove is C++ source code. Name three functions that are defined in it. Answer in one short line."
    for target in (20000, 45000, 60000):
        text = source_text(port, target)
        try:
            c, r, t, wall = chat(port, text + q, 2048, chat_template_kwargs={"reasoning_effort": "low"})
            print("~%dk: %s | %.0fs wall | reasoning %d chars | answer: %r" % (
                target // 1000, fmt(t), wall, len(r), c[:160]), flush=True)
        except Exception as e:                                   # noqa: BLE001
            print("~%dk: FAILED %s: %s" % (target // 1000, type(e).__name__, e), flush=True)
            return


def parallel(port):
    topics = ["a lighthouse keeper", "a lost robot", "a desert caravan", "a deep-sea diver"]
    kw = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7}
    c, r, t, wall = chat(port, "Write a 250-word story about %s." % topics[0], 400, **kw)
    print("single : %s | %.1fs" % (fmt(t), wall), flush=True)
    out = [None] * 4

    def run(i):
        out[i] = chat(port, "Write a 250-word story about %s." % topics[i], 400, **kw)
    th = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    wall = time.time() - t0
    toks = sum(o[2].get("predicted_n", 0) for o in out)
    for i, o in enumerate(out):
        print("slot %d : %s" % (i, fmt(o[2])), flush=True)
    print("4 at once: %d tokens in %.1fs = %.1f tok/s aggregate" % (toks, wall, toks / wall), flush=True)


def png(w, h, px):
    raw = b"".join(b"\x00" + bytes(px[y * w * 3:(y + 1) * w * 3]) for y in range(h))
    chunk = lambda k, d: struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def vision(port):
    w = h = 448
    px = bytearray([255] * (w * h * 3))
    for y in range(h):
        for x in range(w):
            if (x - 120) ** 2 + (y - 224) ** 2 < 80 ** 2:
                px[(y * w + x) * 3:(y * w + x) * 3 + 3] = b"\xdc\x14\x14"      # red circle, left
            elif 260 <= x <= 400 and 154 <= y <= 294:
                px[(y * w + x) * 3:(y * w + x) * 3 + 3] = b"\x14\x3c\xdc"      # blue square, right
    url = "data:image/png;base64," + base64.b64encode(png(w, h, px)).decode()
    content = [{"type": "image_url", "image_url": {"url": url}},
               {"type": "text", "text": "What shapes are in this image, what colour is each, and which is on the left? One sentence."}]
    for i in range(2):
        c, r, t, wall = chat(port, content, 80)
        print("image run %d: %s | %.1fs | %r" % (i + 1, fmt(t), wall, c), flush=True)


if __name__ == "__main__":
    {"longctx": longctx, "parallel": parallel, "vision": vision}[sys.argv[1]](int(sys.argv[2]))
