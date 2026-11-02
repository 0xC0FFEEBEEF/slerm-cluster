"""What a model's thinking can be set to, read from the chat template it is actually served with.

The gateway adds this to /v1/models as slerm.thinking, and Open WebUI's thinking pill draws its options from it,
so a new model gets the right pill without anyone writing rules for it (2026-10-08).

    caps(name, gguf_path) -> {"levels": [...], "default": "..."} or None (thinking can't be steered)

levels are the values a request may send as reasoning_effort: "none" (llama.cpp turns that into
enable_thinking=false), the template's own effort levels, or "on" = send nothing (thinking left on). The template
is the preset's chat-template-file when models.ini sets one, else the GGUF's tokenizer.chat_template. The default
is the preset's reasoning-effort, else the template's own default.
"""
import json
import os
import re
import struct

import slerm_topo

CACHE = os.path.join(slerm_topo.RUN, "thinking-caps-cache.json")
ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "on"]
KNOWN = {"minimal", "low", "medium", "high", "xhigh", "max"}

_T = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _gguf_template(path):
    with open(os.path.realpath(path), "rb") as f:
        if f.read(4) != b"GGUF":
            return ""
        _ver, _nt, nkv = struct.unpack("<IQQ", f.read(20))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n)

        for _ in range(nkv):
            k = rstr().decode("utf-8", "replace")
            t, = struct.unpack("<I", f.read(4))
            if k == "tokenizer.chat_template" and t == 8:
                return rstr().decode("utf-8", "replace")
            if t == 8:
                rstr()
            elif t == 9:
                at, n = struct.unpack("<IQ", f.read(12))
                if at == 8:
                    for _ in range(n):
                        rstr()
                else:
                    f.seek(_T[at] * n, 1)
            else:
                f.seek(_T[t], 1)
    return ""


def parse(template, preset_default=None):
    """{"levels", "default"} for a template, or None."""
    can_off = "enable_thinking" in template
    levels = []
    default = None
    if "reasoning_effort" in template:
        m = re.search(r"reasoning_effort\s+not\s+in\s+\(([^)]*)\)", template)
        listed = re.findall(r"['\"]([a-z]+)['\"]", m.group(1)) if m else []
        levels = [v for v in listed if v in KNOWN] or ["low", "medium", "high"]
        m = (re.search(r"reasoning_effort\s*\|\s*default\(\s*['\"]([a-z]+)['\"]", template)
             or re.search(r"set\s+reasoning_effort\s*=\s*['\"]([a-z]+)['\"]", template))
        default = m.group(1) if m else None
    elif not can_off:
        return None
    out = (["none"] if can_off else []) + (levels or ["on"])
    out = sorted(set(out), key=ORDER.index)
    if preset_default in out:
        default = preset_default
    if default not in out:
        default = "medium" if "medium" in out else out[-1]
    return {"levels": out, "default": default}


def caps(name, gguf_path, ini=None):
    """Thinking caps for one servable model (cached by template source and preset default)."""
    try:
        _top, star, sections = ini or slerm_topo.parse_ini()
        sec = dict(star)
        sec.update(sections.get(name, {}))
        src = sec.get("chat-template-file") or gguf_path
        st = os.stat(os.path.realpath(src))
        key = "%s|%s|%d|%d|%s" % (name, os.path.realpath(src), st.st_size, int(st.st_mtime),
                                  sec.get("reasoning-effort", ""))
        try:
            cache = json.load(open(CACHE))
        except (OSError, ValueError):
            cache = {}
        if key not in cache:
            text = open(src, encoding="utf-8").read() if src != gguf_path else _gguf_template(src)
            cache[key] = parse(text, sec.get("reasoning-effort"))
            tmp = CACHE + ".%d.tmp" % os.getpid()
            json.dump(cache, open(tmp, "w"))
            os.replace(tmp, CACHE)
        return cache[key]
    except Exception:  # noqa: BLE001 -- a pill is never worth breaking /v1/models
        return None


if __name__ == "__main__":
    ini = slerm_topo.parse_ini()
    for n, p in slerm_topo.servable().items():
        print("%-34s %s" % (n, caps(n, p, ini)))
