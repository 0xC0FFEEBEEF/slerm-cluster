#!/usr/bin/env python3
"""Make opencode's slerm model list match the router presets (2026-10-04).

Reads /data/llm/etc/models.ini (via ssh) and the servable list, then in
~/.config/opencode/opencode.json sets limit.context = the preset's ctx-size for
every slerm model, adds servable models it lacks, drops ones no longer servable,
and keeps every other key (names, modalities, output limits) as it was.
Dry run by default; --apply writes it after a timestamped backup.
"""
import json, re, shutil, subprocess, sys, time

CFG = "/home/slerm/.config/opencode/opencode.json"
VISION = {"qwen3.8-27b", "qwen3.6-35b-a3b-ud-q4kxl", "bonsai-27b", "holo-3.1-9b", "holo4-35b-a3b", "qwen3.5-9b", "ui-venus-2-9b", "gemma-3-4b-it-q4km"}


def remote(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "slerm-head", cmd],
                          capture_output=True, text=True, timeout=30, check=True).stdout


def presets():
    ini, out, cur, star = remote("cat /data/llm/etc/models.ini"), {}, None, 8192
    for line in ini.splitlines():
        s = line.strip()
        m = re.match(r"^\[(.+)\]$", s)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^ctx-size\s*=\s*(\d+)", s)
        if m:
            if cur == "*":
                star = int(m.group(1))
            elif cur:
                out[cur] = int(m.group(1))
    return out, star


def main():
    apply = "--apply" in sys.argv
    skip = set()
    for a in sys.argv[1:]:
        if a.startswith("--skip="):
            skip |= set(a[7:].split(","))   # servable but not load-tested yet: leave opencode alone for these
    ctx, star = presets()
    servable = [n[:-5] for n in remote("ls /data/llm/models/servable/").split() if n.endswith(".gguf")]
    servable = [n for n in servable if n not in skip]
    cfg = json.load(open(CFG))
    models = cfg["provider"]["slerm"]["models"]
    for name in sorted(set(models) - set(servable) - skip):
        print("drop    %s (not servable)" % name)
        del models[name]
    for name in sorted(servable):
        want = ctx.get(name, star)
        m = models.get(name)
        if m is None:
            m = models[name] = {"name": "%s (slerm)" % name, "tool_call": True,
                                "limit": {"context": want, "output": 4096}}
            if name in VISION:
                m["modalities"] = {"input": ["text", "image"], "output": ["text"]}
            print("add     %s ctx %d" % (name, want))
            continue
        old = m.setdefault("limit", {}).get("context")
        if old != want:
            print("context %-28s %6s -> %d" % (name, old, want))
            m["limit"]["context"] = want
    if apply:
        bak = CFG + ".bak-p100-" + time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(CFG, bak)
        tmp = CFG + ".tmp"
        json.dump(cfg, open(tmp, "w"), indent=2)
        open(tmp, "a").write("\n")
        shutil.move(tmp, CFG)
        print("written; backup", bak)
    else:
        print("(dry run; --apply to write)")


if __name__ == "__main__":
    main()
