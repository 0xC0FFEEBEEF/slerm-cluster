#!/usr/bin/env python3
"""Which models can run on the slerm devices that are available right now,
and with what tensor-split. Shared by the orchestrator (to start the right
router) and the panel (to grey out model cards). Python 3.9.

The router's device list is fixed when it starts, and etc/models.ini holds
splits tuned for ONE device set -- the full pool (2026-10-04, two Tesla P100s,
both in node04, which is the only GPU node and always the router when up):

    [node01 CPU] [node02 CPU] [node05 CPU] [node04 GPU0] [node04 GPU1]
     (--rpc workers in order, then the router's local CUDA devices)

A topology names NODES ({"router", "rpc"}); slots() expands the router into its
local GPUs, so a split has one value per slot while everything about power,
Slurm and staging still reasons per node. When the pool changes (byteme2 needs
node01/02's RAM, deadlock is busy or off) every split must be recomputed, or
shares land on the wrong devices silently -- the 2026-09-11 "0% local GPU"
trap. Rules:

  * The full pool keeps the tuned splits verbatim (tested=true).
  * GPU shares are never increased: they are tuned to leave KV slack per card.
  * A lost CPU share moves to the remaining CPU devices within their RAM, with
    a margin; if it does not fit, the model is unavailable, with the reason.
    Dense models carry no CPU share in models.ini at all (qwen3-14b with 10% on
    node05: prompt 123 -> 29 tok/s, generation 16.6 -> 9.3).
  * No GPU at all (node04 down or taken by a user VM): the router runs on
    deadlock with the CPU build and a model is offered only when its estimated
    speed is usable (MIN_CPU_TOKS).

    python3 slerm_topo.py plan '{"router":"slerm-node04","rpc":["slerm-node05"]}'
"""
import json
import os
import re
import struct
import sys
import time

LLM = "/data/llm"
INI = os.path.join(LLM, "etc/models.ini")
SERVABLE = os.path.join(LLM, "models/servable")
RUN = os.path.join(LLM, "run")
TOPO_DIR = os.path.join(RUN, "topo")
GGUF_CACHE = os.path.join(RUN, "topo-gguf-cache.json")

FULL_ORDER = ["slerm-node01", "slerm-node02", "slerm-node05", "slerm-node04/gpu0", "slerm-node04/gpu1"]
# GPU node -> Slurm gres type; its local CUDA devices, in CUDA order (gpu0 is the
# x16 card at host 82:00, gpu1 the x8 one at 04:00).
GPU_NODES = {"slerm-node04": "pascal"}
GPU_SLOTS = {"slerm-node04": ["slerm-node04/gpu0", "slerm-node04/gpu1"]}
_GPU_SLOT_SET = {s for v in GPU_SLOTS.values() for s in v}
# RAM a CPU RPC worker may hold: the llm-rpccpu job limits (11G / 28G) minus
# ~1 GB for the server itself. node05's is lowered when the user is using it.
CPU_CAP_GB = {"slerm-node01": 10.0, "slerm-node02": 10.0, "slerm-node05": 26.0}
CPU_MARGIN = 0.90
# CPU-only fallback on deadlock (E5-1620 v2, 4x DDR3): generation is memory
# bandwidth bound, ~ EFFECTIVE_GBPS / bytes read per token.
EFFECTIVE_GBPS = 18.0
MIN_CPU_TOKS = 3.0
CPU_ONLY_CTX = 16384
CPU_ROUTER_RESERVE_GB = 3.0
CPU_ONLY_NODE05_GB = 28.5    # 31.7 GB minus the OS; the host report lowers it further     # KV + compute + prompt cache on the router itself


# ------------------------------------------------------------------ ini ----

def parse_ini(path=INI):
    """-> (globals, [*] dict, {section: {key: value}}), preserving key order."""
    top, star, sections, cur = {}, {}, {}, None
    for line in open(path):
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("#"):
            continue
        m = re.match(r"^\[(.+)\]$", s)
        if m:
            cur = m.group(1)
            if cur != "*":
                sections[cur] = {}
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip()
        if cur is None:
            top[k] = v
        elif cur == "*":
            star[k] = v
        else:
            sections[cur][k] = v
    return top, star, sections


# ----------------------------------------------------------------- gguf ----

_T = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_str(f):
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n)


def _skip_val(f, t):
    if t in _T:
        f.seek(_T[t], 1)
    elif t == 8:
        n, = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
    elif t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et in _T:
            f.seek(_T[et] * n, 1)
        else:
            for _ in range(n):
                _skip_val(f, et)


def _read_val(f, t):
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
    if t in fmt:
        return struct.unpack(fmt[t], f.read(_T[t]))[0]
    if t == 8:
        return _read_str(f).decode("utf-8", "replace")
    _skip_val(f, t)
    return None


def gguf_facts(path):
    """size, architecture, expert counts -- from the header only."""
    real = os.path.realpath(path)
    st = os.stat(real)
    cache = {}
    try:
        cache = json.load(open(GGUF_CACHE))
    except (OSError, ValueError):
        pass
    key = "%s|%d|%d" % (real, st.st_size, int(st.st_mtime))
    if key in cache:
        return cache[key]
    want = ("general.architecture", "expert_count", "expert_used_count", "block_count")
    out = {"bytes": st.st_size}
    with open(real, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF: " + real)
        _ver, _nt, nkv = struct.unpack("<IQQ", f.read(20))
        for _ in range(nkv):
            k = _read_str(f).decode("utf-8", "replace")
            t, = struct.unpack("<I", f.read(4))
            if k == "general.architecture" or k.endswith(want[1:]):
                v = _read_val(f, t)
                out[k.split(".")[-1] if k != "general.architecture" else "arch"] = v
            else:
                _skip_val(f, t)
    ec = out.get("expert_count") or 0
    eu = out.get("expert_used_count") or 0
    out["moe"] = bool(ec and eu and ec > eu)
    # Bytes read per generated token: expert tensors dominate an MoE file.
    out["active_bytes"] = int(st.st_size * ((0.15 + 0.85 * eu / ec) if out["moe"] else 1.0))
    cache[key] = out
    try:
        tmp = GGUF_CACHE + ".%d.tmp" % os.getpid()
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, GGUF_CACHE)
    except OSError:
        pass
    return out


def servable():
    out = {}
    for n in sorted(os.listdir(SERVABLE)):
        if n.endswith(".gguf") and os.path.isfile(os.path.join(SERVABLE, n)):
            out[n[:-5]] = os.path.join(SERVABLE, n)
    return out


# ----------------------------------------------------------------- plan ----

def devices(topo):
    """NODES a router uses: rpc workers in order, then the router node."""
    return list(topo.get("rpc", [])) + [topo["router"]]


def slots(topo):
    """llama.cpp DEVICES, i.e. one tensor-split value each: the rpc workers, then
    the router's local GPUs (or the router itself when it is CPU-only)."""
    return list(topo.get("rpc", [])) + GPU_SLOTS.get(topo["router"], [topo["router"]])


def is_gpu(slot):
    return slot in _GPU_SLOT_SET


def topo_id(topo):
    return "r-%s_%s" % (topo["router"].replace("slerm-", ""),
                        "-".join(h.replace("slerm-", "") for h in topo.get("rpc", [])) or "none")


def cpu_caps(topo):
    caps = dict(CPU_CAP_GB)
    if topo["router"] == "slerm-node05":
        # Router on deadlock owns the node (no separate worker job there), so it
        # may use all of the RAM the host report says is free for slerm.
        caps["slerm-node05"] = CPU_ONLY_NODE05_GB
    if topo.get("node05_cap_gb") is not None:
        caps["slerm-node05"] = min(caps["slerm-node05"], float(topo["node05_cap_gb"]))
    return caps


def _split_str(bytes_by_dev, devs):
    total = sum(bytes_by_dev.values()) or 1
    pct = [bytes_by_dev.get(d, 0) * 100.0 / total for d in devs]
    ints = [max(1, int(round(p))) if p > 0 else 0 for p in pct]
    return ",".join(str(i) for i in ints)


def plan_model(name, keys, facts, topo, host_notes):
    devs = slots(topo)
    size_gb = facts["bytes"] / 1e9
    moe = facts["moe"]
    caps = cpu_caps(topo)
    gpus = [d for d in devs if is_gpu(d)]
    res = {"model": name, "size_gb": round(size_gb, 1), "moe": moe, "keys": dict(keys),
           "tested": False, "feasible": False, "reason": ""}
    if keys.get("split-mode") == "tensor" and topo.get("rpc"):
        # 2026-10-07: the tensor-split/MTP presets need the GPU-only router (build-v050-p100); routers with RPC
        # workers run build-pascal, so such a model falls back to its plain layer-split, q8_0, ub 512 config there.
        for k in ("split-mode", "spec-type", "spec-draft-n-max", "cache-type-k", "cache-type-v", "batch-size", "ubatch-size", "slot-save-path"):
            res["keys"].pop(k, None)
    base = keys.get("tensor-split")

    def fail(reason):
        res["reason"] = reason
        return res

    # --- no GPU at all: CPU-only router on deadlock -------------------------
    if not gpus:
        toks = EFFECTIVE_GBPS / max(0.1, facts["active_bytes"] / 1e9)
        res["est_tok_s"] = round(toks, 1)
        if toks < MIN_CPU_TOKS:
            return fail("GPUs unavailable and too slow on CPU (~%.1f tok/s)" % toks)
        local = topo["router"]
        need = size_gb / CPU_MARGIN
        room = {d: caps.get(d, 0) for d in devs}
        room[local] = max(0.0, room[local] - CPU_ROUTER_RESERVE_GB)
        if sum(room.values()) < need:
            return fail("GPUs unavailable; needs %.0f GB RAM, %.0f GB free for slerm%s"
                        % (need, sum(room.values()), host_notes))
        alloc, left = {}, size_gb
        for d in [local] + [x for x in devs if x != local]:      # keep it local first
            take = min(left, room[d] * CPU_MARGIN)
            if take > 0:
                alloc[d] = take
                left -= take
        res["keys"]["tensor-split"] = _split_str(alloc, devs)
        ctx = int(keys.get("ctx-size", 8192))
        res["keys"]["ctx-size"] = str(min(ctx, CPU_ONLY_CTX))
        res.update(feasible=True, reason="CPU-only fallback (~%.0f tok/s)" % toks)
        return res

    # --- full pool: the tuned split, verbatim ----------------------------
    n = len(FULL_ORDER)
    if base is None:
        base = "0,0,0,1,1"          # no preset section: both P100s, no CPU nodes
    shares = [float(x) for x in base.split(",")] + [0.0] * n
    frac = dict(zip(FULL_ORDER, shares[:n]))
    tot = sum(frac.values()) or 1.0
    want = {d: frac[d] / tot * size_gb for d in FULL_ORDER if frac[d] > 0}
    if devs == FULL_ORDER:
        for d, gb in want.items():
            if not is_gpu(d) and gb > caps.get(d, 0) * CPU_MARGIN:
                return fail("%s has %.0f GB for slerm, split needs %.1f%s" % (d, caps.get(d, 0), gb, host_notes))
        res.update(feasible=True, tested=True, reason="tested split")
        return res

    alloc = {d: gb for d, gb in want.items() if d in devs}
    lost_gpu = sum(gb for d, gb in want.items() if is_gpu(d) and d not in devs)
    lost_cpu = sum(gb for d, gb in want.items() if not is_gpu(d) and d not in devs)

    # node04's GPUs come and go together (they are one VM), so a router on a GPU
    # node has every GPU slot and this only fires for a malformed topology.
    if lost_gpu > 0:
        return fail("needs the P100s in slerm-node04%s" % host_notes)

    # CPU devices must hold what they already had AND anything moved to them.
    cpu_devs = [d for d in devs if not is_gpu(d)]
    for d in cpu_devs:
        if alloc.get(d, 0) > caps.get(d, 0) * CPU_MARGIN:
            return fail("%s has %.0f GB for slerm, split needs %.1f%s"
                        % (d, caps.get(d, 0), alloc[d], host_notes))
    if lost_cpu > 0:
        room = {d: caps.get(d, 0) * CPU_MARGIN - alloc.get(d, 0) for d in cpu_devs}
        if sum(max(0, r) for r in room.values()) < lost_cpu:
            return fail("needs %.1f GB more CPU RAM than available%s" % (lost_cpu, host_notes))
        spare = sum(max(0, r) for r in room.values())
        for d in cpu_devs:
            alloc[d] = alloc.get(d, 0) + lost_cpu * max(0, room[d]) / spare
    if not any(alloc.get(g, 0) > 0 for g in gpus) and gpus:
        return fail("no GPU share left")
    res["keys"]["tensor-split"] = _split_str(alloc, devs)
    if lost_gpu == 0 and lost_cpu == 0 and not any(
            d not in devs and gb > 0 for d, gb in want.items()):
        # Only zero-share devices were dropped (e.g. a small model on the GPUs
        # alone): the same layers land on the same devices as the tested split.
        res.update(feasible=True, tested=True, reason="tested split (idle devices dropped)")
        return res
    res.update(feasible=True, reason="derived split (untested on this device set)")
    return res


def plan(topo, host_notes=""):
    top, star, sections = parse_ini()
    models = servable()
    out = {"topology": topo, "id": topo_id(topo), "devices": devices(topo), "slots": slots(topo),
           "generated": int(time.time()), "models": {}}
    for name, path in models.items():
        keys = sections.get(name, {})
        try:
            facts = gguf_facts(path)
        except (OSError, ValueError) as e:
            out["models"][name] = {"model": name, "feasible": False, "reason": "unreadable: %s" % e}
            continue
        out["models"][name] = plan_model(name, keys, facts, topo, host_notes)
    return out, top, star


def write_preset(p, top, star):
    """models-<id>.ini + allowed-<id>.txt for a router started on this topology."""
    os.makedirs(TOPO_DIR, exist_ok=True)
    tid = p["id"]
    lines = ["; GENERATED by slerm_topo.py from etc/models.ini -- do not edit.",
             "; Devices, one tensor-split value each (rpc..., local GPUs last): "
             + " ".join(p.get("slots") or p["devices"]), ""]
    lines += ["%s = %s" % kv for kv in top.items()]
    lines += ["", "[*]"] + ["%s = %s" % kv for kv in star.items()]
    allowed = []
    for name, m in sorted(p["models"].items()):
        if not m.get("feasible"):
            lines += ["", "; %s -- unavailable: %s" % (name, m.get("reason"))]
            continue
        allowed.append(name)
        lines += ["", "; " + m["reason"], "[%s]" % name]
        lines += ["%s = %s" % kv for kv in m["keys"].items()]
    ini = os.path.join(TOPO_DIR, "models-%s.ini" % tid)
    alw = os.path.join(TOPO_DIR, "allowed-%s.txt" % tid)
    for path, text in ((ini, "\n".join(lines) + "\n"), (alw, "\n".join(allowed) + "\n")):
        tmp = path + ".tmp"
        open(tmp, "w").write(text)
        os.replace(tmp, path)
    return ini, alw


def main():
    if len(sys.argv) >= 3 and sys.argv[1] in ("plan", "write"):
        topo = json.loads(sys.argv[2])
        p, top, star = plan(topo)
        if sys.argv[1] == "write":
            ini, alw = write_preset(p, top, star)
            print(ini)
            print(alw)
            return
        for name, m in sorted(p["models"].items()):
            print("%-28s %-5s %-24s %s" % (name, "ok" if m.get("feasible") else "--",
                                          m.get("keys", {}).get("tensor-split", ""), m.get("reason")))
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
