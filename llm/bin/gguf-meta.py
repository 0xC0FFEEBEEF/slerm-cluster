#!/usr/bin/env python3
"""Read GGUF header + KV metadata only (no tensor data, no model load).

Cheap way to answer "why won't llama.cpp load this?" -- the known failure modes
on this cluster are metadata-shaped: a wrong general.architecture string, or
rope.dimension_sections with the wrong element count.
"""
import struct
import sys

FMT = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
       5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
       12: ("<d", 8)}


def read(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None, "bad magic"
        ver = struct.unpack("<I", f.read(4))[0]
        ntensor = struct.unpack("<Q", f.read(8))[0]
        nkv = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                vals = [rval(et) for _ in range(min(n, 12))]
                return {"n": n, "head": vals}
            fmt, sz = FMT[t]
            return struct.unpack(fmt, f.read(sz))[0]

        kv = {}
        for _ in range(nkv):
            k = rstr()
            t = struct.unpack("<I", f.read(4))[0]
            kv[k] = rval(t)
        return (ver, ntensor, kv), None


for path in sys.argv[1:]:
    try:
        res, err = read(path)
    except Exception as e:
        print("== %s\n   ERROR: %s" % (path.split("/")[-1], e))
        continue
    if err:
        print("== %s\n   ERROR: %s" % (path.split("/")[-1], err))
        continue
    ver, nt, kv = res
    arch = kv.get("general.architecture")
    print("== %s" % path.split("/")[-1])
    print("   gguf v%d, %d tensors" % (ver, nt))
    print("   general.architecture = %r" % arch)
    for k in ("general.name", "general.size_label", "general.file_type",
              "general.basename"):
        if k in kv:
            print("   %s = %s" % (k, kv[k]))
    # rope.dimension_sections is the known Qwen3.5 trap: upstream wants 4 elems
    for k in sorted(kv):
        if "dimension_sections" in k or "rope" in k.lower():
            v = kv[k]
            if isinstance(v, dict):
                print("   %s = n=%d head=%s" % (k, v["n"], v["head"]))
            else:
                print("   %s = %s" % (k, v))
    akeys = sorted(k for k in kv if arch and k.startswith(arch + "."))
    print("   %d arch keys: %s" % (len(akeys), ", ".join(a.split(".", 1)[1] for a in akeys[:12])))
    print()
