"""Model facts read from the GGUF file itself.

The router fills in a model's `meta` (size, parameter count, quant) only while
that model is resident, so on the AI tab every unloaded model showed "-". The
same facts are in the file's header on NFS. Reading all 17 cold takes ~9 s
(tokenizer arrays have to be walked), so `lookup()` never blocks a request:
it answers from the cache and queues a miss for a background thread.
"""
import os
import re
import struct
import threading

# llama.h llama_ftype -> llama_ftype_name(), the same strings the router's
# meta.ftype carries for a loaded model.
_FTYPES = {
    0: "all F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K - Medium", 11: "Q3_K - Small", 12: "Q3_K - Medium", 13: "Q3_K - Large",
    14: "Q4_K - Small", 15: "Q4_K - Medium", 16: "Q5_K - Small", 17: "Q5_K - Medium",
    18: "Q6_K", 19: "IQ2_XXS - 2.0625 bpw", 20: "IQ2_XS - 2.3125 bpw", 21: "Q2_K - Small",
    22: "IQ3_XS - 3.3 bpw", 23: "IQ3_XXS - 3.0625 bpw", 24: "IQ1_S - 1.5625 bpw",
    25: "IQ4_NL - 4.5 bpw", 26: "IQ3_S - 3.4375 bpw", 27: "IQ3_S mix - 3.66 bpw",
    28: "IQ2_S - 2.5 bpw", 29: "IQ2_M - 2.7 bpw", 30: "IQ4_XS - 4.25 bpw",
    31: "IQ1_M - 1.75 bpw", 32: "BF16", 36: "TQ1_0 - 1.69 bpw ternary",
    37: "TQ2_0 - 2.06 bpw ternary", 38: "MXFP4 MoE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_gguf_cache = {}


def _shards(path):
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(2))
    return ["%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(2)) for i in range(1, n + 1)]


def _read_header(path):
    """(file_type, n_params) from one GGUF file's header, or raises."""
    with open(path, "rb") as f:
        rd = f.read
        if rd(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, = struct.unpack("<I", rd(4))
        if version < 2:
            raise ValueError("GGUF v%d not supported" % version)
        n_tensors, n_kv = struct.unpack("<QQ", rd(16))
        ftype = None

        def skip_value(t):
            if t in _FIXED:
                f.seek(_FIXED[t], 1)
            elif t == 8:
                n, = struct.unpack("<Q", rd(8)); f.seek(n, 1)
            elif t == 9:
                it, cnt = struct.unpack("<IQ", rd(12))
                if it in _FIXED:
                    f.seek(_FIXED[it] * cnt, 1)
                else:
                    for _ in range(cnt):
                        skip_value(it)
            else:
                raise ValueError("unknown GGUF type %d" % t)

        for _ in range(n_kv):
            klen, = struct.unpack("<Q", rd(8))
            key = rd(klen)
            t, = struct.unpack("<I", rd(4))
            if key == b"general.file_type" and t in (4, 5):
                ftype, = struct.unpack("<I" if t == 4 else "<i", rd(4))
            else:
                skip_value(t)
        params = 0
        for _ in range(n_tensors):
            nlen, = struct.unpack("<Q", rd(8)); f.seek(nlen, 1)
            nd, = struct.unpack("<I", rd(4))
            dims = struct.unpack("<%dQ" % nd, rd(8 * nd))
            f.seek(12, 1)                       # type u32 + offset u64
            count = 1
            for d in dims:
                count *= d
            params += count
        return ftype, params


def gguf_meta(path):
    """Size, parameter count and quant of a model on disk, for a model the
    router has not loaded (its meta is only filled in while resident).
    Cached by (path, mtime, size); None if the file cannot be read."""
    if not path:
        return None
    try:
        files = [os.path.realpath(p) for p in _shards(path)]
        stats = [os.stat(p) for p in files]
    except OSError:
        return None
    key = tuple((p, s.st_mtime, s.st_size) for p, s in zip(files, stats))
    if key in _gguf_cache:
        return _gguf_cache[key]
    try:
        ftype, params = None, 0
        for i, p in enumerate(files):
            ft, n = _read_header(p)
            params += n
            if i == 0:
                ftype = ft
        out = {"size": sum(s.st_size for s in stats), "n_params": params,
               "ftype": _FTYPES.get(ftype, "unknown") if ftype is not None else None}
    except Exception:                                           # noqa: BLE001
        out = None
    _gguf_cache[key] = out
    return out


_pending = set()
_pending_lock = threading.Lock()
_worker = {"thread": None}


def _drain():
    while True:
        with _pending_lock:
            if not _pending:
                _worker["thread"] = None
                return
            path = _pending.pop()
        gguf_meta(path)


def lookup(path):
    """Cached facts for `path`, or None now and queued for the next poll."""
    if not path:
        return None
    try:
        key = tuple((p, s.st_mtime, s.st_size) for p, s in
                    ((q, os.stat(q)) for q in
                     (os.path.realpath(x) for x in _shards(path))))
    except OSError:
        return None
    if key in _gguf_cache:
        return _gguf_cache[key]
    with _pending_lock:
        _pending.add(path)
        if _worker["thread"] is None:
            t = threading.Thread(target=_drain, name="gguf-meta", daemon=True)
            _worker["thread"] = t
            t.start()
    return None
