#!/usr/bin/env python3
"""Which GGUF tensors an RPC worker will receive, so staging can pre-fill its
ggml-rpc-server --cache directory before the model is ever loaded.

llama.cpp (src/llama-model.cpp, load_tensors) puts repeating layer IL on device
upper_bound(cumulative normalised tensor-split, float(IL) / (n_layer + 1)) --
float32 arithmetic, n_gpu_layers left unset so every layer is offloaded -- and
the output layer (output.weight / output_norm, or token_embd duplicated when
the embeddings are tied) on the device of layer n_layer. token_embd itself
always stays on the router's CPU. Device order is the --rpc list, then the
router's local GPU last, the same as the tensor-split values.

A worker is sent each of its tensors whole (one set_tensor per tensor), and
ggml-rpc-server caches only tensors over 10 MB, so those are the ones staged.

    model_stage.py tensors MODEL.gguf SPLIT DEV_INDEX    (debug: list them)

Python 3.9 (the nodes' version).
"""
import importlib.util
import os
import re
import struct
import sys

HASH_THRESHOLD = 10 * 1024 * 1024
_CONST = "/data/llm/src/llama.cpp/gguf-py/gguf/constants.py"
_sizes = None


def quant_sizes():
    """{ggml type id: (block elements, block bytes)} from llama.cpp's gguf-py."""
    global _sizes
    if _sizes is None:
        spec = importlib.util.spec_from_file_location("gguf_constants", _CONST)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _sizes = {int(k): v for k, v in mod.GGML_QUANT_SIZES.items()}
    return _sizes


_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _str(f):
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n)


def _val(f, t):
    if t in _SCALAR:
        b = f.read(_SCALAR[t])
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, b)[0]
    if t == 8:
        return _str(f).decode("utf-8", "replace")
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et in _SCALAR:
            f.seek(_SCALAR[et] * n, 1)
        else:
            for _ in range(n):
                _val(f, et)
        return None
    raise ValueError("bad GGUF value type %d" % t)


def read_gguf(path):
    """-> (block_count, [(name, absolute offset, nbytes)])"""
    qs = quant_sizes()
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF: " + path)
        _ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
        align, blocks = 32, None
        for _ in range(n_kv):
            k = _str(f).decode("utf-8", "replace")
            t, = struct.unpack("<I", f.read(4))
            v = _val(f, t)
            if k == "general.alignment":
                align = v
            elif k.endswith(".block_count") and blocks is None:
                blocks = v
        infos = []
        for _ in range(n_t):
            name = _str(f).decode("utf-8", "replace")
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            typ, off = struct.unpack("<IQ", f.read(12))
            be, bb = qs[typ]
            n = 1
            for d in dims:
                n *= d
            infos.append((name, off, n // be * bb))
        base = f.tell()
    base = (base + align - 1) // align * align
    return blocks, [(n, base + o, s) for n, o, s in infos]


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def layer_devices(n_layer, split):
    """Device index of layers 0..n_layer (n_layer = the output layer)."""
    cum, s = [], _f32(0.0)
    for v in split:
        s = _f32(s + _f32(float(v)))
        cum.append(s)
    cum = [_f32(c / s) for c in cum]
    act = n_layer + 1
    out = []
    for il in range(n_layer + 1):
        x = _f32(_f32(float(il)) / act)
        out.append(next((i for i, c in enumerate(cum) if c > x), len(cum) - 1))
    return out


_BLK = re.compile(r"^blk\.(\d+)\.")


def worker_tensors(path, split, dev_index):
    """[(name, offset, nbytes)] that device DEV_INDEX receives and caches."""
    n_layer, tensors = read_gguf(path)
    devs = layer_devices(n_layer, split)
    names = set(t[0] for t in tensors)
    tied = "output.weight" not in names
    out = []
    for name, off, size in tensors:
        m = _BLK.match(name)
        if m:
            il = int(m.group(1))
            mine = il < n_layer and devs[il] == dev_index
        elif name.startswith("output") or (tied and name == "token_embd.weight"):
            mine = devs[n_layer] == dev_index
        else:
            mine = False
        if mine and size > HASH_THRESHOLD:
            out.append((name, off, size))
    return out


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "tensors":
        split = [float(x) for x in sys.argv[3].split(",")]
        ts = worker_tensors(os.path.realpath(sys.argv[2]), split, int(sys.argv[4]))
        for t in ts:
            print("%s %d %d" % t)
        print("# %d tensors, %.2f GB" % (len(ts), sum(t[2] for t in ts) / 1e9), file=sys.stderr)
    else:
        sys.exit(__doc__)
