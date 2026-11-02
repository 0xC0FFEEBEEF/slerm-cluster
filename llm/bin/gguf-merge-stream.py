"""Merge a two-shard llama.cpp GGUF into one file, cloning the data instead of copying it where it can.

llama-gguf-split --merge reads each tensor whole into RAM (a 28.8 GB tensor = std::bad_alloc on a small
host) and copies every byte. This writes a new header and moves the data with copy_file_range, which on
XFS/btrfs with reflink CLONES whole 4 KiB blocks (seconds, no extra space) when source and destination
offsets agree mod 4096, and falls back to a normal copy otherwise.

Constraints (ggml/src/gguf.cpp): tensor data must be laid out in tensor-info order, each tensor at the
padded end of the one before it. llama.cpp finds tensors by name, so the order itself is free. Layout:

    header | A = shard-1 tensors before boundary b | T = the shard-2 tensor(s) | B = the rest of shard 1

The header carries one extra KV (slerm.merge_padding, spaces) sized so the data section starts at an
offset congruent to shard 1's (A clones). b is the LAST shard-1 tensor boundary at which T also lands
congruent to shard 2 (T clones); only B, after it, is copied. llama.cpp ignores unknown KVs.

usage: gguf_merge_stream.py OUT SHARD1 SHARD2 [--plan]
"""
import os
import struct
import sys

BLOCK = 4096
PAD_KEY = b"slerm.merge_padding"
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SPLIT_KEYS = {"split.no", "split.count", "split.tensors.count"}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def skip_val(f, t):
    if t in SCALAR:
        f.seek(SCALAR[t], 1)
    elif t == 8:
        f.seek(rd(f, "<Q"), 1)
    elif t == 9:
        et, n = rd(f, "<I"), rd(f, "<Q")
        if et in SCALAR:
            f.seek(SCALAR[et] * n, 1)
        else:
            for _ in range(n):
                skip_val(f, et)
    else:
        raise ValueError("unknown GGUF value type %d" % t)


def parse(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF: " + path)
        ver, nt, nkv = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
        if ver != 3:
            raise ValueError("GGUF v%d not handled" % ver)
        kvs, align = [], 32
        for _ in range(nkv):
            start = f.tell()
            key = f.read(rd(f, "<Q")).decode()
            t = rd(f, "<I")
            if key == "general.alignment":
                align = rd(f, "<I") if t == 4 else rd(f, "<Q")
            else:
                skip_val(f, t)
            end = f.tell()
            f.seek(start)
            kvs.append((key, f.read(end - start)))
        tensors = []
        for _ in range(nt):
            start = f.tell()
            name = f.read(rd(f, "<Q")).decode()
            nd = rd(f, "<I")
            f.seek(8 * nd + 4, 1)
            body_end = f.tell()
            off = rd(f, "<Q")
            f.seek(start)
            body = f.read(body_end - start)
            f.seek(8, 1)
            tensors.append((name, body, off))
        data_start = (f.tell() + align - 1) // align * align
    size = os.path.getsize(path)
    return {"kvs": kvs, "align": align, "tensors": tensors, "data_start": data_start, "size": size,
            "data_len": size - data_start}


def pad_to(x, a):
    return (x + a - 1) // a * a


def move(src_path, src_off, out_fd, dst_off, length, log):
    """copy_file_range in a loop; a plain write for the bytes before the first whole block."""
    cloned_guess = 0
    with open(src_path, "rb") as src:
        if src_off % BLOCK == dst_off % BLOCK:
            head = min(length, (BLOCK - src_off % BLOCK) % BLOCK)
            if head:
                src.seek(src_off)
                os.pwrite(out_fd, src.read(head), dst_off)
                src_off, dst_off, length = src_off + head, dst_off + head, length - head
            cloned_guess = length
        while length:
            n = os.copy_file_range(src.fileno(), out_fd, min(length, 1 << 30), src_off, dst_off)
            if n <= 0:
                sys.exit("copy_file_range returned %d with %d bytes left" % (n, length))
            src_off, dst_off, length = src_off + n, dst_off + n, length - n
    return cloned_guess


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    plan_only = "--plan" in sys.argv
    out, s1, s2 = args
    if os.path.exists(out) and not plan_only:
        sys.exit("refusing to overwrite " + out)
    p1, p2 = parse(s1), parse(s2)
    align = p1["align"]
    if p2["align"] != align or BLOCK % align:
        sys.exit("alignment problem: %d / %d" % (align, p2["align"]))

    # shard-1 tensors must already be contiguous in info order (they are, or llama.cpp would reject the shard)
    t1 = p1["tensors"]
    for (_, _, a), (_, _, b) in zip(t1, t1[1:]):
        if b <= a:
            sys.exit("shard 1 tensor infos are not in offset order")
    if t1[0][2] != 0:
        sys.exit("shard 1 data does not start at offset 0")
    end1 = pad_to(p1["data_len"], align)                   # padded end of shard 1's last tensor
    if [o for _, _, o in p2["tensors"]][0] != 0:
        sys.exit("shard 2 data does not start at offset 0")
    len2 = pad_to(p2["data_len"], align)                   # padded size of shard 2's tensor block

    kvs = [raw for key, raw in p1["kvs"] if key not in SPLIT_KEYS]
    ntensors = len(t1) + len(p2["tensors"])
    info_len = sum(len(body) + 8 for _, body, _ in t1 + p2["tensors"])
    fixed = 4 + 20 + sum(len(r) for r in kvs) + info_len + (8 + len(PAD_KEY) + 4 + 8)
    data_start = fixed + (p1["data_start"] - fixed) % BLOCK  # == shard 1's data start mod BLOCK, so A clones
    pad = data_start - fixed

    # last boundary b (a tensor start, or the end) where T lands congruent to shard 2
    bounds = [o for _, _, o in t1] + [end1]
    good = [i for i, b in enumerate(bounds) if (data_start + b - p2["data_start"]) % BLOCK == 0]
    k = good[-1] if good else len(t1)                      # no match: T after everything, T is copied
    b = bounds[k]
    tail = end1 - b
    print("data_start %d (shard1 %d), split before tensor %d of %d at %.2f GB; copied tail %.2f GB; "
          "T clones: %s" % (data_start, p1["data_start"], k, len(t1), b / 1e9, tail / 1e9, bool(good)), flush=True)
    if plan_only:
        return

    infos = []
    for _, body, o in t1[:k]:
        infos.append(body + struct.pack("<Q", o))
    for _, body, o in p2["tensors"]:
        infos.append(body + struct.pack("<Q", b + o))
    for _, body, o in t1[k:]:
        infos.append(body + struct.pack("<Q", o + len2))

    with open(out + ".tmp", "wb") as o:
        o.write(b"GGUF" + struct.pack("<IQQ", 3, ntensors, len(kvs) + 1))
        for raw in kvs:
            o.write(raw)
        o.write(struct.pack("<Q", len(PAD_KEY)) + PAD_KEY + struct.pack("<IQ", 8, pad) + b" " * pad)
        for raw in infos:
            o.write(raw)
        if o.tell() != data_start:
            sys.exit("header is %d bytes, planned %d" % (o.tell(), data_start))
        o.flush()
        fd = o.fileno()
        move(s1, p1["data_start"], fd, data_start, b, None)
        print("A: %.2f GB" % (b / 1e9), flush=True)
        move(s2, p2["data_start"], fd, data_start + b, p2["data_len"], None)
        print("T: %.2f GB" % (p2["data_len"] / 1e9), flush=True)
        if tail:
            move(s1, p1["data_start"] + b, fd, data_start + b + len2, p1["data_len"] - b, None)
        print("B: %.2f GB" % ((p1["data_len"] - b) / 1e9), flush=True)
        o.truncate(data_start + b + len2 + max(0, p1["data_len"] - b))
    os.replace(out + ".tmp", out)
    print("wrote %s: %d tensors, %d KVs, %d bytes" % (out, ntensors, len(kvs) + 1, os.path.getsize(out)))


if __name__ == "__main__":
    main()
