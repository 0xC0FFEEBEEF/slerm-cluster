#!/usr/bin/env python3
"""Pad Ollama's 3-element qwen35 rope sections to the 4 upstream llama.cpp wants.

Ollama writes `qwen35.rope.dimension_sections` with 3 values; upstream stores
`std::array<int,4> rope_sections` and rejects the file with

    key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3

llama-model.cpp tests s[0]||s[1]||s[2]||s[3] and llama-hparams.cpp validates
s[0]>0 && s[1]>0, so the fourth slot is simply unused for a 3-section mrope:
padding with 0 is what the format expects, not a guess.

Copies the file (tensors are streamed, not held in RAM) with the key rewritten.
"""
import os, sys, argparse
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

PAD_KEYS = ("rope.dimension_sections", "mrope_sections", "rope.mrope_section")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=4)
    a = ap.parse_args()

    r = GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.dst, arch, endianess=r.endianess)

    patched = []
    for name, f in r.fields.items():
        # GGUFReader.fields also exposes synthetic HEADER entries (GGUF.version,
        # GGUF.tensor_count, GGUF.kv_count). Writing those back out as metadata
        # produces a file whose header appears twice and no reader will open it.
        if name.startswith("GGUF."):
            continue
        if name == "general.architecture":
            continue
        val = f.contents()
        if any(name.endswith(k) for k in PAD_KEYS) and isinstance(val, list):
            if len(val) < a.width:
                val = list(val) + [0] * (a.width - len(val))
                patched.append("%s -> %s" % (name, val))
                w.add_key_value(name, val, GGUFValueType.ARRAY,
                                sub_type=GGUFValueType.INT32)
                continue
        t = f.types[0] if f.types else None
        if t == GGUFValueType.ARRAY and not val:
            # An empty array carries no element type, so it cannot be written
            # back out. It also carries no information -- e.g. this model has an
            # empty qwen35.vision.deepstack_visual_indexes, a vision key that is
            # meaningless for a text-only model. Drop it.
            print("  dropping empty array:", name)
            continue
        if t == GGUFValueType.ARRAY:
            w.add_key_value(name, val, GGUFValueType.ARRAY, sub_type=f.types[-1])
        else:
            w.add_key_value(name, val, t)

    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes,
                          raw_dtype=t.tensor_type)

    print("patched:", ", ".join(patched) if patched else "NOTHING (already 4?)")
    if not patched:
        sys.exit("refusing to write an identical copy")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data)
    w.close()
    print("wrote", a.dst, os.path.getsize(a.dst), "bytes")


if __name__ == "__main__":
    main()
