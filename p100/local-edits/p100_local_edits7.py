#!/usr/bin/env python3
"""Seventh round (brick 2): mmvh one column only (multi-token goes back to MMVQ, which handles it better), half-warp
mode only when it removes the idle step (K/256 % 4 == 0), IQ4_XS off by default, and the FFN SWIGLU gate/up fusion
through mmvh on GP100 (knob MMVH_FUSE, default on).
Usage: p100_local_edits7.py <llama.cpp source dir> <dir with mmvh-pascal.cu/.cuh>."""
import shutil, sys
from pathlib import Path

root = Path(sys.argv[1]) / "ggml/src/ggml-cuda"
for f in ("mmvh-pascal.cu", "mmvh-pascal.cuh"):
    shutil.copy(Path(sys.argv[2]) / f, root / f)
    print(f"ok  {f} copied")


def edit(name, old, new):
    p = root / name
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"{name}: expected 1 match, found {n}:\n{old[:300]}")
    p.write_text(s.replace(old, new))
    print(f"ok  {name}")


edit("p100-tune.cuh", "    int mmvh_iq4 = 1; // IQ4_XS through mmvh too\n",
     "    int mmvh_iq4 = 0; // IQ4_XS through mmvh too (measured slower than MMVQ, off)\n"
     "    int mmvh_fuse = 1; // FFN SWIGLU gate/up fusion through mmvh\n")
edit("ggml-cuda.cu", """    else if (strcmp(key, "MMVH_IQ4") == 0) { ggml_cuda_tune.mmvh_iq4 = atoi(val); }
""", """    else if (strcmp(key, "MMVH_IQ4") == 0) { ggml_cuda_tune.mmvh_iq4 = atoi(val); }
    else if (strcmp(key, "MMVH_FUSE") == 0) { ggml_cuda_tune.mmvh_fuse = atoi(val); }
""")
edit("ggml-cuda.cu", """            {"GGML_CUDA_MMVH_PASCAL", "MMVH"}, {"GGML_CUDA_MMVH_IQ4", "MMVH_IQ4"},
""", """            {"GGML_CUDA_MMVH_PASCAL", "MMVH"}, {"GGML_CUDA_MMVH_IQ4", "MMVH_IQ4"}, {"GGML_CUDA_MMVH_FUSE", "MMVH_FUSE"},
""")
edit("ggml-cuda.cu", """#include "ggml-cuda/p100-tune.cuh"
""", """#include "ggml-cuda/p100-tune.cuh"
#include "ggml-cuda/mmvh-pascal.cuh"
""")
edit("ggml-cuda.cu", """    // fusion is not universally faster on Pascal; GGML_CUDA_FUSION_PASCAL=1 allows it to measure on the P100s
    static const bool fusion_pascal = getenv("GGML_CUDA_FUSION_PASCAL") != nullptr && atoi(getenv("GGML_CUDA_FUSION_PASCAL"));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL && !fusion_pascal) {
        return false;
    }""", """    // fusion is not universally faster on Pascal; GGML_CUDA_FUSION_PASCAL=1 allows it to measure on the P100s, and on
    // GP100 the FFN gate/up pair is fused when mmvh will run it (one kernel, shared activations, SWIGLU in the epilogue)
    static const bool fusion_pascal = getenv("GGML_CUDA_FUSION_PASCAL") != nullptr && atoi(getenv("GGML_CUDA_FUSION_PASCAL"));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL && !fusion_pascal && !ggml_cuda_mmvh_pascal_fusable(tensor)) {
        return false;
    }""")
edit("mmvq.cu", """    if (ggml_cuda_mmvh_pascal_supported(src0, src1, ids, dst, fusion != nullptr)) {
        ggml_cuda_mmvh_pascal(ctx, src0, src1, dst);
        return;
    }""", """    if (ggml_cuda_mmvh_pascal_supported(src0, src1, ids, dst, fusion)) {
        ggml_cuda_mmvh_pascal(ctx, src0, src1, dst, fusion);
        return;
    }""")
