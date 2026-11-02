#!/usr/bin/env python3
"""Fifth round: HFMA2 mat-vec for Q4_K/Q5_K on GP100 (mmvh-pascal.cu/.cuh copied in beside this), knob MMVH.
Usage: p100_local_edits5.py <llama.cpp source dir> <dir with mmvh-pascal.cu/.cuh>."""
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


edit("p100-tune.cuh", """// mmvq_s/mmvq_l: one-token mat-vec presets for K <= 6144 / longer K.
struct ggml_cuda_tune_knobs {
    int ar      = 0;
""", """// mmvq_s/mmvq_l: one-token mat-vec presets for K <= 6144 / longer K; mmvh: Q4_K/Q5_K HFMA2 mat-vec on GP100
// (0 off, else rows per warp 1/2/4).
struct ggml_cuda_tune_knobs {
    int mmvh    = 0;
    int ar      = 0;
""")

edit("ggml-cuda.cu", """    else if (strcmp(key, "GRAPHS")  == 0) { ggml_cuda_tune.graphs  = atoi(val); }
""", """    else if (strcmp(key, "GRAPHS")  == 0) { ggml_cuda_tune.graphs  = atoi(val); }
    else if (strcmp(key, "MMVH")    == 0) { ggml_cuda_tune.mmvh    = atoi(val); }
""")

edit("ggml-cuda.cu", """            {"GGML_CUDA_GRAPHS_PASCAL", "GRAPHS"}, {"GGML_CUDA_MMVQ_PASCAL_CFG", "MMVQ"},
""", """            {"GGML_CUDA_GRAPHS_PASCAL", "GRAPHS"}, {"GGML_CUDA_MMVQ_PASCAL_CFG", "MMVQ"},
            {"GGML_CUDA_MMVH_PASCAL", "MMVH"},
""")

edit("ggml-cuda.cu", """    GGML_LOG_INFO("%s: AR=%d AR_BF16=%d GRAPHS=%d MMVQ=%d,%d\\n", __func__, ggml_cuda_tune.ar, ggml_cuda_tune.ar_bf16,
                  ggml_cuda_tune.graphs, ggml_cuda_tune.mmvq_s, ggml_cuda_tune.mmvq_l);
""", """    GGML_LOG_INFO("%s: AR=%d AR_BF16=%d GRAPHS=%d MMVQ=%d,%d MMVH=%d\\n", __func__, ggml_cuda_tune.ar, ggml_cuda_tune.ar_bf16,
                  ggml_cuda_tune.graphs, ggml_cuda_tune.mmvq_s, ggml_cuda_tune.mmvq_l, ggml_cuda_tune.mmvh);
""")

edit("mmvq.cu", """#include "mmvq.cuh"
#include "p100-tune.cuh"
""", """#include "mmvq.cuh"
#include "p100-tune.cuh"
#include "mmvh-pascal.cuh"
""")

edit("mmvq.cu", """    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
""", """    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    // GP100: Q4_K/Q5_K one-token mat-vec on HFMA2 instead of emulated dp4a, behind the MMVH knob
    if (ggml_cuda_mmvh_pascal_supported(src0, src1, ids, dst, fusion != nullptr)) {
        ggml_cuda_mmvh_pascal(ctx, src0, src1, dst);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
""")
