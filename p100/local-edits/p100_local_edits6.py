#!/usr/bin/env python3
"""Sixth round (brick 1): mmvh v2 (1-4 columns, IQ4_XS, half-warp mode, prepared-activation reuse), a per-device node
counter for that reuse, and the F32 mat-vec cap for pre-Turing NVIDIA raised from 3 to 8 columns (no tensor cores, so
4-column MTP verify batches stop falling into cuBLAS SGEMM).
Usage: p100_local_edits6.py <llama.cpp source dir> <dir with mmvh-pascal.cu>."""
import shutil, sys
from pathlib import Path

root = Path(sys.argv[1]) / "ggml/src/ggml-cuda"
shutil.copy(Path(sys.argv[2]) / "mmvh-pascal.cu", root / "mmvh-pascal.cu")
print("ok  mmvh-pascal.cu copied")


def edit(name, old, new):
    p = root / name
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"{name}: expected 1 match, found {n}:\n{old[:300]}")
    p.write_text(s.replace(old, new))
    print(f"ok  {name}")


edit("p100-tune.cuh", """extern ggml_cuda_tune_knobs ggml_cuda_tune;
""", """extern ggml_cuda_tune_knobs ggml_cuda_tune;
// Bumped for every node a CUDA backend computes (fused or not), per device: lets mmvh reuse prepared activations only
// when nothing ran in between.
extern uint64_t ggml_cuda_tune_node_seq[GGML_CUDA_MAX_DEVICES];
""")

edit("ggml-cuda.cu", """ggml_cuda_tune_knobs ggml_cuda_tune;
""", """ggml_cuda_tune_knobs ggml_cuda_tune;
uint64_t ggml_cuda_tune_node_seq[GGML_CUDA_MAX_DEVICES] = {};
""")

edit("ggml-cuda.cu", """                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
""", """                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                ggml_cuda_tune_node_seq[cuda_ctx->device]++;

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
""")

edit("mmvf.cu", """                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {""", """                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                // pre-Turing has no matrix cores, like the other branches that return 8; at 3 the 4-column MTP
                // verify batches fell into cuBLAS SGEMM on the P100s
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {""")
