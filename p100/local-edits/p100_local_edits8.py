#!/usr/bin/env python3
"""Eighth round (brick 3): mmvh multi-column path (2-4 tokens, Q4_K/Q5_K/Q6_K) for MTP verify, knob MMVH_MC (default on).
Usage: p100_local_edits8.py <llama.cpp source dir> <dir with mmvh-pascal.cu>."""
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1]) / "ggml/src/ggml-cuda"
shutil.copy(Path(sys.argv[2]) / "mmvh-pascal.cu", root / "mmvh-pascal.cu"); print("ok  mmvh-pascal.cu copied")
def edit(name, old, new):
    p = root / name; s = p.read_text(); n = s.count(old)
    if n != 1:
        sys.exit(f"{name}: expected 1 match, found {n}:\n{old[:300]}")
    p.write_text(s.replace(old, new)); print(f"ok  {name}")
edit("p100-tune.cuh", "    int mmvh_fuse = 1; // FFN SWIGLU gate/up fusion through mmvh\n",
     "    int mmvh_fuse = 1; // FFN SWIGLU gate/up fusion through mmvh\n    int mmvh_mc   = 1; // 2-4 token batches (MTP verify) through mmvh\n")
edit("ggml-cuda.cu", """    else if (strcmp(key, "MMVH_FUSE") == 0) { ggml_cuda_tune.mmvh_fuse = atoi(val); }
""", """    else if (strcmp(key, "MMVH_FUSE") == 0) { ggml_cuda_tune.mmvh_fuse = atoi(val); }
    else if (strcmp(key, "MMVH_MC")   == 0) { ggml_cuda_tune.mmvh_mc   = atoi(val); }
""")
edit("ggml-cuda.cu", """{"GGML_CUDA_MMVH_FUSE", "MMVH_FUSE"},
""", """{"GGML_CUDA_MMVH_FUSE", "MMVH_FUSE"},
            {"GGML_CUDA_MMVH_MC", "MMVH_MC"},
""")
