#!/usr/bin/env python3
"""Local P100 (sm_60) changes on top of official llama.cpp v0.5.0, written for slerm node04.

1. dp4a on sm_60 via PTX vmad (bit-exact integer math; GGML_CUDA_NO_VMAD_DP4A turns it off at build time).
2. MMVQ: 4 output rows per block for one-token mat-vec on pre-Volta NVIDIA (GGML_CUDA_MMVQ_PASCAL_RPB).
3. CUDA graphs on pre-Volta when GGML_CUDA_GRAPHS_PASCAL=1 (upstream always turns them off there).
4. mmvq fusion (gate/up/bias) on Pascal when GGML_CUDA_FUSION_PASCAL=1 (upstream always turns it off there).

Usage: p100_local_edits.py <llama.cpp source dir>. Each edit must match exactly once.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "ggml/src/ggml-cuda"


def edit(name, old, new):
    p = root / name
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"{name}: expected 1 match, found {n}:\n{old[:200]}")
    p.write_text(s.replace(old, new))
    print(f"ok  {name}")


# 1. dp4a without __dp4a
edit("common.cuh",
"""#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
""",
"""#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
#if defined(__CUDA_ARCH__) && !defined(GGML_CUDA_NO_VMAD_DP4A)
    // sm_60 (P100) has no __dp4a. vmad multiplies the byte lanes its selectors pick, sign-extended by the .s32
    // operand types, and adds the accumulator, so four of them skip the byte extraction the fallback below compiles to.
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\\n\\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\\n\\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\\n\\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(c) : "r"(a), "r"(b));
    return c;
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // defined(__CUDA_ARCH__) && !defined(GGML_CUDA_NO_VMAD_DP4A)
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
""")

# 2. MMVQ rows per block on pre-Volta NVIDIA
edit("mmvq.cu",
"""    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};
""",
"""    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    MMVQ_PARAMETERS_PASCAL,  // NVIDIA before Volta: GENERIC, except rows per block at ncols_dst == 1
};

// Output rows per block for one-token mat-vec on pre-Volta NVIDIA. With one row per block every row re-reads the
// q8_1 activations from L2, and on the P100 that load rather than DRAM bound decode (~30% of memory bandwidth).
#ifndef GGML_CUDA_MMVQ_PASCAL_RPB
#define GGML_CUDA_MMVQ_PASCAL_RPB 4
#endif
""")

edit("mmvq.cu",
"""#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}
""",
"""#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#elif defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    return MMVQ_PARAMETERS_PASCAL;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}
""")

edit("mmvq.cu",
"""    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
""",
"""    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_VOLTA) {
        return MMVQ_PARAMETERS_PASCAL;
    }
""")

edit("mmvq.cu",
"""static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
""",
"""static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_PASCAL) {
""")

edit("mmvq.cu",
"""static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
""",
"""static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst == 1) {
        return small_k && nwarps > GGML_CUDA_MMVQ_PASCAL_RPB ? nwarps : GGML_CUDA_MMVQ_PASCAL_RPB;
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL) {
        table_id = MMVQ_PARAMETERS_GENERIC;
    }
""")

# 3. CUDA graphs on pre-Volta, opt-in
edit("ggml-cuda.cu",
"""    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
""",
"""    // Upstream turns graphs off before Volta; GGML_CUDA_GRAPHS_PASCAL=1 keeps them on to measure that on the P100s.
    static const bool graphs_pascal = getenv("GGML_CUDA_GRAPHS_PASCAL") != nullptr && atoi(getenv("GGML_CUDA_GRAPHS_PASCAL"));
    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA && !graphs_pascal) {
""")

# 4. mmvq fusion on Pascal, opt-in
edit("ggml-cuda.cu",
"""    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
""",
"""    // fusion is not universally faster on Pascal; GGML_CUDA_FUSION_PASCAL=1 allows it to measure on the P100s
    static const bool fusion_pascal = getenv("GGML_CUDA_FUSION_PASCAL") != nullptr && atoi(getenv("GGML_CUDA_FUSION_PASCAL"));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL && !fusion_pascal) {
        return false;
    }
""")
