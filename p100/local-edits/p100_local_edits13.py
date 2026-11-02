#!/usr/bin/env python3
"""Brick F1 (2026-10-17, Qwen3.8-Flash-Next decode on the P100s). Applies to /data/llm/src/llama.cpp-v050. Idempotent.

- MMVQ_ID knob (env GGML_CUDA_MMVQ_ID_PASCAL): Pascal (warps, rows per block) preset for one-token MUL_MAT_ID of the
  types Flash-Next keeps on the cards (IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0). Expert rows are short (K 2560 for
  gate/up, 640 for down) and the default 4-warp block left 37-69% of its lanes idle: 50-100 GB/s measured live.
  Presets 8-10 are new (1x8, 1x2, 2x2). 0 = upstream behaviour.
- MMVF_SHORTK knob (env GGML_CUDA_MMVF_SHORTK_PASCAL = rows per warp, 0 off): one-token F32/F16/BF16 mat-vec for K <= 1024
  on pre-Volta with one warp per R rows. The hyper-connection up projections (K 320, 10240 rows, BF16) got 10240 blocks
  of 160 threads, one FMA each plus a shared-memory reduction: 72 us for 6.25 MiB.
- Profiler: GET_ROWS keys carry src0 type, ne0, ne1 (power-of-two bucket) and the number of gathered rows.
- test-backend-ops: Flash-Next one-token shapes in perf mode, smaller copies of them in eval mode.
"""
import shutil
import sys

SRC = "/data/llm/src/llama.cpp-v050"
STAGE = "/data/llm/src/p100-local-files"
MARK = "2026-10-17"


def edit(rel, pairs, marker=MARK):
    path = f"{SRC}/{rel}"
    s = open(path).read()
    if marker in s and all(new in s for _, new in pairs):
        print(f"{rel}: already applied")
        return
    for old, new in pairs:
        if new in s:
            continue
        n = s.count(old)
        if n != 1:
            sys.exit(f"{rel}: anchor found {n} times:\n{old[:200]}")
        s = s.replace(old, new)
    open(path, "w").write(s)
    print(f"{rel}: edited")


# ---- knobs ------------------------------------------------------------------------------------------------------------
edit("ggml/src/ggml-cuda/p100-tune.cuh", [(
    "    int fa_gqa_pad = 0; // FA tile kernel: pack 4 or 8 Q heads per K/V head when no power of two divides the GQA ratio\n",
    "    int fa_gqa_pad = 0; // FA tile kernel: pack 4 or 8 Q heads per K/V head when no power of two divides the GQA ratio\n"
    "    int mmvq_id     = 0; // 2026-10-17: Pascal preset (1..10) for one-token MUL_MAT_ID of the IQ/Q2_0 expert types\n"
    "    int mmvf_shortk = 0; // 2026-10-17: rows per warp (1/2/4/8) for one-token F32/F16/BF16 mat-vec with K <= 1024\n",
)])

edit("ggml/src/ggml-cuda/ggml-cuda.cu", [
    ("    else if (strcmp(key, \"MMVH_MC_Q6\")  == 0) { ggml_cuda_tune.mmvh_mc_q6  = atoi(val); }\n",
     "    else if (strcmp(key, \"MMVH_MC_Q6\")  == 0) { ggml_cuda_tune.mmvh_mc_q6  = atoi(val); }\n"
     "    // 2026-10-17 (Flash-Next)\n"
     "    else if (strcmp(key, \"MMVQ_ID\")     == 0) { ggml_cuda_tune.mmvq_id     = atoi(val); }\n"
     "    else if (strcmp(key, \"MMVF_SHORTK\") == 0) { ggml_cuda_tune.mmvf_shortk = atoi(val); }\n"),
    ("            {\"GGML_CUDA_MMVH_MC_Q6\", \"MMVH_MC_Q6\"},\n",
     "            {\"GGML_CUDA_MMVH_MC_Q6\", \"MMVH_MC_Q6\"},\n"
     "            {\"GGML_CUDA_MMVQ_ID_PASCAL\", \"MMVQ_ID\"}, {\"GGML_CUDA_MMVF_SHORTK_PASCAL\", \"MMVF_SHORTK\"}, // 2026-10-17\n"),
])

# ---- MMVQ presets for one-token MUL_MAT_ID ----------------------------------------------------------------------------
edit("ggml/src/ggml-cuda/mmvq.cu", [
    ("static constexpr __host__ __device__ int mmvq_pascal_cfg_nwarps(int c) {\n"
     "    return c == 1 ? 4 : c == 2 ? 4 : c == 3 ? 2 : c == 4 ? 2 : c == 5 ? 8 : c == 6 ? 8 : c == 7 ? 1 : 0;\n"
     "}\n"
     "static constexpr __host__ __device__ int mmvq_pascal_cfg_rpb(int c) {\n"
     "    return c == 1 ? 2 : c == 2 ? 8 : c == 3 ? 4 : c == 4 ? 8 : c == 5 ? 2 : c == 6 ? 4 : c == 7 ? 4 : 0;\n"
     "}\n",
     "// 8..10 added 2026-10-17 for the short expert rows of Flash-Next (MMVQ_ID): 1 warp x 8 rows, 1 x 2, 2 x 2.\n"
     "static constexpr __host__ __device__ int mmvq_pascal_cfg_nwarps(int c) {\n"
     "    return c == 1 ? 4 : c == 2 ? 4 : c == 3 ? 2 : c == 4 ? 2 : c == 5 ? 8 : c == 6 ? 8 : c == 7 ? 1 :\n"
     "           c == 8 ? 1 : c == 9 ? 1 : c == 10 ? 2 : 0;\n"
     "}\n"
     "static constexpr __host__ __device__ int mmvq_pascal_cfg_rpb(int c) {\n"
     "    return c == 1 ? 2 : c == 2 ? 8 : c == 3 ? 4 : c == 4 ? 8 : c == 5 ? 2 : c == 6 ? 4 : c == 7 ? 4 :\n"
     "           c == 8 ? 8 : c == 9 ? 2 : c == 10 ? 2 : 0;\n"
     "}\n"
     "// Types whose one-token MUL_MAT_ID takes a preset through MMVQ_ID (2026-10-17, Flash-Next's expert types).\n"
     "static constexpr bool mmvq_pascal_tunable_id(ggml_type type) {\n"
     "    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL ||\n"
     "           type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q2_0;\n"
     "}\n"),
    ("            using pcfg0 = std::integral_constant<int, 0>;\n"
     "            if (should_use_small_k(c_ncols_dst)) {\n",
     "            using pcfg0 = std::integral_constant<int, 0>;\n"
     "            // P100 work (2026-10-17, Flash-Next): MMVQ_ID picks a preset for one-token MUL_MAT_ID. Expert rows of K 640 /\n"
     "            // 2560 left most of the default 4-warp block idle, and the IQ grid types never get small_k on Pascal.\n"
     "            if constexpr (mmvq_pascal_tunable_id(type)) {\n"
     "                const int pcfg_id = table_id == MMVQ_PARAMETERS_PASCAL && has_ids ? ggml_cuda_tune.mmvq_id : 0;\n"
     "                switch (pcfg_id) {\n"
     "                    case 1:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 1>{});  return;\n"
     "                    case 2:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 2>{});  return;\n"
     "                    case 3:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 3>{});  return;\n"
     "                    case 4:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 4>{});  return;\n"
     "                    case 5:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 5>{});  return;\n"
     "                    case 6:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 6>{});  return;\n"
     "                    case 7:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 7>{});  return;\n"
     "                    case 8:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 8>{});  return;\n"
     "                    case 9:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 9>{});  return;\n"
     "                    case 10: launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 10>{}); return;\n"
     "                    default: break;\n"
     "                }\n"
     "            }\n"
     "            if (should_use_small_k(c_ncols_dst)) {\n"),
])

# ---- short-K float mat-vec ----------------------------------------------------------------------------------------------
SHORTK = r'''
// P100 work (2026-10-17, Flash-Next): one-token mat-vec for short rows on pre-Volta (MMVF_SHORTK = rows per warp).
// The generic kernel gives every row its own block sized to K; at K = 320 (hyper-connection up projections, 10240
// rows, BF16) that is 10240 blocks of 160 threads doing one FMA each plus a shared-memory reduction, 72 us for 6.25 MiB.
// Here a warp owns R rows, its lanes stride over K in pairs and reduce with shuffles. F32 accumulation throughout.
template <typename T>
static __device__ __forceinline__ float2 mmvf_shortk_pair(const T * row, const int c) {
    if constexpr (std::is_same_v<T, float>) {
        return ((const float2 *) row)[c];
    } else if constexpr (std::is_same_v<T, half>) {
        return __half22float2(((const half2 *) row)[c]);
    } else {
        const nv_bfloat162 v = ((const nv_bfloat162 *) row)[c];
        return make_float2(ggml_cuda_cast<float>(v.x), ggml_cuda_cast<float>(v.y));
    }
}

template <typename T, int R>
static __global__ void __launch_bounds__(4*WARP_SIZE) mmvf_shortk_kernel(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols2, const int nrows, const int64_t stride_row) {
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*blockDim.y + threadIdx.y)*R;
    if (row0 >= nrows) {
        return;
    }
    const float2 * y2 = (const float2 *) y;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sum[r] = 0.0f;
    }
    for (int c = lane; c < ncols2; c += WARP_SIZE) {
        const float2 yv = y2[c];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r < nrows) {
                const float2 xv = mmvf_shortk_pair<T>(x + (row0 + r)*stride_row, c);
                sum[r] = fmaf(xv.x, yv.x, sum[r]);
                sum[r] = fmaf(xv.y, yv.y, sum[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sum[r] = warp_reduce_sum(sum[r]);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r < nrows) {
                dst[row0 + r] = sum[r];
            }
        }
    }
}

template <typename T>
static void mmvf_shortk_cuda(const T * x, const float * y, float * dst, const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int rows_per_warp, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const int  ncols2 = ncols/2;
    switch (rows_per_warp) {
        case 1: mmvf_shortk_kernel<T, 1><<<(nrows + 3)/4,  block_dims, 0, stream>>>(x, y, dst, ncols2, nrows, stride_row); break;
        case 2: mmvf_shortk_kernel<T, 2><<<(nrows + 7)/8,  block_dims, 0, stream>>>(x, y, dst, ncols2, nrows, stride_row); break;
        case 8: mmvf_shortk_kernel<T, 8><<<(nrows + 31)/32, block_dims, 0, stream>>>(x, y, dst, ncols2, nrows, stride_row); break;
        default: mmvf_shortk_kernel<T, 4><<<(nrows + 15)/16, block_dims, 0, stream>>>(x, y, dst, ncols2, nrows, stride_row); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
static void mul_mat_vec_f_cuda(
'''

edit("ggml/src/ggml-cuda/mmvf.cu", [
    ('#include "convert.cuh"\n', '#include "convert.cuh"\n#include "p100-tune.cuh"\n'),
    ("\ntemplate<typename T>\nstatic void mul_mat_vec_f_cuda(\n", SHORTK),
    ("        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {\n\n"
     "    if constexpr(std::is_same_v<T, half>) {\n",
     "        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {\n\n"
     "    // P100 work (2026-10-17): short-row one-token path, see mmvf_shortk_kernel.\n"
     "    if (ggml_cuda_tune.mmvf_shortk > 0 && ids == nullptr && ncols_dst == 1 && ncols <= 1024 &&\n"
     "            fusion.gate == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&\n"
     "            nchannels_x == 1 && nchannels_dst == 1 && nsamples_x == 1 && nsamples_dst == 1 &&\n"
     "            ggml_cuda_info().devices[ggml_cuda_get_device()].cc < GGML_CUDA_CC_VOLTA) {\n"
     "        mmvf_shortk_cuda(x, y, dst, ncols, nrows, stride_row, ggml_cuda_tune.mmvf_shortk, stream);\n"
     "        return;\n"
     "    }\n\n"
     "    if constexpr(std::is_same_v<T, half>) {\n"),
])

# ---- profiler: GET_ROWS shapes ------------------------------------------------------------------------------------------
PROF_OLD = ("        k += \" n=\" + std::to_string(node->src[1]->ne[1]);\n"
            "    } else {\n")
PROF_NEW = ("        k += \" n=\" + std::to_string(node->src[1]->ne[1]);\n"
            "    } else if (node->op == GGML_OP_GET_ROWS && node->src[0] && node->src[1]) {\n"
            "        // 2026-10-17: which gather is slow (Flash-Next GET_ROWS averaged 157 us). ne1 bucketed (KV views grow).\n"
            "        int64_t b = 1;\n"
            "        while (b < node->src[0]->ne[1]) {\n"
            "            b *= 2;\n"
            "        }\n"
            "        k += simplify(node->name);\n"
            "        k += ' ';\n"
            "        k += ggml_type_name(node->src[0]->type);\n"
            "        k += \" ne0=\" + std::to_string(node->src[0]->ne[0]) + \" ne1<=\" + std::to_string(b) +\n"
            "             \" ne2=\" + std::to_string(node->src[0]->ne[2]) + \" rows=\" + std::to_string(ggml_nelements(node->src[1]));\n"
            "    } else {\n")
edit("ggml/src/ggml-cuda/p100-prof.cu", [(PROF_OLD, PROF_NEW)])
shutil.copy(f"{SRC}/ggml/src/ggml-cuda/p100-prof.cu", f"{STAGE}/p100-prof.cu")

# ---- tests --------------------------------------------------------------------------------------------------------------
edit("tests/test-backend-ops.cpp", [
    ("    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 32, 2, false, 2880, 32, 2880));\n\n"
     "    // multiple blocks per row:",
     "    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 32, 2, false, 2880, 32, 2880));\n\n"
     "    // P100 work (2026-10-17): Flash-Next one-token expert shapes (32 instead of 256 experts) for the MMVQ_ID presets,\n"
     "    // and short-K float mat-vecs for MMVF_SHORTK (incl. row counts that are not a multiple of the rows per block).\n"
     "    for (ggml_type type_a : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS}) {\n"
     "        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 32, 10, true, 640, 1, 2560));\n"
     "    }\n"
     "    for (ggml_type type_a : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0}) {\n"
     "        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 32, 10, false, 2560, 1, 640));\n"
     "    }\n"
     "    for (ggml_type type_a : {GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32}) {\n"
     "        for (auto mk : std::vector<std::pair<int64_t, int64_t>>{{10240, 320}, {1001, 96}, {37, 1024}, {33, 2}}) {\n"
     "            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, mk.first, 1, mk.second, {1, 1}, {1, 1}));\n"
     "        }\n"
     "    }\n\n"
     "    // multiple blocks per row:"),
    ("            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, mk.first, 4, mk.second, {1, 1}, {1, 1})); // MTP verify\n"
     "        }\n"
     "    }\n\n"
     "    return test_cases;\n",
     "            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, mk.first, 4, mk.second, {1, 1}, {1, 1})); // MTP verify\n"
     "        }\n"
     "    }\n\n"
     "    // P100 work (2026-10-17): Qwen3.8-Flash-Next one-token decode. Experts (64 of the 256 allocated to keep host RAM\n"
     "    // small; 10 picked at random each run, so still DRAM-bound): gate/up 640 x 2560 with the input shared, down\n"
     "    // 2560 x 640 with one input row per expert. Hyper-connection up (10240 x 320) and down (320 x 10240) projections.\n"
     "    for (ggml_type type_a : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS}) {\n"
     "        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 64, 10, true, 640, 1, 2560));\n"
     "    }\n"
     "    for (ggml_type type_a : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0}) {\n"
     "        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 64, 10, false, 2560, 1, 640));\n"
     "    }\n"
     "    for (ggml_type type_a : {GGML_TYPE_BF16, GGML_TYPE_F16}) {\n"
     "        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 10240, 1, 320, {1, 1}, {1, 1}));\n"
     "        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 320, 1, 10240, {1, 1}, {1, 1}));\n"
     "    }\n\n"
     "    return test_cases;\n"),
])

# CPU thread count for perf runs (NUMA test of the CPU expert layers, 2026-10-17)
edit("tests/test-backend-ops.cpp", [(
    "            ggml_backend_set_n_threads_fn(backend.get(), std::max<int>(1, N_THREADS/2));\n",
    "            // 2026-10-17: GGML_TEST_THREADS overrides the CPU thread count (P100/NUMA work)\n"
    "            const char * test_threads = getenv(\"GGML_TEST_THREADS\");\n"
    "            ggml_backend_set_n_threads_fn(backend.get(), test_threads ? std::max(1, atoi(test_threads)) : std::max<int>(1, N_THREADS/2));\n",
)], marker="GGML_TEST_THREADS")
print("done")
