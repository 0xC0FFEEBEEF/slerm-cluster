#!/usr/bin/env python3
"""Brick 6b (2026-10-08): FA tile kernel with padded GQA packing (FA_GQA_PAD knob) for GQA ratios no power of two
divides (Qwen3.8: 6 -> K/V read 3x per pass with ncols2 = 2). Applies to /data/llm/src/llama.cpp-v050. Idempotent."""
import sys

SRC = "/data/llm/src/llama.cpp-v050"


def edit(rel, pairs):
    path = f"{SRC}/{rel}"
    s = open(path).read()
    for old, new in pairs:
        if new in s:
            continue
        n = s.count(old)
        if n != 1:
            sys.exit(f"{rel}: anchor found {n} times: {old[:90]!r}")
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("edited", rel)


edit("ggml/src/ggml-cuda/p100-tune.cuh", [
    ("    int mmvq_l  = 0;\n};",
     "    int mmvq_l  = 0;\n"
     "    int fa_gqa_pad = 0; // FA tile kernel: pack 4 or 8 Q heads per K/V head when no power of two divides the GQA ratio\n};"),
])

edit("ggml/src/ggml-cuda/ggml-cuda.cu", [
    ('    else if (strcmp(key, "PROF")      == 0) { ggml_cuda_prof_arm(atoi(val)); }\n',
     '    else if (strcmp(key, "PROF")      == 0) { ggml_cuda_prof_arm(atoi(val)); }\n'
     '    else if (strcmp(key, "FA_GQA_PAD") == 0) { ggml_cuda_tune.fa_gqa_pad = atoi(val); }\n'),
    ('            {"GGML_CUDA_MMVH_MC", "MMVH_MC"},\n',
     '            {"GGML_CUDA_MMVH_MC", "MMVH_MC"}, {"GGML_CUDA_FA_GQA_PAD", "FA_GQA_PAD"},\n'),
])

edit("ggml/src/ggml-cuda/fattn-tile.cuh", [
    ('#include "fattn-common.cuh"\n',
     '#include "fattn-common.cuh"\n#include "p100-tune.cuh"\n'),
    ("    const int sequence = blockIdx.z / (ne02/ncols2);\n"
     "    const int head0 = blockIdx.z*ncols2 - sequence*ne02; // == blockIdx.z % (ne02/ncols2)\n"
     "    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.\n"
     "    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);\n"
     "    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));\n"
     "    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape\n",
     "    // 2026-10-08 P100 work: blocks are (sequence, K/V head, group of ncols2 Q heads) as in launch_fattn's grid, so\n"
     "    // ncols2 need not divide the GQA ratio: heads past the ratio are padding (zero Q, never written). With a dividing\n"
     "    // ncols2 this is the old mapping (head0 == blockIdx.z*ncols2 modulo ne02).\n"
     "    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.\n"
     "    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;\n"
     "    const int sequence = blockIdx.z / (ntiles_z_gqa*ne12);\n"
     "    const int z_rem    = blockIdx.z - sequence*(ntiles_z_gqa*ne12);\n"
     "    const int z_KV     = z_rem / ntiles_z_gqa;\n"
     "    const int zt_gqa   = z_rem - z_KV*ntiles_z_gqa;\n"
     "    const int head0    = z_KV*gqa_ratio + zt_gqa*ncols2;\n"
     "    const int nvalid2  = min(ncols2, gqa_ratio - zt_gqa*ncols2); // real Q heads in this block\n"
     "    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);\n"
     "    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*z_KV);\n"
     "    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*z_KV); // K and V have same shape\n"),
    ("                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};\n"
     "                ggml_cuda_memcpy_1<sizeof(tmp_f)>\n"
     "                    (tmp_f, &Q_f[c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float))\n"
     "                                 + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);\n",
     "                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};\n"
     "                if (c < nvalid2) { // padding heads keep zero Q\n"
     "                    ggml_cuda_memcpy_1<sizeof(tmp_f)>\n"
     "                        (tmp_f, &Q_f[c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float))\n"
     "                                     + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);\n"
     "                }\n"),
    ("            const float sink = ((const float *) sinks)[head0 + jc % ncols2];\n",
     "            const float sink = jc % ncols2 < nvalid2 ? ((const float *) sinks)[head0 + jc % ncols2] : -INFINITY;\n"),
    ("        if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) {\n            return;\n        }\n",
     "        if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) {\n            return;\n        }\n"
     "        if (c >= nvalid2) {\n            continue; // padding head\n        }\n"),
    ("    if constexpr (DKQ <= 512 && DKQ != 320 && DKQ != 192) {\n"
     "        if (use_gqa_opt && gqa_ratio % 8 == 0) {\n",
     "    if constexpr (DKQ <= 512 && DKQ != 320 && DKQ != 192) {\n"
     "        // 2026-10-08 P100 work: a GQA ratio that no power of two divides well (Qwen3.8: 6) packs 2 heads below and\n"
     "        // reads every K/V head 3 times per pass; FA_GQA_PAD=4/8 packs 4 or 8 instead, padding the rest.\n"
     "        if (use_gqa_opt && ggml_cuda_tune.fa_gqa_pad >= 4 && gqa_ratio % 4 != 0 && gqa_ratio > 2) {\n"
     "            if (ggml_cuda_tune.fa_gqa_pad >= 8 && gqa_ratio > 4) {\n"
     "                launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(ctx, dst);\n"
     "            } else {\n"
     "                launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(ctx, dst);\n"
     "            }\n"
     "            return;\n"
     "        }\n\n"
     "        if (use_gqa_opt && gqa_ratio % 8 == 0) {\n"),
])

edit("tests/test-backend-ops.cpp", [
    ("    // Multi-column MMVQ coverage for the Q4_K weight-reuse path",
     "    // P100 work (2026-10-08): Qwen3.8 attention per GPU = 2 K/V heads x GQA 6, head size 256 (FA_GQA_PAD packing)\n"
     "    for (int64_t nb : {1, 4, 8, 33}) {\n"
     "        for (bool sinks : {false, true}) {\n"
     "            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, 1024, nb, true, sinks, 0.0f, 0.0f,\n"
     "                                                            GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));\n"
     "        }\n"
     "    }\n\n"
     "    // Multi-column MMVQ coverage for the Q4_K weight-reuse path"),
    ("    // P100 work (slerm, 2026-10-07): Qwen3.8-27B one-token decode shapes",
     "    // P100 work (2026-10-08): Qwen3.8 attention per GPU at 8k and 92k context, decode (1) and MTP verify (4)\n"
     "    for (int64_t kv : {8192, 92160}) {\n"
     "        for (int64_t nb : {1, 4}) {\n"
     "            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {6, 1}, kv, nb, true, false, 0.0f, 0.0f,\n"
     "                                                            GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));\n"
     "        }\n"
     "    }\n\n"
     "    // P100 work (slerm, 2026-10-07): Qwen3.8-27B one-token decode shapes"),
])
print("done")
