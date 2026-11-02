#!/usr/bin/env python3
"""Brick 7 (2026-10-08): v4 multi-column mmvh + IQ4_XS, occupancy variants (rows per row set x min blocks per SM),
MMVH_IQ4_MC knob; FA_GQA_PAD only for one-token batches. Edits the tree under /data/llm/src/llama.cpp-v050 and the
staged copy in /data/llm/src/p100-local-files. Idempotent."""
import shutil, sys

SRC = "/data/llm/src/llama.cpp-v050"
STAGE = "/data/llm/src/p100-local-files/mmvh-pascal.cu"


def edit_text(s, pairs, name):
    for old, new in pairs:
        if new in s:
            continue
        n = s.count(old)
        if n != 1:
            sys.exit(f"{name}: anchor found {n} times: {old[:100]!r}")
        s = s.replace(old, new)
    return s


def edit(rel, pairs):
    p = f"{SRC}/{rel}"
    s = edit_text(open(p).read(), pairs, rel)
    open(p, "w").write(s)
    print("edited", rel)


# ---- mmvh-pascal.cu (staged copy is the source of truth; copied into the tree at the end)
s = open(STAGE).read()
s = edit_text(s, [
    # raw4: IQ4_XS keeps its 32 code bytes in q[8] and the run multipliers
    ("    } else { // Q6_K\n        const int n = c >> 1, lh = c & 1;\n        mmvh_load16_a2(b +       64*n +      16*lh, raw.q);\n"
     "        mmvh_load16_a2(b +       64*n + 32 + 16*lh, raw.q + 4);\n        mmvh_load16_a2(b + 128 + 32*n +      16*lh, raw.h);\n"
     "        const int8_t * sc = (const int8_t *) (b + 192 + 8*n) + lh;\n        const float d = __half2float(*reinterpret_cast<const half *>(b + 208));\n"
     "#pragma unroll\n        for (int k = 0; k < 4; ++k) {\n            raw.mul[k]  = d * (float) sc[2*k];\n            raw.mmin[k] = 0.0f;\n        }\n    }\n}\n\n"
     "// run k of the compact raw block",
     "    } else if constexpr (type == GGML_TYPE_Q6_K) {\n        const int n = c >> 1, lh = c & 1;\n        mmvh_load16_a2(b +       64*n +      16*lh, raw.q);\n"
     "        mmvh_load16_a2(b +       64*n + 32 + 16*lh, raw.q + 4);\n        mmvh_load16_a2(b + 128 + 32*n +      16*lh, raw.h);\n"
     "        const int8_t * sc = (const int8_t *) (b + 192 + 8*n) + lh;\n        const float d = __half2float(*reinterpret_cast<const half *>(b + 208));\n"
     "#pragma unroll\n        for (int k = 0; k < 4; ++k) {\n            raw.mul[k]  = d * (float) sc[2*k];\n            raw.mmin[k] = 0.0f;\n        }\n"
     "    } else { // IQ4_XS: lane c owns qs[32c .. 32c+31] = sub-blocks 2c, 2c+1 (8-byte aligned blocks)\n"
     "        const uint2 hdr = *reinterpret_cast<const uint2 *>(b);\n"
     "#pragma unroll\n        for (int i = 0; i < 4; ++i) {\n"
     "            const uint2 q2 = *reinterpret_cast<const uint2 *>(b + 8 + 32*c + 8*i);\n"
     "            raw.q[2*i] = q2.x;\n            raw.q[2*i + 1] = q2.y;\n        }\n"
     "        const float    d   = __half2float(*reinterpret_cast<const half *>(&hdr.x));\n"
     "        const uint32_t sh_ = hdr.x >> 16;            // scales_h\n"
     "        const uint32_t sl  = (hdr.y >> (8*c)) & 0xFF; // scales_l[c]: low nibble sub-block 2c, high 2c+1\n"
     "        const int ls0 = (int) ((sl & 0xF) | (((sh_ >> (4*c))     & 3) << 4)) - 32;\n"
     "        const int ls1 = (int) ((sl >> 4)  | (((sh_ >> (4*c + 2)) & 3) << 4)) - 32;\n"
     "        raw.mul[0] = raw.mul[1] = d * (float) ls0;\n        raw.mul[2] = raw.mul[3] = d * (float) ls1;\n"
     "        raw.mmin[0] = raw.mmin[1] = raw.mmin[2] = raw.mmin[3] = 0.0f;\n    }\n}\n\n"
     "// run k of the compact raw block"),
    # unpack: IQ4_XS run k = low (k even) or high (k odd) nibbles of words 4*(k/2) .. +3, through the codebook
    ("            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x003F003Fu), off);\n            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x003F003Fu), off);\n"
     "        }\n    }\n}\n\n// run k for every row and column",
     "            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x003F003Fu), off);\n            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x003F003Fu), off);\n"
     "        }\n"
     "    } else { // IQ4_XS: run k = low (k even) / high (k odd) nibbles of words 4*(k/2) .. +3, looked up in the codebook\n"
     "        const half2 off = __float2half2_rn(1152.0f); // 1024 + the table's +128\n"
     "#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n"
     "            uint32_t lo, hi;\n            mmvh_iq4_lut(raw.q[4*(k >> 1) + j], lo, hi);\n"
     "            const uint32_t v = (k & 1) ? hi : lo;\n"
     "            w[2*j + 0] = __hsub2(mmvh_h2( v       & 0x00FF00FFu), off);\n"
     "            w[2*j + 1] = __hsub2(mmvh_h2((v >> 8) & 0x00FF00FFu), off);\n        }\n    }\n}\n\n// run k for every row and column"),
    # v4 decoder: Q6_K becomes an explicit branch so IQ4_XS can follow it
    ("                v |= ((raw.h[i >> 1] >> (2*(i & 1) + k)) << 4) & 0x10101010u;\n            }\n"
     "            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x001F001Fu), off);\n            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x001F001Fu), off);\n"
     "        }\n    } else { // Q6_K: ql0",
     "                v |= ((raw.h[i >> 1] >> (2*(i & 1) + k)) << 4) & 0x10101010u;\n            }\n"
     "            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x001F001Fu), off);\n            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x001F001Fu), off);\n"
     "        }\n    } else if constexpr (type == GGML_TYPE_Q6_K) { // Q6_K: ql0"),
    # occupancy as a template parameter
    ("template <ggml_type type, int R, int NC, bool HALF>\nstatic __global__ void __launch_bounds__(128, 1) mmvh_mc_kernel(",
     "template <ggml_type type, int R, int NC, bool HALF, int MINB>\nstatic __global__ void __launch_bounds__(128, MINB) mmvh_mc_kernel("),
    ("template <ggml_type type, int R, int NC>\nstatic void mmvh_mc_launch_t(",
     "template <ggml_type type, int R, int NC, int MINB>\nstatic void mmvh_mc_launch_t("),
    ("        mmvh_mc_kernel<type, R, NC, true ><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);\n    } else {\n"
     "        mmvh_mc_kernel<type, R, NC, false><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);",
     "        mmvh_mc_kernel<type, R, NC, true,  MINB><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);\n    } else {\n"
     "        mmvh_mc_kernel<type, R, NC, false, MINB><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);"),
    ("    if (r >= 4) { mmvh_mc_launch_t<type, 4, NC>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); }\n"
     "    else        { mmvh_mc_launch_t<type, 2, NC>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); }\n",
     "    // MMVH_MC: rows per row set, or rows*10 + min blocks per SM (register cap for occupancy)\n"
     "    switch (r) {\n"
     "        case 2:  mmvh_mc_launch_t<type, 2, NC, 1>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "        case 24: mmvh_mc_launch_t<type, 2, NC, 4>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "        case 32: mmvh_mc_launch_t<type, 3, NC, 2>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "        case 33: mmvh_mc_launch_t<type, 3, NC, 3>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "        case 43: mmvh_mc_launch_t<type, 4, NC, 3>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "        default: mmvh_mc_launch_t<type, 4, NC, 1>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "    }\n"),
    # dispatch: v4 for every type incl. IQ4_XS; v3 stays Q4_K/Q5_K/Q6_K
    ("    if constexpr (type != GGML_TYPE_IQ4_XS) {\n        if (ggml_cuda_tune.mmvh_mc >= 2) { // v4: MMVH_MC=2 -> 2 rows per row set, 4 -> 4\n"
     "            const int r = ggml_cuda_tune.mmvh_mc;\n            switch (nc) {\n"
     "                case 2:  mmvh_mc_launch_r<type, 2>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "                case 3:  mmvh_mc_launch_r<type, 3>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "                default: mmvh_mc_launch_r<type, 4>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "            }\n            return;\n        }\n",
     "    if (ggml_cuda_tune.mmvh_mc >= 2) { // v4 (any of the four types)\n"
     "        const int r = ggml_cuda_tune.mmvh_mc;\n        switch (nc) {\n"
     "            case 2:  mmvh_mc_launch_r<type, 2>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "            case 3:  mmvh_mc_launch_r<type, 3>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "            default: mmvh_mc_launch_r<type, 4>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
     "        }\n        return;\n    }\n"
     "    if constexpr (type != GGML_TYPE_IQ4_XS) {\n"),
    # supported(): IQ4_XS one-column still needs MMVH_IQ4; multi-column only through v4 and MMVH_IQ4_MC
    ("    if (!ggml_cuda_mmvh_pascal_type_ok(src0->type)) {\n        return false;\n    }\n"
     "    const int64_t nc = src1->ne[1];\n"
     "    if (nc < 1 || nc > 4 || (nc > 1 && (!ggml_cuda_tune.mmvh_mc || src0->type == GGML_TYPE_IQ4_XS || fusion != nullptr)) ||\n",
     "    const int64_t nc = src1->ne[1];\n"
     "    if (nc == 1 && !ggml_cuda_mmvh_pascal_type_ok(src0->type)) {\n        return false; // IQ4_XS one column: MMVH_IQ4 (slower than MMVQ, off)\n    }\n"
     "    const bool iq4_mc_ok = src0->type != GGML_TYPE_IQ4_XS || (ggml_cuda_tune.mmvh_mc >= 2 && ggml_cuda_tune.mmvh_iq4_mc);\n"
     "    if (nc < 1 || nc > 4 || (nc > 1 && (!ggml_cuda_tune.mmvh_mc || !iq4_mc_ok || fusion != nullptr)) ||\n"),
], "mmvh-pascal.cu")
open(STAGE, "w").write(s)
shutil.copy(STAGE, f"{SRC}/ggml/src/ggml-cuda/mmvh-pascal.cu")
print("edited mmvh-pascal.cu (staged + tree)")

edit("ggml/src/ggml-cuda/p100-tune.cuh", [
    ("    int mmvh_mc   = 1; // 2-4 token batches (MTP verify) through mmvh\n",
     "    int mmvh_mc   = 1; // 2-4 token batches (MTP verify) through mmvh: 1 v3, 2/4 v4 rows per row set, rows*10+min blocks\n"
     "    int mmvh_iq4_mc = 1; // IQ4_XS 2-4 token batches through v4 (when mmvh_mc >= 2)\n"),
])
edit("ggml/src/ggml-cuda/ggml-cuda.cu", [
    ('    else if (strcmp(key, "FA_GQA_PAD") == 0) { ggml_cuda_tune.fa_gqa_pad = atoi(val); }\n',
     '    else if (strcmp(key, "FA_GQA_PAD") == 0) { ggml_cuda_tune.fa_gqa_pad = atoi(val); }\n'
     '    else if (strcmp(key, "MMVH_IQ4_MC") == 0) { ggml_cuda_tune.mmvh_iq4_mc = atoi(val); }\n'),
    ('{"GGML_CUDA_FA_GQA_PAD", "FA_GQA_PAD"},\n',
     '{"GGML_CUDA_FA_GQA_PAD", "FA_GQA_PAD"},\n            {"GGML_CUDA_MMVH_IQ4_MC", "MMVH_IQ4_MC"},\n'),
])
edit("ggml/src/ggml-cuda/fattn-tile.cuh", [
    ("        // Decode / MTP verify batches only: with many tokens per block the token tiles already share K/V and a\n"
     "        // wider head group would only shrink them.\n"
     "        if (use_gqa_opt && ggml_cuda_tune.fa_gqa_pad >= 4 && gqa_ratio % 4 != 0 && gqa_ratio > 2 && Q->ne[1] <= 8) {\n",
     "        // One-token batches only (measured 2026-10-08 at 92k: 1 token 537 -> 392 us with 8; 4 tokens no gain, and\n"
     "        // slower at 8k, the wider tile being compute-bound on P100).\n"
     "        if (use_gqa_opt && ggml_cuda_tune.fa_gqa_pad >= 4 && gqa_ratio % 4 != 0 && gqa_ratio > 2 && Q->ne[1] == 1) {\n"),
])
edit("tests/test-backend-ops.cpp", [
    ("        // Qwen3.8 K per GPU: 5120 = 20 super-blocks (half-warp mode), 8704 = 34 (full mode, partial last step); P100 mmvh v4\n",
     "        // Qwen3.8 K per GPU: 5120 = 20 super-blocks (half-warp mode), 8704 = 34 (full mode, partial last step); P100 mmvh v4\n"
     "        // (IQ4_XS included since brick 7)\n"),
])
print("done")
