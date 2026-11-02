#!/usr/bin/env python3
"""Brick 8 (2026-10-08): v5 multi-column mmvh (MMVH_MC = 100 + R): activations staged in shared memory, one raw row
live per lane, so R can grow to 6-8. IQ4_XS decodes run pairs from one codebook lookup. Edits the staged
mmvh-pascal.cu and copies it into the tree. Idempotent."""
import shutil, sys

SRC = "/data/llm/src/llama.cpp-v050"
STAGE = "/data/llm/src/p100-local-files/mmvh-pascal.cu"
s = open(STAGE).read()

V5 = r'''
// ---------------------------------------------------------------------------------------------------------------------
// v5 multi-column (2026-10-08, MMVH_MC = 100 + R). v4 keeps every column's activations of a run in registers next to
// R raw rows (~190-240 registers at R = 4, 8 warps per SM), and more rows per row set was the only change that ever
// helped. Here the block stages the current step's activations (all NC columns) in shared memory once, and a lane
// keeps a single raw row live (plus the next one, prefetched), so R can be 6-8 at ~16 warps per SM.
// Shared layout: one 128-byte segment per (column, super-block slot, lane c) with that lane's 64 activations in run
// order; its eight 16-byte chunks are XOR-swizzled by (slot*4 + c) & 7, so the 8 lanes of an LDS.128 phase read 8
// different bank groups.

// IQ4_XS run pair p (runs 2p = low nibbles, 2p+1 = high nibbles of words 4p .. 4p+3): one codebook lookup per word
static __device__ __forceinline__ void mmvh_iq4_pair(const mmvh_raw4 & raw, const int p, half2 * w0, half2 * w1) {
    const half2 off = __float2half2_rn(1152.0f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi;
        mmvh_iq4_lut(raw.q[4*p + j], lo, hi);
        w0[2*j + 0] = __hsub2(mmvh_h2( lo       & 0x00FF00FFu), off);
        w0[2*j + 1] = __hsub2(mmvh_h2((lo >> 8) & 0x00FF00FFu), off);
        w1[2*j + 0] = __hsub2(mmvh_h2( hi       & 0x00FF00FFu), off);
        w1[2*j + 1] = __hsub2(mmvh_h2((hi >> 8) & 0x00FF00FFu), off);
    }
}

// one run (RL2 half2 of weights) against every column, activations from the staged segment
template <ggml_type type, int NC, int STEP, int k>
static __device__ __forceinline__ void mmvh_mc5_run(const half2 * w, const mmvh_raw4 & raw, const int c, const int slot,
                                                    const int kb, const int4 * __restrict__ xs,
                                                    const float2 * __restrict__ ss, const int ng, float (&acc)[NC]) {
    constexpr int NRUN = mmvh_traits<type>::nrun;
    constexpr int RL2  = 32 / NRUN;
    constexpr int CPR  = RL2 / 4;
    const int sw = (slot*4 + c) & 7;
    const int g  = (kb*256 + mmvh_run_off<type>(c, k)) / 32;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const int4 * seg = xs + ((j*STEP + slot)*4 + c)*8;
        half2 xv[RL2];
#pragma unroll
        for (int v = 0; v < CPR; ++v) {
            *reinterpret_cast<int4 *>(&xv[4*v]) = seg[(k*CPR + v) ^ sw];
        }
        const float2 sk = ss[(int64_t) j*ng + g];
        half2 a0 = __hmul2(w[0], xv[0]);
        half2 a1 = __hmul2(w[1], xv[1]);
#pragma unroll
        for (int v = 2; v < RL2; v += 2) {
            a0 = __hfma2(w[v],     xv[v],     a0);
            a1 = __hfma2(w[v + 1], xv[v + 1], a1);
        }
        const float2 f0 = __half22float2(a0), f1 = __half22float2(a1);
        acc[j] += raw.mul[k] * sk.x * ((f0.x + f0.y) + (f1.x + f1.y));
        if constexpr (mmvh_traits<type>::mins) {
            acc[j] -= raw.mmin[k] * sk.y;
        }
    }
}

template <ggml_type type, int NC, int STEP>
static __device__ __forceinline__ void mmvh_mc5_row(const mmvh_raw4 & raw, const int c, const int slot, const int kb,
                                                    const int4 * __restrict__ xs, const float2 * __restrict__ ss,
                                                    const int ng, float (&acc)[NC]) {
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        half2 w0[8], w1[8];
        mmvh_iq4_pair(raw, 0, w0, w1);
        mmvh_mc5_run<type, NC, STEP, 0>(w0, raw, c, slot, kb, xs, ss, ng, acc);
        mmvh_mc5_run<type, NC, STEP, 1>(w1, raw, c, slot, kb, xs, ss, ng, acc);
        mmvh_iq4_pair(raw, 1, w0, w1);
        mmvh_mc5_run<type, NC, STEP, 2>(w0, raw, c, slot, kb, xs, ss, ng, acc);
        mmvh_mc5_run<type, NC, STEP, 3>(w1, raw, c, slot, kb, xs, ss, ng, acc);
    } else {
        constexpr int RL2 = 32 / mmvh_traits<type>::nrun;
        half2 w[RL2];
        mmvh_unpack_run4<type, 0>(raw, w);
        mmvh_mc5_run<type, NC, STEP, 0>(w, raw, c, slot, kb, xs, ss, ng, acc);
        mmvh_unpack_run4<type, 1>(raw, w);
        mmvh_mc5_run<type, NC, STEP, 1>(w, raw, c, slot, kb, xs, ss, ng, acc);
        if constexpr (mmvh_traits<type>::nrun == 4) {
            mmvh_unpack_run4<type, 2>(raw, w);
            mmvh_mc5_run<type, NC, STEP, 2>(w, raw, c, slot, kb, xs, ss, ng, acc);
            mmvh_unpack_run4<type, 3>(raw, w);
            mmvh_mc5_run<type, NC, STEP, 3>(w, raw, c, slot, kb, xs, ss, ng, acc);
        }
    }
}

template <ggml_type type, int R, int NC, bool HALF>
static __global__ void __launch_bounds__(128, 1) mmvh_mc5_kernel(
        const char * __restrict__ vx, const half * __restrict__ xh, const float2 * __restrict__ ss,
        float * __restrict__ dst, const int nrows, const int nsb, const int64_t row_bytes, const int64_t s_dst) {
    using tr = mmvh_traits<type>;
    constexpr int NRUN = tr::nrun;
    constexpr int CPR  = (32 / NRUN) / 4;     // 16-byte chunks per run
    constexpr int STEP = HALF ? 4 : 8;        // super-blocks per step
    constexpr int RPW  = HALF ? 2*R : R;      // rows per warp
    constexpr int NSEG = NC*STEP*4;           // 128-byte segments per step
    constexpr int L    = HALF ? 16 : 32;      // lanes per row set
    __shared__ int4 xs[NSEG*8];               // 16 KB at NC = 4 (full), 8 KB (half)

    const int lane = threadIdx.x;
    const int tid  = threadIdx.y*32 + lane;
    const int c    = lane & 3;
    const int slot = HALF ? ((lane >> 2) & 3) : (lane >> 2);
    const int rset = HALF ? (lane >> 4) : 0;
    const int wrow = (blockIdx.x*blockDim.y + threadIdx.y) * RPW;
    const bool active = wrow < nrows;         // no early return: every warp joins the copies and barriers
    const int row0 = wrow + rset*R;
    const int ng   = nsb*8;

    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    for (int kb0 = 0; kb0 < nsb; kb0 += STEP) {
        for (int i = tid; i < NSEG*8; i += 128) {
            const int q   = i & 7;                      // chunk in the lane's 64 activations, run order
            const int seg = i >> 3;                     // (j*STEP + slot)*4 + cc
            const int cc  = seg & 3;
            const int sl  = (seg >> 2) % STEP;
            const int j   = (seg >> 2) / STEP;
            const int kb  = kb0 + sl;
            int4 v = make_int4(0, 0, 0, 0);
            if (kb < nsb) {
                const int e = kb*256 + mmvh_run_off<type>(cc, q / CPR) + (q % CPR)*8;
                v = *reinterpret_cast<const int4 *>(xh + (int64_t) j*ng*32 + e);
            }
            xs[seg*8 + (q ^ ((sl*4 + cc) & 7))] = v;
        }
        __syncthreads();

        const int kb = kb0 + slot;
        if (active && kb < nsb) {
            const char * base = vx + (int64_t) kb*tr::bs;
            mmvh_raw4 cur;
            mmvh_load_raw4<type>(base + (int64_t) min(row0, nrows - 1)*row_bytes, c, cur);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                mmvh_raw4 nxt;
                if (r + 1 < R) {
                    mmvh_load_raw4<type>(base + (int64_t) min(row0 + r + 1, nrows - 1)*row_bytes, c, nxt);
                }
                mmvh_mc5_row<type, NC, STEP>(cur, c, slot, kb, xs, ss, ng, acc[r]);
                if (r + 1 < R) {
                    cur = nxt;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = L/2; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, o, 32);
            }
            acc[r][j] = v;
        }
    }
    const int lrow = lane & (L - 1);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (active && lrow == (r*NC + j) % L && row0 + r < nrows) {
                dst[(int64_t) j*s_dst + row0 + r] = acc[r][j];
            }
        }
    }
}

'''
marker = "// ---------------------------------------------------------------------------------------------------------------------\n// dispatch\n"
if "mmvh_mc5_kernel" not in s:
    assert s.count(marker) == 1
    s = s.replace(marker, V5.lstrip("\n") + marker)

LAUNCH5 = '''template <ggml_type type, int R, int NC>
static void mmvh_mc5_launch_t(const char * vx, const half * xh, const float2 * ss, float * d, int nrows, int nsb,
                              int64_t rb, int64_t s_dst, cudaStream_t stream) {
    const int  mode = ggml_cuda_tune.mmvh / 10; // 0 auto, 1 full, 2 half (as for one column)
    const bool half_mode = mode == 2 || (mode == 0 && nsb % 8 != 0 && nsb % 4 == 0);
    const int  rpw = half_mode ? 2*R : R;
    const int  nb  = (nrows + 4*rpw - 1) / (4*rpw);
    if (half_mode) {
        mmvh_mc5_kernel<type, R, NC, true ><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);
    } else {
        mmvh_mc5_kernel<type, R, NC, false><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);
    }
}

template <ggml_type type, int NC>
static void mmvh_mc5_launch_r(int r, const char * vx, const half * xh, const float2 * ss, float * d, int nrows, int nsb,
                              int64_t rb, int64_t s_dst, cudaStream_t stream) {
    switch (r) {
        case 4:  mmvh_mc5_launch_t<type, 4, NC>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        case 6:  mmvh_mc5_launch_t<type, 6, NC>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        default: mmvh_mc5_launch_t<type, 8, NC>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
    }
}

template <ggml_type type>
static void mmvh_launch_type('''
old_lt = "template <ggml_type type>\nstatic void mmvh_launch_type("
if "mmvh_mc5_launch_t" not in s:
    assert s.count(old_lt) == 1
    s = s.replace(old_lt, LAUNCH5)

old_d = "    if (ggml_cuda_tune.mmvh_mc >= 2) { // v4 (any of the four types)\n"
new_d = ("    if (ggml_cuda_tune.mmvh_mc >= 100) { // v5: MMVH_MC = 100 + rows per row set (4, 6, 8)\n"
         "        const int r = ggml_cuda_tune.mmvh_mc - 100;\n        switch (nc) {\n"
         "            case 2:  mmvh_mc5_launch_r<type, 2>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
         "            case 3:  mmvh_mc5_launch_r<type, 3>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
         "            default: mmvh_mc5_launch_r<type, 4>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;\n"
         "        }\n        return;\n    }\n" + old_d)
if "mmvh_mc >= 100" not in s:
    assert s.count(old_d) == 1
    s = s.replace(old_d, new_d)

open(STAGE, "w").write(s)
shutil.copy(STAGE, f"{SRC}/ggml/src/ggml-cuda/mmvh-pascal.cu")
print("ok", s.count("mmvh_mc5_kernel"))
