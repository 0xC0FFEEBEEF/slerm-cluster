#include "mmvh-pascal.cuh"
#include "p100-tune.cuh"

// One- to four-token mat-vec for Q4_K / Q5_K / Q6_K / IQ4_XS on GP100 (P100, sm_60), local P100 work 2026-10-07.
//
// GP100 has no __dp4a but full-rate HFMA2. The generic MMVQ path emulates the byte dot products: ~8.2 instructions per
// weight on Q5_K, issue-bound (72-83% issue slots, ~2% memory stalls) at ~300 GB/s. Here quants become half2 with the
// 0x6400 exponent trick (0x6400 | q == 1024 + q exactly), the offset comes off with one HSUB2 and products accumulate
// with HFMA2 against fp16 activations. Each lane unpacks its weights once and reuses them for every column, so the
// 2-4 token verify batches of MTP speculative decoding cost little more than one token.
//
// Activations: every 32-value group is divided by its max |x| (|xh| <= 1, so a lane's <= 16 fp16 products stay far
// from overflow) and stored as half, permuted [0,2,1,3] within each 4 so the two half2 a byte-quad yields line up with
// it. The group scale and the exact fp32 group sum (for the Q4_K/Q5_K mins) sit beside it as float2.
//
// Layout: 4 lanes per 256-weight super-block, each lane owns 64 weights split into runs of one scale each. Full mode:
// a warp walks 8 super-blocks per step for R rows. Half mode (K/256 not a multiple of 8, e.g. 20 or 34): each half-warp
// walks 4 super-blocks for its own R rows, which wastes fewer lanes on the last step. 4 warps per block.

// ---------------------------------------------------------------------------------------------------------------------
// activation prep

static __global__ void mmvh_prep_x(const float * __restrict__ x, half * __restrict__ xh, float2 * __restrict__ ss,
                                   const int ngroups, const int64_t sx_col) {
    const int g   = blockIdx.x*blockDim.y + threadIdx.y;
    const int col = blockIdx.y;
    if (g >= ngroups) {
        return;
    }
    const int   lane = threadIdx.x;
    const float v    = x[col*sx_col + g*32 + lane];
    const float amax = warp_reduce_max<32>(fabsf(v));
    const float sum  = warp_reduce_sum<32>(v);
    const float s    = amax > 0.0f ? amax : 1.0f;
    const int   p    = (lane & ~3) | ((lane & 1) << 1) | ((lane & 2) >> 1); // 0,1,2,3 -> 0,2,1,3
    xh[(int64_t) col*ngroups*32 + g*32 + p] = __float2half(v / s);
    if (lane == 0) {
        ss[(int64_t) col*ngroups + g] = make_float2(s, sum);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// per-type unpack: lane c of a super-block -> 32 half2 weights (signed, offset removed) in run order, plus per run the
// multiplier (d * scale) and the min coefficient (dmin * m, Q4_K/Q5_K only)

static __device__ __forceinline__ half2 mmvh_h2(const uint32_t q) {
    const uint32_t t = q | 0x64006400u;
    return *reinterpret_cast<const half2 *>(&t);
}

template <ggml_type type> struct mmvh_traits;
template <> struct mmvh_traits<GGML_TYPE_Q4_K>   { static constexpr int bs = sizeof(block_q4_K);   static constexpr int nrun = 2; static constexpr int align = 16; static constexpr bool mins = true;  };
template <> struct mmvh_traits<GGML_TYPE_Q5_K>   { static constexpr int bs = sizeof(block_q5_K);   static constexpr int nrun = 2; static constexpr int align = 16; static constexpr bool mins = true;  };
template <> struct mmvh_traits<GGML_TYPE_Q6_K>   { static constexpr int bs = sizeof(block_q6_K);   static constexpr int nrun = 4; static constexpr int align = 4;  static constexpr bool mins = false; };
template <> struct mmvh_traits<GGML_TYPE_IQ4_XS> { static constexpr int bs = sizeof(block_iq4_xs); static constexpr int nrun = 4; static constexpr int align = 8;  static constexpr bool mins = false; };

// element offset (within the super-block) of run k of lane c; runs hold 64/nrun elements
template <ggml_type type>
static __device__ __forceinline__ int mmvh_run_off(const int c, const int k) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        return 64*c + 32*k;                   // sub-blocks 2c (low nibbles) and 2c+1 (high nibbles)
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return 128*(c >> 1) + 32*k + 16*(c & 1);
    } else {
        return 64*c + 16*k;                   // IQ4_XS: sub-block 2c lo/hi nibbles, then 2c+1 lo/hi
    }
}

// Q6_K blocks are 2-byte aligned: read 16 bytes as five aligned words, funnel-shifted (never leaves the block)
static __device__ __forceinline__ void mmvh_load16_a2(const char * p, uint32_t out[4]) {
    const uint32_t * w  = (const uint32_t *) ((uintptr_t) p & ~(uintptr_t) 3);
    const uint32_t   sh = ((uint32_t) (uintptr_t) p & 3) * 8;
    const uint32_t a0 = w[0], a1 = w[1], a2 = w[2], a3 = w[3], a4 = w[4];
    out[0] = __funnelshift_r(a0, a1, sh);
    out[1] = __funnelshift_r(a1, a2, sh);
    out[2] = __funnelshift_r(a2, a3, sh);
    out[3] = __funnelshift_r(a3, a4, sh);
}

// IQ4_XS codebook (kvalues_iq4nl) + 128, so a looked-up byte u is 0..255 and 0x6400 | u == 1024 + 128 + value
static __device__ __forceinline__ void mmvh_iq4_lut(const uint32_t q, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u;
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1); // bit 3 of a nibble picks the upper 8 entries
    uint32_t t[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t qi = q >> (16*i);
        t[i] = __byte_perm(__byte_perm(T0, T1, qi), __byte_perm(T2, T3, qi), sel >> (16*i));
    }
    lo = __byte_perm(t[0], t[1], 0x6420); // low nibbles of bytes 0..3, in order
    hi = __byte_perm(t[0], t[1], 0x7531); // high nibbles
}

template <ggml_type type>
static __device__ __forceinline__ void mmvh_unpack(const char * __restrict__ b, const int c,
                                                   half2 w[32], float mul[4], float mmin[4]) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr bool q5     = type == GGML_TYPE_Q5_K;
        constexpr int  qs_off = q5 ? 48 : 16;
        const int4 hdr = *reinterpret_cast<const int4 *>(b);
        int4 q[2];
        q[0] = *reinterpret_cast<const int4 *>(b + qs_off + 32*c);
        q[1] = *reinterpret_cast<const int4 *>(b + qs_off + 32*c + 16);
        [[maybe_unused]] int4 h[2];
        if constexpr (q5) {
            h[0] = *reinterpret_cast<const int4 *>(b + 16);
            h[1] = *reinterpret_cast<const int4 *>(b + 32);
        }
        const int * qi = reinterpret_cast<const int *>(q);
        [[maybe_unused]] const int * hi32 = reinterpret_cast<const int *>(h);
        const half2 off = __float2half2_rn(1024.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t lo = (uint32_t) qi[i] & 0x0F0F0F0Fu;
            uint32_t hi = ((uint32_t) qi[i] >> 4) & 0x0F0F0F0Fu;
            if constexpr (q5) {
                const uint32_t qh = (uint32_t) hi32[i];
                lo |= ((qh >> (2*c))     << 4) & 0x10101010u;
                hi |= ((qh >> (2*c + 1)) << 4) & 0x10101010u;
            }
            w[     2*i + 0] = __hsub2(mmvh_h2( lo       & 0x001F001Fu), off);
            w[     2*i + 1] = __hsub2(mmvh_h2((lo >> 8) & 0x001F001Fu), off);
            w[16 + 2*i + 0] = __hsub2(mmvh_h2( hi       & 0x001F001Fu), off);
            w[16 + 2*i + 1] = __hsub2(mmvh_h2((hi >> 8) & 0x001F001Fu), off);
        }
        // get_scale_min_k4 for sub-blocks 2c, 2c+1
        const uint32_t S0 = (uint32_t) hdr.y, S1 = (uint32_t) hdr.z, S2 = (uint32_t) hdr.w;
        const bool hi_form = c >= 2;
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&hdr.x));
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sh = 8*((2*c + k) & 3);
            const uint32_t a = S0 >> sh, bb = S1 >> sh, e = S2 >> sh;
            const int sc = hi_form ? (int) ((e & 0xF) | ((a  >> 2) & 0x30))        : (int) (a  & 63);
            const int m  = hi_form ? (int) (((e >> 4) & 0xF) | ((bb >> 2) & 0x30)) : (int) (bb & 63);
            mul[k]  = dm.x * (float) sc;
            mmin[k] = dm.y * (float) m;
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const int n = c >> 1, lh = c & 1;
        uint32_t ql0[4], ql1[4], qh[4];
        mmvh_load16_a2(b +       64*n +      16*lh, ql0);
        mmvh_load16_a2(b +       64*n + 32 + 16*lh, ql1);
        mmvh_load16_a2(b + 128 + 32*n +      16*lh, qh);
        const half2 off = __float2half2_rn(1056.0f); // 1024 + q6_K's 32
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t h = qh[i];
            uint32_t q[4];
            q[0] = ( ql0[i]       & 0x0F0F0F0Fu) | ((h << 4) & 0x30303030u);
            q[1] = ( ql1[i]       & 0x0F0F0F0Fu) | ((h << 2) & 0x30303030u);
            q[2] = ((ql0[i] >> 4) & 0x0F0F0F0Fu) | ( h       & 0x30303030u);
            q[3] = ((ql1[i] >> 4) & 0x0F0F0F0Fu) | ((h >> 2) & 0x30303030u);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                w[8*k + 2*i + 0] = __hsub2(mmvh_h2( q[k]       & 0x003F003Fu), off);
                w[8*k + 2*i + 1] = __hsub2(mmvh_h2((q[k] >> 8) & 0x003F003Fu), off);
            }
        }
        const int8_t * sc = (const int8_t *) (b + 192 + 8*n) + lh;
        const float d = __half2float(*reinterpret_cast<const half *>(b + 208));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            mul[k]  = d * (float) sc[2*k];
            mmin[k] = 0.0f;
        }
    } else { // IQ4_XS: d, scales_h (u16), scales_l[4], qs[128]; lane c owns qs[32c .. 32c+31] = sub-blocks 2c, 2c+1
        const uint2 hdr = *reinterpret_cast<const uint2 *>(b);
        uint2 q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = *reinterpret_cast<const uint2 *>(b + 8 + 32*c + 8*i);
        }
        const uint32_t * qi = reinterpret_cast<const uint32_t *>(q);
        const half2 off = __float2half2_rn(1152.0f); // 1024 + the table's +128
#pragma unroll
        for (int i = 0; i < 8; ++i) { // ints 0..3: sub-block 2c, 4..7: sub-block 2c+1
            uint32_t lo, hi;
            mmvh_iq4_lut(qi[i], lo, hi);
            const int k0 = (i >> 2) * 2;      // runs 0,1 (2c lo/hi) or 2,3 (2c+1 lo/hi)
            const int j  = i & 3;
            w[8*(k0 + 0) + 2*j + 0] = __hsub2(mmvh_h2( lo       & 0x00FF00FFu), off);
            w[8*(k0 + 0) + 2*j + 1] = __hsub2(mmvh_h2((lo >> 8) & 0x00FF00FFu), off);
            w[8*(k0 + 1) + 2*j + 0] = __hsub2(mmvh_h2( hi       & 0x00FF00FFu), off);
            w[8*(k0 + 1) + 2*j + 1] = __hsub2(mmvh_h2((hi >> 8) & 0x00FF00FFu), off);
        }
        const float    d   = __half2float(*reinterpret_cast<const half *>(&hdr.x));
        const uint32_t sh_ = hdr.x >> 16;            // scales_h
        const uint32_t sl  = (hdr.y >> (8*c)) & 0xFF; // scales_l[c]: low nibble sub-block 2c, high 2c+1
        const int ls0 = (int) ((sl & 0xF) | (((sh_ >> (4*c))     & 3) << 4)) - 32;
        const int ls1 = (int) ((sl >> 4)  | (((sh_ >> (4*c + 2)) & 3) << 4)) - 32;
        mul[0] = mul[1] = d * (float) ls0;
        mul[2] = mul[3] = d * (float) ls1;
        mmin[0] = mmin[1] = mmin[2] = mmin[3] = 0.0f;
    }
}

// activations of column j for lane c's runs in super-block kb
template <ggml_type type>
static __device__ __forceinline__ void mmvh_load_x(const half * __restrict__ xh, const float2 * __restrict__ ss, const int ng,
                                                   const int j, const int kb, const int c, half2 xv[32], float2 sk[4]) {
    constexpr int NRUN = mmvh_traits<type>::nrun;
    constexpr int RL2  = 32 / NRUN;
#pragma unroll
    for (int k = 0; k < NRUN; ++k) {
        const int e0 = kb*256 + mmvh_run_off<type>(c, k);
        const int4 * xa = reinterpret_cast<const int4 *>(xh + (int64_t) j*ng*32 + e0);
#pragma unroll
        for (int v = 0; v < RL2/4; ++v) {
            *reinterpret_cast<int4 *>(&xv[k*RL2 + 4*v]) = xa[v];
        }
        sk[k] = ss[(int64_t) j*ng + e0/32];
    }
}

// one lane's 64 weights against one column. Two accumulators per run, runs interleaved: a single 16-long HFMA2 chain
// per run left the scheduler waiting on HFMA2 latency at this occupancy. Each accumulator half sums <= 8 products.
template <ggml_type type>
static __device__ __forceinline__ float mmvh_dot(const half2 w[32], const half2 xv[32], const float mul[4],
                                                 const float mmin[4], const float2 sk[4]) {
    constexpr int NRUN = mmvh_traits<type>::nrun;
    constexpr int RL2  = 32 / NRUN;
    half2 a[NRUN][2];
#pragma unroll
    for (int k = 0; k < NRUN; ++k) {
        a[k][0] = __hmul2(w[k*RL2 + 0], xv[k*RL2 + 0]);
        a[k][1] = __hmul2(w[k*RL2 + 1], xv[k*RL2 + 1]);
    }
#pragma unroll
    for (int v = 2; v < RL2; ++v) {
#pragma unroll
        for (int k = 0; k < NRUN; ++k) {
            a[k][v & 1] = __hfma2(w[k*RL2 + v], xv[k*RL2 + v], a[k][v & 1]);
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < NRUN; ++k) {
        const float2 f0 = __half22float2(a[k][0]);
        const float2 f1 = __half22float2(a[k][1]);
        sum += mul[k] * sk[k].x * ((f0.x + f0.y) + (f1.x + f1.y));
        if constexpr (mmvh_traits<type>::mins) {
            sum -= mmin[k] * sk[k].y; // Q4_K/Q5_K: one min per 32-value run times the exact group sum
        }
    }
    return sum;
}

// Multi-column (MTP verify, 2-4 tokens): weights are unpacked one run at a time (16 or 8 half2 live) and each run is
// reused for every column, with all columns' activations for that run loaded together so their latencies overlap.
struct mmvh_raw {
    uint32_t q[8];   // Q4_K/Q5_K: qs words; Q6_K: ql0[4], ql1[4]
    uint32_t h[8];   // Q5_K: qh words; Q6_K: qh[4]
    float    mul[4];
    float    mmin[4];
};

template <ggml_type type>
static __device__ __forceinline__ void mmvh_load_raw(const char * __restrict__ b, const int c, mmvh_raw & raw) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr bool q5     = type == GGML_TYPE_Q5_K;
        constexpr int  qs_off = q5 ? 48 : 16;
        const int4 hdr = *reinterpret_cast<const int4 *>(b);
        const int4 q0 = *reinterpret_cast<const int4 *>(b + qs_off + 32*c);
        const int4 q1 = *reinterpret_cast<const int4 *>(b + qs_off + 32*c + 16);
        raw.q[0] = q0.x; raw.q[1] = q0.y; raw.q[2] = q0.z; raw.q[3] = q0.w;
        raw.q[4] = q1.x; raw.q[5] = q1.y; raw.q[6] = q1.z; raw.q[7] = q1.w;
        if constexpr (q5) {
            const int4 h0 = *reinterpret_cast<const int4 *>(b + 16);
            const int4 h1 = *reinterpret_cast<const int4 *>(b + 32);
            raw.h[0] = h0.x; raw.h[1] = h0.y; raw.h[2] = h0.z; raw.h[3] = h0.w;
            raw.h[4] = h1.x; raw.h[5] = h1.y; raw.h[6] = h1.z; raw.h[7] = h1.w;
        }
        const uint32_t S0 = (uint32_t) hdr.y, S1 = (uint32_t) hdr.z, S2 = (uint32_t) hdr.w;
        const bool hi_form = c >= 2;
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&hdr.x));
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sh = 8*((2*c + k) & 3);
            const uint32_t a = S0 >> sh, bb = S1 >> sh, e = S2 >> sh;
            const int sc = hi_form ? (int) ((e & 0xF) | ((a  >> 2) & 0x30))        : (int) (a  & 63);
            const int m  = hi_form ? (int) (((e >> 4) & 0xF) | ((bb >> 2) & 0x30)) : (int) (bb & 63);
            raw.mul[k]  = dm.x * (float) sc;
            raw.mmin[k] = dm.y * (float) m;
        }
    } else { // Q6_K
        const int n = c >> 1, lh = c & 1;
        mmvh_load16_a2(b +       64*n +      16*lh, raw.q);
        mmvh_load16_a2(b +       64*n + 32 + 16*lh, raw.q + 4);
        mmvh_load16_a2(b + 128 + 32*n +      16*lh, raw.h);
        const int8_t * sc = (const int8_t *) (b + 192 + 8*n) + lh;
        const float d = __half2float(*reinterpret_cast<const half *>(b + 208));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            raw.mul[k]  = d * (float) sc[2*k];
            raw.mmin[k] = 0.0f;
        }
    }
}

// run k of the raw block as half2 (16 for Q4_K/Q5_K, 8 for Q6_K), offset removed
template <ggml_type type, int k>
static __device__ __forceinline__ void mmvh_unpack_run(const mmvh_raw & raw, const int c, half2 * w) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const half2 off = __float2half2_rn(1024.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t v = k == 0 ? (raw.q[i] & 0x0F0F0F0Fu) : ((raw.q[i] >> 4) & 0x0F0F0F0Fu);
            if constexpr (type == GGML_TYPE_Q5_K) {
                v |= ((raw.h[i] >> (2*c + k)) << 4) & 0x10101010u;
            }
            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x001F001Fu), off);
            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x001F001Fu), off);
        }
    } else { // Q6_K: ql0 = raw.q[0..3], ql1 = raw.q[4..7], qh = raw.h[0..3]
        const half2 off = __float2half2_rn(1056.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t h = raw.h[i];
            uint32_t v;
            if constexpr (k == 0) { v = ( raw.q[i]           & 0x0F0F0F0Fu) | ((h << 4) & 0x30303030u); }
            if constexpr (k == 1) { v = ( raw.q[4 + i]       & 0x0F0F0F0Fu) | ((h << 2) & 0x30303030u); }
            if constexpr (k == 2) { v = ((raw.q[i]     >> 4) & 0x0F0F0F0Fu) | ( h       & 0x30303030u); }
            if constexpr (k == 3) { v = ((raw.q[4 + i] >> 4) & 0x0F0F0F0Fu) | ((h >> 2) & 0x30303030u); }
            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x003F003Fu), off);
            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x003F003Fu), off);
        }
    }
}

// one run of every column: activations of all NC columns loaded first, then NC independent dot products
template <ggml_type type, int NC, int k>
static __device__ __forceinline__ void mmvh_run_cols(const mmvh_raw & raw, const int c, const int kb, const half * __restrict__ xh,
                                                     const float2 * __restrict__ ss, const int ng, float * acc) {
    constexpr int NRUN = mmvh_traits<type>::nrun;
    constexpr int RL2  = 32 / NRUN;
    half2 w[RL2];
    mmvh_unpack_run<type, k>(raw, c, w);
    const int e0 = kb*256 + mmvh_run_off<type>(c, k);
    half2  xv[NC][RL2];
    float2 sk[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const int4 * xa = reinterpret_cast<const int4 *>(xh + (int64_t) j*ng*32 + e0);
#pragma unroll
        for (int v = 0; v < RL2/4; ++v) {
            *reinterpret_cast<int4 *>(&xv[j][4*v]) = xa[v];
        }
        sk[j] = ss[(int64_t) j*ng + e0/32];
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        half2 a0 = __hmul2(w[0], xv[j][0]);
        half2 a1 = __hmul2(w[1], xv[j][1]);
#pragma unroll
        for (int v = 2; v < RL2; v += 2) {
            a0 = __hfma2(w[v],     xv[j][v],     a0);
            a1 = __hfma2(w[v + 1], xv[j][v + 1], a1);
        }
        const float2 f0 = __half22float2(a0), f1 = __half22float2(a1);
        acc[j] += raw.mul[k] * sk[j].x * ((f0.x + f0.y) + (f1.x + f1.y));
        if constexpr (mmvh_traits<type>::mins) {
            acc[j] -= raw.mmin[k] * sk[j].y;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// kernel

template <ggml_type type, int R, int NC, bool HALF, bool GATE>
static __global__ void __launch_bounds__(128, 1) mmvh_kernel(
        const char * __restrict__ vx, const char * __restrict__ vg, const half * __restrict__ xh, const float2 * __restrict__ ss,
        float * __restrict__ dst, const int nrows, const int nsb, const int64_t row_bytes, const int64_t s_dst) {
    using tr = mmvh_traits<type>;
    constexpr int NRUN = tr::nrun;
    constexpr int RL2  = 32 / NRUN;           // half2 per run
    constexpr int STEP = HALF ? 4 : 8;        // super-blocks per step
    constexpr int RPW  = HALF ? 2*R : R;      // rows per warp

    const int lane = threadIdx.x;
    const int c    = lane & 3;
    const int sbi  = HALF ? ((lane >> 2) & 3) : (lane >> 2);
    const int rset = HALF ? (lane >> 4) : 0;
    const int wrow = (blockIdx.x*blockDim.y + threadIdx.y) * RPW;
    if (wrow >= nrows) {
        return; // uniform per warp
    }
    const int row0 = wrow + rset*R;
    const int ng   = nsb*8;                   // activation groups per column

    const char * xrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        xrow[r] = vx + (int64_t) min(row0 + r, nrows - 1) * row_bytes;
    }
    float acc[R][NC];
    [[maybe_unused]] float accg[R]; // FFN gate rows (GATE: dst = up . silu(gate), the SWIGLU fusion)
#pragma unroll
    for (int r = 0; r < R; ++r) {
        accg[r] = 0.0f;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    for (int kb = sbi; kb < nsb; kb += STEP) {
        if constexpr (NC == 1) {
            // one column: activations once, then each row unpacked and consumed right away
            half2  xv[32];
            float2 sk[4];
            mmvh_load_x<type>(xh, ss, ng, 0, kb, c, xv, sk);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                half2 w[32];
                float mul[4], mmin[4];
                mmvh_unpack<type>(xrow[r] + (int64_t) kb*tr::bs, c, w, mul, mmin);
                acc[r][0] += mmvh_dot<type>(w, xv, mul, mmin, sk);
                if constexpr (GATE) {
                    const char * grow = vg + (xrow[r] - vx); // same shape and stride as the up weights
                    mmvh_unpack<type>(grow + (int64_t) kb*tr::bs, c, w, mul, mmin);
                    accg[r] += mmvh_dot<type>(w, xv, mul, mmin, sk);
                }
            }
        } else {
            static_assert(R == 1 && !GATE, "mmvh: multi-column runs one row per row set, no fusion");
            mmvh_raw raw;
            mmvh_load_raw<type>(xrow[0] + (int64_t) kb*tr::bs, c, raw);
            mmvh_run_cols<type, NC, 0>(raw, c, kb, xh, ss, ng, acc[0]);
            mmvh_run_cols<type, NC, 1>(raw, c, kb, xh, ss, ng, acc[0]);
            if constexpr (tr::nrun == 4) {
                mmvh_run_cols<type, NC, 2>(raw, c, kb, xh, ss, ng, acc[0]);
                mmvh_run_cols<type, NC, 3>(raw, c, kb, xh, ss, ng, acc[0]);
            }
        }
    }

    // reduce over the lanes that share a row set: all 32 (full) or 16 (half)
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = HALF ? 8 : 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, o, 32);
            }
            acc[r][j] = v;
        }
        if constexpr (GATE) {
            float g = accg[r];
#pragma unroll
            for (int o = HALF ? 8 : 16; o > 0; o >>= 1) {
                g += __shfl_xor_sync(0xffffffff, g, o, 32);
            }
            acc[r][0] *= g / (1.0f + expf(-g)); // silu(gate) * up
        }
    }
    const int lrow = lane & (HALF ? 15 : 31);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (lrow == r*NC + j && row0 + r < nrows) {
                dst[(int64_t) j*s_dst + row0 + r] = acc[r][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// v4 multi-column (2026-10-08, MMVH_MC >= 2). v3 above runs one row per row set, so every row re-reads all columns'
// activations: ~8 bytes of L1/L2 traffic per weight at 4 columns against ~0.7 bytes of weights, and it lost to MMVQ.
// Here each row set owns R rows: a run's activations for all NC columns are loaded once and used for all R rows, and
// the 4 warps of a block walk the same super-blocks, so the activations mostly come from L1. The per-lane raw block is
// compacted (Q5_K keeps only this lane's two high bits per byte) to leave registers for R rows.

struct mmvh_raw4 {
    uint32_t q[8];   // Q4_K/Q5_K: qs words; Q6_K: ql0[4], ql1[4]
    uint32_t h[4];   // Q5_K: bits 2c and 2c+1 of qh, two words per entry; Q6_K: qh[4]
    float    mul[4];
    float    mmin[4];
};

template <ggml_type type>
static __device__ __forceinline__ void mmvh_load_raw4(const char * __restrict__ b, const int c, mmvh_raw4 & raw) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr bool q5     = type == GGML_TYPE_Q5_K;
        constexpr int  qs_off = q5 ? 48 : 16;
        const int4 hdr = *reinterpret_cast<const int4 *>(b);
        const int4 q0 = *reinterpret_cast<const int4 *>(b + qs_off + 32*c);
        const int4 q1 = *reinterpret_cast<const int4 *>(b + qs_off + 32*c + 16);
        raw.q[0] = q0.x; raw.q[1] = q0.y; raw.q[2] = q0.z; raw.q[3] = q0.w;
        raw.q[4] = q1.x; raw.q[5] = q1.y; raw.q[6] = q1.z; raw.q[7] = q1.w;
        if constexpr (q5) {
            const int4 h0 = *reinterpret_cast<const int4 *>(b + 16);
            const int4 h1 = *reinterpret_cast<const int4 *>(b + 32);
            const uint32_t hw[8] = {(uint32_t) h0.x, (uint32_t) h0.y, (uint32_t) h0.z, (uint32_t) h0.w,
                                    (uint32_t) h1.x, (uint32_t) h1.y, (uint32_t) h1.z, (uint32_t) h1.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                raw.h[i] = ((hw[2*i] >> (2*c)) & 0x03030303u) | (((hw[2*i + 1] >> (2*c)) & 0x03030303u) << 2);
            }
        }
        const uint32_t S0 = (uint32_t) hdr.y, S1 = (uint32_t) hdr.z, S2 = (uint32_t) hdr.w;
        const bool hi_form = c >= 2;
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&hdr.x));
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int sh = 8*((2*c + k) & 3);
            const uint32_t a = S0 >> sh, bb = S1 >> sh, e = S2 >> sh;
            const int sc = hi_form ? (int) ((e & 0xF) | ((a  >> 2) & 0x30))        : (int) (a  & 63);
            const int m  = hi_form ? (int) (((e >> 4) & 0xF) | ((bb >> 2) & 0x30)) : (int) (bb & 63);
            raw.mul[k]  = dm.x * (float) sc;
            raw.mmin[k] = dm.y * (float) m;
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const int n = c >> 1, lh = c & 1;
        mmvh_load16_a2(b +       64*n +      16*lh, raw.q);
        mmvh_load16_a2(b +       64*n + 32 + 16*lh, raw.q + 4);
        mmvh_load16_a2(b + 128 + 32*n +      16*lh, raw.h);
        const int8_t * sc = (const int8_t *) (b + 192 + 8*n) + lh;
        const float d = __half2float(*reinterpret_cast<const half *>(b + 208));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            raw.mul[k]  = d * (float) sc[2*k];
            raw.mmin[k] = 0.0f;
        }
    } else { // IQ4_XS: lane c owns qs[32c .. 32c+31] = sub-blocks 2c, 2c+1 (8-byte aligned blocks)
        const uint2 hdr = *reinterpret_cast<const uint2 *>(b);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint2 q2 = *reinterpret_cast<const uint2 *>(b + 8 + 32*c + 8*i);
            raw.q[2*i] = q2.x;
            raw.q[2*i + 1] = q2.y;
        }
        const float    d   = __half2float(*reinterpret_cast<const half *>(&hdr.x));
        const uint32_t sh_ = hdr.x >> 16;            // scales_h
        const uint32_t sl  = (hdr.y >> (8*c)) & 0xFF; // scales_l[c]: low nibble sub-block 2c, high 2c+1
        const int ls0 = (int) ((sl & 0xF) | (((sh_ >> (4*c))     & 3) << 4)) - 32;
        const int ls1 = (int) ((sl >> 4)  | (((sh_ >> (4*c + 2)) & 3) << 4)) - 32;
        raw.mul[0] = raw.mul[1] = d * (float) ls0;
        raw.mul[2] = raw.mul[3] = d * (float) ls1;
        raw.mmin[0] = raw.mmin[1] = raw.mmin[2] = raw.mmin[3] = 0.0f;
    }
}

// run k of the compact raw block as half2 (16 for Q4_K/Q5_K, 8 for Q6_K), offset removed
template <ggml_type type, int k>
static __device__ __forceinline__ void mmvh_unpack_run4(const mmvh_raw4 & raw, half2 * w) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const half2 off = __float2half2_rn(1024.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t v = k == 0 ? (raw.q[i] & 0x0F0F0F0Fu) : ((raw.q[i] >> 4) & 0x0F0F0F0Fu);
            if constexpr (type == GGML_TYPE_Q5_K) {
                // word i's bits sit at 2*(i & 1) in entry i/2: bit k of that pair is the 5th bit of run k
                v |= ((raw.h[i >> 1] >> (2*(i & 1) + k)) << 4) & 0x10101010u;
            }
            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x001F001Fu), off);
            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x001F001Fu), off);
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) { // Q6_K: ql0 = raw.q[0..3], ql1 = raw.q[4..7], qh = raw.h[0..3]
        const half2 off = __float2half2_rn(1056.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t h = raw.h[i];
            uint32_t v;
            if constexpr (k == 0) { v = ( raw.q[i]           & 0x0F0F0F0Fu) | ((h << 4) & 0x30303030u); }
            if constexpr (k == 1) { v = ( raw.q[4 + i]       & 0x0F0F0F0Fu) | ((h << 2) & 0x30303030u); }
            if constexpr (k == 2) { v = ((raw.q[i]     >> 4) & 0x0F0F0F0Fu) | ( h       & 0x30303030u); }
            if constexpr (k == 3) { v = ((raw.q[4 + i] >> 4) & 0x0F0F0F0Fu) | ((h >> 2) & 0x30303030u); }
            w[2*i + 0] = __hsub2(mmvh_h2( v       & 0x003F003Fu), off);
            w[2*i + 1] = __hsub2(mmvh_h2((v >> 8) & 0x003F003Fu), off);
        }
    } else { // IQ4_XS: run k = low (k even) / high (k odd) nibbles of words 4*(k/2) .. +3, looked up in the codebook
        const half2 off = __float2half2_rn(1152.0f); // 1024 + the table's +128
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t lo, hi;
            mmvh_iq4_lut(raw.q[4*(k >> 1) + j], lo, hi);
            const uint32_t v = (k & 1) ? hi : lo;
            w[2*j + 0] = __hsub2(mmvh_h2( v       & 0x00FF00FFu), off);
            w[2*j + 1] = __hsub2(mmvh_h2((v >> 8) & 0x00FF00FFu), off);
        }
    }
}

// run k for every row and column: the NC columns' activations are loaded once, then each row is unpacked and used
template <ggml_type type, int R, int NC, int k>
static __device__ __forceinline__ void mmvh_mc_run(const mmvh_raw4 (&raw)[R], const int c, const int kb,
                                                   const half * __restrict__ xh, const float2 * __restrict__ ss,
                                                   const int ng, float (&acc)[R][NC]) {
    constexpr int NRUN = mmvh_traits<type>::nrun;
    constexpr int RL2  = 32 / NRUN;
    const int e0 = kb*256 + mmvh_run_off<type>(c, k);
    half2  xv[NC][RL2];
    float2 sk[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const int4 * xa = reinterpret_cast<const int4 *>(xh + (int64_t) j*ng*32 + e0);
#pragma unroll
        for (int v = 0; v < RL2/4; ++v) {
            *reinterpret_cast<int4 *>(&xv[j][4*v]) = xa[v];
        }
        sk[j] = ss[(int64_t) j*ng + e0/32];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        half2 w[RL2];
        mmvh_unpack_run4<type, k>(raw[r], w);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            half2 a0 = __hmul2(w[0], xv[j][0]);
            half2 a1 = __hmul2(w[1], xv[j][1]);
#pragma unroll
            for (int v = 2; v < RL2; v += 2) {
                a0 = __hfma2(w[v],     xv[j][v],     a0);
                a1 = __hfma2(w[v + 1], xv[j][v + 1], a1);
            }
            const float2 f0 = __half22float2(a0), f1 = __half22float2(a1);
            acc[r][j] += raw[r].mul[k] * sk[j].x * ((f0.x + f0.y) + (f1.x + f1.y));
            if constexpr (mmvh_traits<type>::mins) {
                acc[r][j] -= raw[r].mmin[k] * sk[j].y;
            }
        }
    }
}

template <ggml_type type, int R, int NC, bool HALF, int MINB>
static __global__ void __launch_bounds__(128, MINB) mmvh_mc_kernel(
        const char * __restrict__ vx, const half * __restrict__ xh, const float2 * __restrict__ ss,
        float * __restrict__ dst, const int nrows, const int nsb, const int64_t row_bytes, const int64_t s_dst) {
    using tr = mmvh_traits<type>;
    constexpr int STEP = HALF ? 4 : 8;        // super-blocks per step
    constexpr int RPW  = HALF ? 2*R : R;      // rows per warp
    static_assert(R*NC <= (HALF ? 16 : 32), "mmvh v4: one writing lane per output");

    const int lane = threadIdx.x;
    const int c    = lane & 3;
    const int sbi  = HALF ? ((lane >> 2) & 3) : (lane >> 2);
    const int rset = HALF ? (lane >> 4) : 0;
    const int wrow = (blockIdx.x*blockDim.y + threadIdx.y) * RPW;
    if (wrow >= nrows) {
        return; // uniform per warp
    }
    const int row0 = wrow + rset*R;
    const int ng   = nsb*8;

    const char * xrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        xrow[r] = vx + (int64_t) min(row0 + r, nrows - 1) * row_bytes;
    }
    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    for (int kb = sbi; kb < nsb; kb += STEP) {
        mmvh_raw4 raw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            mmvh_load_raw4<type>(xrow[r] + (int64_t) kb*tr::bs, c, raw[r]);
        }
        mmvh_mc_run<type, R, NC, 0>(raw, c, kb, xh, ss, ng, acc);
        mmvh_mc_run<type, R, NC, 1>(raw, c, kb, xh, ss, ng, acc);
        if constexpr (tr::nrun == 4) {
            mmvh_mc_run<type, R, NC, 2>(raw, c, kb, xh, ss, ng, acc);
            mmvh_mc_run<type, R, NC, 3>(raw, c, kb, xh, ss, ng, acc);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = HALF ? 8 : 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, o, 32);
            }
            acc[r][j] = v;
        }
    }
    const int lrow = lane & (HALF ? 15 : 31);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (lrow == r*NC + j && row0 + r < nrows) {
                dst[(int64_t) j*s_dst + row0 + r] = acc[r][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// dispatch

static bool mmvh_type_ok(const ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ4_XS;
}

bool ggml_cuda_mmvh_pascal_type_ok(const ggml_type t) {
    return mmvh_type_ok(t) && (t != GGML_TYPE_IQ4_XS || ggml_cuda_tune.mmvh_iq4);
}

bool ggml_cuda_mmvh_pascal_fusable(const ggml_tensor * mm) {
    return ggml_cuda_tune.mmvh > 0 && ggml_cuda_tune.mmvh_fuse && mm->op == GGML_OP_MUL_MAT &&
           ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_PASCAL &&
           ggml_cuda_mmvh_pascal_type_ok(mm->src[0]->type) && mm->src[1]->ne[1] == 1;
}

bool ggml_cuda_mmvh_pascal_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                     const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    if (ggml_cuda_tune.mmvh <= 0 || ids != nullptr) {
        return false;
    }
    if (fusion != nullptr) {
        // only the plain FFN SWIGLU pair: gate weights laid out exactly like the up weights, no biases or scales
        if (!ggml_cuda_tune.mmvh_fuse || fusion->gate == nullptr || fusion->x_bias || fusion->gate_bias ||
            fusion->x_scale || fusion->gate_scale || fusion->glu_op != GGML_GLU_OP_SWIGLU ||
            fusion->gate->type != src0->type || fusion->gate->nb[1] != src0->nb[1] ||
            fusion->gate->ne[0] != src0->ne[0] || fusion->gate->ne[1] != src0->ne[1] ||
            ((uintptr_t) fusion->gate->data) % 16 != 0) {
            return false;
        }
    }
    // GP100 only: GP102/GP104 (sm_61) run HFMA2 at 1/64 rate
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    if (!mmvh_type_ok(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t nc = src1->ne[1];
    if (nc == 1 && !ggml_cuda_mmvh_pascal_type_ok(src0->type)) {
        return false; // IQ4_XS one column: MMVH_IQ4 (slower than MMVQ, off)
    }
    const bool iq4_mc_ok = src0->type != GGML_TYPE_IQ4_XS || (ggml_cuda_tune.mmvh_mc >= 2 && ggml_cuda_tune.mmvh_iq4_mc);
    if (nc < 1 || nc > 4 || (nc > 1 && (!ggml_cuda_tune.mmvh_mc || !iq4_mc_ok || fusion != nullptr)) ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    const size_t align = src0->type == GGML_TYPE_Q6_K ? 4 : src0->type == GGML_TYPE_IQ4_XS ? 8 : 16;
    if (src0->ne[0] % QK_K != 0 || src0->nb[1] % align != 0 || ((uintptr_t) src0->data) % align != 0 ||
        src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || dst->ne[0] != src0->ne[1]) {
        return false;
    }
    return true;
}

// Prepared activations are reused when the very previous node on this device prepared the same src1 on the same
// stream (FFN gate/up, GDN qkv/z/beta/alpha, attention q/k/v read one input). Any node in between invalidates it,
// so a reused buffer address can never hand back stale data.
struct mmvh_prep_cache {
    const void * src = nullptr;
    int64_t      ne10 = 0, ne11 = 0, nb11 = 0;
    cudaStream_t stream = nullptr;
    uint64_t     seq = 0;
    half *       xh = nullptr;
    float2 *     ss = nullptr;
    size_t       cap = 0; // elements of xh
};
static mmvh_prep_cache g_mmvh_cache[GGML_CUDA_MAX_DEVICES];

template <ggml_type type, int R, int NC, bool GATE>
static void mmvh_launch_t(const char * vx, const char * vg, const half * xh, const float2 * ss, float * d, int nrows, int nsb,
                          int64_t rb, int64_t s_dst, cudaStream_t stream) {
    // half-warp mode when it removes the idle lanes of the last step (K/256 = 20 -> 5 full steps); full otherwise
    const int  mode = ggml_cuda_tune.mmvh / 10; // 0 auto, 1 full, 2 half
    const bool half_mode = mode == 2 || (mode == 0 && nsb % 8 != 0 && nsb % 4 == 0);
    const int  rpw = half_mode ? 2*R : R;
    const int  nb  = (nrows + 4*rpw - 1) / (4*rpw);
    if (half_mode) {
        mmvh_kernel<type, R, NC, true,  GATE><<<nb, dim3(32, 4), 0, stream>>>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst);
    } else {
        mmvh_kernel<type, R, NC, false, GATE><<<nb, dim3(32, 4), 0, stream>>>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst);
    }
}

template <ggml_type type, int R, int NC, int MINB>
static void mmvh_mc_launch_t(const char * vx, const half * xh, const float2 * ss, float * d, int nrows, int nsb,
                             int64_t rb, int64_t s_dst, cudaStream_t stream) {
    const int  mode = ggml_cuda_tune.mmvh / 10; // 0 auto, 1 full, 2 half (as for one column)
    const bool half_mode = mode == 2 || (mode == 0 && nsb % 8 != 0 && nsb % 4 == 0);
    const int  rpw = half_mode ? 2*R : R;
    const int  nb  = (nrows + 4*rpw - 1) / (4*rpw);
    if (half_mode) {
        mmvh_mc_kernel<type, R, NC, true,  MINB><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);
    } else {
        mmvh_mc_kernel<type, R, NC, false, MINB><<<nb, dim3(32, 4), 0, stream>>>(vx, xh, ss, d, nrows, nsb, rb, s_dst);
    }
}

template <ggml_type type, int NC>
static void mmvh_mc_launch_r(int r, const char * vx, const half * xh, const float2 * ss, float * d, int nrows, int nsb,
                             int64_t rb, int64_t s_dst, cudaStream_t stream) {
    // MMVH_MC: rows per row set, or rows*10 + min blocks per SM (register cap for occupancy)
    switch (r) {
        case 2:  mmvh_mc_launch_t<type, 2, NC, 1>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        case 24: mmvh_mc_launch_t<type, 2, NC, 4>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        case 32: mmvh_mc_launch_t<type, 3, NC, 2>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        case 33: mmvh_mc_launch_t<type, 3, NC, 3>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        case 43: mmvh_mc_launch_t<type, 4, NC, 3>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        default: mmvh_mc_launch_t<type, 4, NC, 1>(vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
    }
}

template <ggml_type type>
static void mmvh_launch_type(int nc, int rsel, const char * vx, const char * vg, const half * xh, const float2 * ss, float * d,
                             int nrows, int nsb, int64_t rb, int64_t s_dst, cudaStream_t stream) {
    if (nc == 1) {
        if (vg) {
            if (rsel >= 2) { mmvh_launch_t<type, 2, 1, true >(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); }
            else           { mmvh_launch_t<type, 1, 1, true >(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); }
        } else {
            if (rsel >= 2) { mmvh_launch_t<type, 2, 1, false>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); }
            else           { mmvh_launch_t<type, 1, 1, false>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); }
        }
        return;
    }
    if (ggml_cuda_tune.mmvh_mc >= 2) { // v4 (any of the four types)
        const int r = ggml_cuda_tune.mmvh_mc;
        switch (nc) {
            case 2:  mmvh_mc_launch_r<type, 2>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
            case 3:  mmvh_mc_launch_r<type, 3>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
            default: mmvh_mc_launch_r<type, 4>(r, vx, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        }
        return;
    }
    if constexpr (type != GGML_TYPE_IQ4_XS) {
        switch (nc) {
            case 2:  mmvh_launch_t<type, 1, 2, false>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
            case 3:  mmvh_launch_t<type, 1, 3, false>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
            default: mmvh_launch_t<type, 1, 4, false>(vx, vg, xh, ss, d, nrows, nsb, rb, s_dst, stream); break;
        }
    }
}

void ggml_cuda_mmvh_pascal(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                           const ggml_cuda_mm_fusion_args_host * fusion) {
    const int64_t K     = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int     nc    = (int) src1->ne[1];
    const int     ng    = (int) (K/32);
    cudaStream_t stream = ctx.stream();

    mmvh_prep_cache & pc = g_mmvh_cache[ctx.device];
    const uint64_t seq = ggml_cuda_tune_node_seq[ctx.device];
    const bool reuse = pc.src == src1->data && pc.ne10 == K && pc.ne11 == nc && pc.nb11 == (int64_t) src1->nb[1] &&
                       pc.stream == stream && pc.seq + 1 == seq && pc.xh != nullptr;
    if (!reuse) {
        const size_t need = (size_t) K * nc;
        if (need > pc.cap) {
            if (pc.xh) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaFree(pc.xh));
                CUDA_CHECK(cudaFree(pc.ss));
            }
            pc.cap = std::max<size_t>(need, 32768);
            CUDA_CHECK(cudaMalloc(&pc.xh, pc.cap*sizeof(half)));
            CUDA_CHECK(cudaMalloc(&pc.ss, pc.cap/32*sizeof(float2)));
        }
        mmvh_prep_x<<<dim3((ng + 3)/4, nc), dim3(32, 4), 0, stream>>>(
            (const float *) src1->data, pc.xh, pc.ss, ng, (int64_t) (src1->nb[1]/sizeof(float)));
        pc.src  = src1->data;
        pc.ne10 = K;
        pc.ne11 = nc;
        pc.nb11 = (int64_t) src1->nb[1];
        pc.stream = stream;
    }
    pc.seq = seq;

    const int nsb  = (int) (K/QK_K);
    const int rsel = ggml_cuda_tune.mmvh % 10;
    const char * vx = (const char *) src0->data;
    const char * vg = fusion ? (const char *) fusion->gate->data : nullptr;
    float * d = (float *) dst->data;
    const int64_t rb    = (int64_t) src0->nb[1];
    const int64_t s_dst = (int64_t) (dst->nb[1]/sizeof(float));
    switch (src0->type) {
        case GGML_TYPE_Q4_K:   mmvh_launch_type<GGML_TYPE_Q4_K>  (nc, rsel, vx, vg, pc.xh, pc.ss, d, (int) nrows, nsb, rb, s_dst, stream); break;
        case GGML_TYPE_Q5_K:   mmvh_launch_type<GGML_TYPE_Q5_K>  (nc, rsel, vx, vg, pc.xh, pc.ss, d, (int) nrows, nsb, rb, s_dst, stream); break;
        case GGML_TYPE_Q6_K:   mmvh_launch_type<GGML_TYPE_Q6_K>  (nc, rsel, vx, vg, pc.xh, pc.ss, d, (int) nrows, nsb, rb, s_dst, stream); break;
        default:               mmvh_launch_type<GGML_TYPE_IQ4_XS>(nc, rsel, vx, vg, pc.xh, pc.ss, d, (int) nrows, nsb, rb, s_dst, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
}
