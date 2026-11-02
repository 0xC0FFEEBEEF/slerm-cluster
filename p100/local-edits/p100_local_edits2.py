#!/usr/bin/env python3
"""Second round of local P100 changes on llama.cpp v0.5.0 (on top of p100_local_edits.py).

5. Internal AllReduce (tensor split) on pre-Volta when GGML_CUDA_AR_PASCAL=1. Upstream refuses it below Volta only
   because the spin-wait backs off with __nanosleep; on Pascal the spin runs without it. Otherwise every
   all-reduce falls back to the meta backend's host-staged copies (~10 ms of idle per Qwen3.8-27B token).
6. MMVQ block shape for one-token mat-vec on Pascal selectable at runtime for tuning:
   GGML_CUDA_MMVQ_PASCAL_CFG="S,L" picks a (warps, rows per block) preset for K <= 6144 (S) and longer K (L);
   0 = the compiled default (4 warps, GGML_CUDA_MMVQ_PASCAL_RPB rows). Only the types Qwen3.8 uses get presets.

Usage: p100_local_edits2.py <llama.cpp source dir>. Each edit must match exactly once.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "ggml/src/ggml-cuda"


def edit(name, old, new):
    p = root / name
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        sys.exit(f"{name}: expected 1 match, found {n}:\n{old[:300]}")
    p.write_text(s.replace(old, new))
    print(f"ok  {name}")


# 5. AllReduce on Pascal
edit("allreduce.cu",
"""#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // GGML_USE_HIP
""",
"""#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#elif defined(__CUDA_ARCH__)
            // Pascal has no __nanosleep. Each poll is an uncached read of host memory over PCIe, which paces the loop.
#else
            NO_DEVICE_CODE;
#endif // GGML_USE_HIP
""")

edit("allreduce.cu",
"""    // The chunked kernel uses __nanosleep (NVIDIA, sm70+) or
    // __builtin_amdgcn_s_sleep (AMD).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
""",
"""    // The chunked kernel uses __nanosleep (NVIDIA, sm70+) or
    // __builtin_amdgcn_s_sleep (AMD). GGML_CUDA_AR_PASCAL=1 allows Pascal, which spins without it.
    static const bool ar_pascal = getenv("GGML_CUDA_AR_PASCAL") != nullptr && atoi(getenv("GGML_CUDA_AR_PASCAL"));
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA && !(ar_pascal && cc >= GGML_CUDA_CC_PASCAL)) {
""")

# 6. runtime-selectable MMVQ block shape on Pascal
edit("mmvq.cu",
"""#ifndef GGML_CUDA_MMVQ_PASCAL_RPB
#define GGML_CUDA_MMVQ_PASCAL_RPB 4
#endif
""",
"""#ifndef GGML_CUDA_MMVQ_PASCAL_RPB
#define GGML_CUDA_MMVQ_PASCAL_RPB 4
#endif

// Tuning presets for one-token mat-vec on Pascal, chosen at runtime by GGML_CUDA_MMVQ_PASCAL_CFG ("S,L").
// 0 is the compiled default; 1..7 = (warps, rows per block) below.
static constexpr __host__ __device__ int mmvq_pascal_cfg_nwarps(int c) {
    return c == 1 ? 4 : c == 2 ? 4 : c == 3 ? 2 : c == 4 ? 2 : c == 5 ? 8 : c == 6 ? 8 : c == 7 ? 1 : 0;
}
static constexpr __host__ __device__ int mmvq_pascal_cfg_rpb(int c) {
    return c == 1 ? 2 : c == 2 ? 8 : c == 3 ? 4 : c == 4 ? 8 : c == 5 ? 2 : c == 6 ? 4 : c == 7 ? 4 : 0;
}
static constexpr bool mmvq_pascal_tunable(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K;
}
""")

edit("mmvq.cu",
"""template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
""",
"""template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int pcfg = 0>
__launch_bounds__((pcfg ? mmvq_pascal_cfg_nwarps(pcfg) : calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters))*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
""")

edit("mmvq.cu",
"""    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
""",
"""    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = pcfg ? mmvq_pascal_cfg_nwarps(pcfg) : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = pcfg ? mmvq_pascal_cfg_rpb(pcfg) : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
""")

edit("mmvq.cu",
"""        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
""",
"""        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false,
        const int pcfg = 0) {
    const int nwarps = pcfg ? mmvq_pascal_cfg_nwarps(pcfg) : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = pcfg ? mmvq_pascal_cfg_rpb(pcfg) : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
""")

edit("mmvq.cu",
"""template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
""",
"""template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int pcfg = 0>
static void mul_mat_vec_q_switch_fusion(
""")

edit("mmvq.cu",
"""            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
""",
"""            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, pcfg>, launch_params,
""")

edit("mmvq.cu",
"""    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
""",
"""    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, pcfg>, launch_params,
""")

edit("mmvq.cu",
"""            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
""",
"""            const auto launch = [&](auto small_k_tag, auto halve_iters_tag, auto pcfg_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                constexpr int  c_pcfg    = decltype(pcfg_tag)::value;
""")

edit("mmvq.cu",
"""                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
""",
"""                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters, c_pcfg);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters, c_pcfg>(
""")

edit("mmvq.cu",
"""            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
""",
"""            using pcfg0 = std::integral_constant<int, 0>;
            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{}, pcfg0{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{}, pcfg0{});
            } else if constexpr (mmvq_pascal_tunable(type)) {
                static const std::pair<int, int> env_cfg = [] {
                    int s = 0, l = 0;
                    const char * e = getenv("GGML_CUDA_MMVQ_PASCAL_CFG");
                    if (e != nullptr && sscanf(e, "%d,%d", &s, &l) < 2) {
                        l = s;
                    }
                    return std::make_pair(s, l);
                }();
                const int pcfg = table_id == MMVQ_PARAMETERS_PASCAL && !has_ids ?
                    (ncols_x <= 6144 ? env_cfg.first : env_cfg.second) : 0;
                switch (pcfg) {
                    case 1:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 1>{}); break;
                    case 2:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 2>{}); break;
                    case 3:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 3>{}); break;
                    case 4:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 4>{}); break;
                    case 5:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 5>{}); break;
                    case 6:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 6>{}); break;
                    case 7:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 7>{}); break;
                    default: launch(std::false_type{}, std::false_type{}, pcfg0{}); break;
                }
            } else {
                launch(std::false_type{}, std::false_type{}, pcfg0{});
            }
        } break;
""")
