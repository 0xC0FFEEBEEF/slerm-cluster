#pragma once

#include "common.cuh"

// One-token mat-vec on HFMA2 for GP100 (P100) for Q4_K/Q5_K/Q6_K (+IQ4_XS behind MMVH_IQ4), behind the MMVH knob
// (GGML_CUDA_MMVH_PASCAL=1|2 rows per row set, +10 full-warp, +20 half-warp, or the tune file). Also takes the FFN
// SWIGLU gate/up fusion (MMVH_FUSE). See mmvh-pascal.cu.
bool ggml_cuda_mmvh_pascal_type_ok(ggml_type t);
bool ggml_cuda_mmvh_pascal_fusable(const ggml_tensor * mm);
bool ggml_cuda_mmvh_pascal_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                     const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
void ggml_cuda_mmvh_pascal(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                           const ggml_cuda_mm_fusion_args_host * fusion);
