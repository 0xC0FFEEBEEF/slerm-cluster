#pragma once

// Pascal (P100) switches, see ggml_cuda_tune_refresh() in ggml-cuda.cu. ar: internal AllReduce on pre-Volta;
// ar_bf16: BF16 wire threshold in bytes (-1 = GGML_CUDA_AR_BF16_THRESHOLD/default); graphs: CUDA graphs on pre-Volta;
// mmvq_s/mmvq_l: one-token mat-vec presets for K <= 6144 / longer K; mmvh: Q4_K/Q5_K HFMA2 mat-vec on GP100
// (0 off, else rows per warp 1/2/4).
struct ggml_cuda_tune_knobs {
    int mmvh    = 0;
    int mmvh_iq4 = 0; // IQ4_XS through mmvh too (measured slower than MMVQ, off)
    int mmvh_fuse = 1; // FFN SWIGLU gate/up fusion through mmvh
    int mmvh_mc   = 1; // 2-4 token batches (MTP verify) through mmvh: 1 v3, 2/4 v4 rows per row set, rows*10+min blocks
    int mmvh_mc_q6  = 0; // Q6_K override of mmvh_mc (0 = same)
    int mmvh_iq4_mc = 0; // IQ4_XS 2-4 token batches through v4/v5 (when mmvh_mc >= 2); v4 measured slower end to end
    int ar      = 0;
    int ar_bf16 = -1;
    int graphs  = 0;
    int mmvq_s  = 0;
    int mmvq_l  = 0;
    int fa_gqa_pad = 0; // FA tile kernel: pack 4 or 8 Q heads per K/V head when no power of two divides the GQA ratio
    int mmvq_id     = 0; // 2026-10-17: Pascal preset (1..10) for one-token MUL_MAT_ID of the IQ/Q2_0 expert types
    int mmvf_shortk = 0; // 2026-10-17: rows per warp (1/2/4/8) for one-token F32/F16/BF16 mat-vec with K <= 1024
};
extern ggml_cuda_tune_knobs ggml_cuda_tune;
// Bumped for every node a CUDA backend computes (fused or not), per device: lets mmvh reuse prepared activations only
// when nothing ran in between.
extern uint64_t ggml_cuda_tune_node_seq[GGML_CUDA_MAX_DEVICES];
// Whether GGML_CUDA_TUNE_FILE is set: AllReduce is then initialized on Pascal so the file can switch it at runtime.
bool ggml_cuda_tune_file_active();
void ggml_cuda_tune_refresh();
