#!/usr/bin/env python3
"""Fourth round of local P100 changes: strength-reduced block addressing in the one-token mat-vec loop on Pascal.

Pascal has no 32-bit integer multiply (each is 3 XMAD). The Q5_K mmvq loop spent 124 of its 525 instructions on
XMAD, mostly re-deriving every block address from kbx per iteration. On pre-Volta NVIDIA (non-fused kernels) the
loop now keeps one pointer per row and steps it by a constant. Same arithmetic, bit-identical results.

Usage: p100_local_edits4.py <llama.cpp source dir>. Each edit must match exactly once.
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


edit("mmvq.cu",
"""    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
""",
"""    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    constexpr bool pascal_ptr_loop = !has_fusion;
#else
    constexpr bool pascal_ptr_loop = false;
#endif
    if constexpr (pascal_ptr_loop) {
        // Pascal has no 32-bit integer multiply (3 XMAD each): step one pointer per row instead of re-deriving every
        // block address from kbx each iteration, which was about a quarter of the Q5_K loop's instructions.
        constexpr int    kyb = qk/QK8_1; // q8_1 blocks per x block
        constexpr size_t bs  = ggml_cuda_type_traits<type>::bs;
        const int kbx0 = tid / (qi/vdr);
        const int kqs  = vdr * (tid % (qi/vdr));
        const char * xp[rows_per_cuda_block];
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            xp[i] = (const char *) vx + (size_t) (kbx_offset + i*stride_row_x + kbx0) * bs;
        }
        const block_q8_1 * yp = y + kbx0*kyb;
        for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(xp[i], yp + j*stride_col_y, 0, kqs);
                }
            }
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                xp[i] += blocks_per_iter*bs;
            }
            yp += blocks_per_iter*kyb;
        }
    } else {
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
""")

edit("mmvq.cu",
"""                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
""",
"""                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }
    } // pascal_ptr_loop

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
""")
