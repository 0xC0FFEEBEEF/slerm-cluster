#pragma once

#include "common.cuh"

// Live GPU timeline for the P100 work (2026-10-08). A line PROF=N in GGML_CUDA_TUNE_FILE makes the next N graph
// segments per device record a CUDA event before every computed node, around each AllReduce and at segment ends,
// then appends a per-key table to /var/tmp/ggml-prof-<pid>.txt (a decode pass is ~130 segments with tensor split).
// Keys starting with '~' are not kernels: ~allreduce = AllReduce incl. waiting for the peer GPU; ~after_ar,
// ~seg_begin and ~tail = GPU time until the next recorded work, i.e. idle or launch latency (~tail>2ms = pass
// boundaries, where the host samples and builds the next graph). Unarmed cost: one int test per node.
extern int ggml_cuda_prof_gen;   // bumped by every PROF= line read; 0 = never armed
extern int ggml_cuda_prof_live;  // devices currently recording

void ggml_cuda_prof_arm(int n_segments);
void ggml_cuda_prof_segment_begin(int device, cudaStream_t stream);
void ggml_cuda_prof_segment_end(int device, cudaStream_t stream);
void ggml_cuda_prof_node(int device, cudaStream_t stream, const ggml_tensor * node);
void ggml_cuda_prof_mark(int device, cudaStream_t stream, const char * key);
