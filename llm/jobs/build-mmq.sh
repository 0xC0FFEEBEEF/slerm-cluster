#!/bin/bash
#SBATCH --job-name=llm-build-mmq
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/data/llm/logs/buildmmq-%j.out

# Builds llama.cpp ONCE for BOTH cards.
#   sm_52 = GTX 980 Ti (node04, Maxwell)
#   sm_75 = GTX 1660 SUPER (node03, Turing)
# CUDA 12.2 is the newest toolkit that still emits sm_52 AND stays within
# node04 driver 535.309 (CUDA 12.2). CUDA 13 dropped Maxwell entirely.
#
# GGML_NATIVE=OFF matters: this compiles on a compute node but runs on the GPU
# nodes. Pin the ISA to the Haswell baseline (AVX2/FMA/F16C) that every host
# here shares, so the binary cannot pick up instructions node03/04 lack.

set -euo pipefail
source /data/llm/bin/llm-env.sh

SRC=$LLM_ROOT/src/llama.cpp
BUILD=$LLM_ROOT/build-mmq

echo "host: $(hostname)  cores: ${SLURM_CPUS_ON_NODE:-?}"
nvcc --version | tail -2

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES="52;61-virtual" \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build "$BUILD" -j "${SLURM_CPUS_ON_NODE:-8}"

echo "=== built binaries ==="
ls -1 "$BUILD/bin" | head -30
