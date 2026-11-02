#!/bin/bash
#SBATCH --job-name=llm-build-pascal
#SBATCH --partition=gpu
#SBATCH --nodelist=slerm-node04
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/data/llm/logs/buildpascal-%j.out

# Builds llama.cpp for the Tesla P100s (GP100, sm_60), which replaced the
# 1660 SUPER and 980 Ti on 2026-10-04 (those cards are gone for good).
# CUDA 12.2 still emits sm_60 (CUDA 13 dropped Pascal) and matches node04's
# driver 535.309. Same source and flags as build.sh, only the arch differs.
# Runs on node04 itself (6 cores, -j 5 to stay inside 8G): the compute nodes
# are usually powered off by slerm-orch.
#
# GGML_NATIVE=OFF matters: pin the ISA to the Haswell baseline (AVX2/FMA/F16C)
# that every host here shares.

set -euo pipefail
source /data/llm/bin/llm-env.sh

SRC=$LLM_ROOT/src/llama.cpp
BUILD=$LLM_ROOT/build-pascal

echo "host: $(hostname)  cores: ${SLURM_CPUS_ON_NODE:-?}"
nvcc --version | tail -2

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60" \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build "$BUILD" -j 5

echo "=== built binaries ==="
ls -1 "$BUILD/bin" | head -30
