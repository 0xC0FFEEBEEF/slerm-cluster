#!/bin/bash
#SBATCH --job-name=llm-build-cpu
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/data/llm/logs/buildcpu-%j.out

# CPU-only build into build-cpu/. The CUDA build links libcuda.so.1, which only
# exists on the GPU nodes, so it cannot run on compute at all. This variant lets
# the same batch pipeline run on the 32 compute cores when the GPU nodes are
# busy or down. Same Haswell ISA baseline as the CUDA build.

set -euo pipefail
source /data/llm/bin/llm-env.sh

cmake -S "$LLM_ROOT/src/llama.cpp" -B "$LLM_ROOT/build-cpu" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build "$LLM_ROOT/build-cpu" -j "${SLURM_CPUS_ON_NODE:-8}"
echo "=== done ==="; ls -1 "$LLM_ROOT/build-cpu/bin" | grep -E "^llama-(server|cli|bench)$"
