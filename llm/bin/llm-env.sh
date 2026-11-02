#!/bin/bash
# Shared environment for every llama.cpp job on slerm.
# Sourced by job scripts; safe to source from an interactive shell too.
export LLM_ROOT=/data/llm
export CUDA_HOME=$LLM_ROOT/toolchain/cuda122

# Which build to run. Default is the CUDA build, which REQUIRES a GPU node --
# it links libcuda.so.1, which does not exist on the compute nodes. Set
#   LLM_BUILD=/data/llm/build-cpu
# to run the same jobs on the compute partition instead.
# build-pascal (sm_60) since 2026-10-04: both GPUs are Tesla P100s. build/ is the old sm_52/75 one.
export LLM_BUILD=${LLM_BUILD:-$LLM_ROOT/build-pascal}

export PATH=$LLM_BUILD/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLM_BUILD/bin:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Default model + serving port. Override by exporting before submit.
export LLM_MODEL=${LLM_MODEL:-$LLM_ROOT/models/Qwen3-8B-Q4_K_M.gguf}
export LLM_PORT=${LLM_PORT:-8080}
