#!/bin/bash
#SBATCH --job-name=llm-rpccpu
#SBATCH --partition=llmcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=11G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --output=/data/llm/logs/rpccpu-%j.out

# Exposes a COMPUTE node CPU + RAM as an extra RPC device, so layers that do not
# fit in the two GPUs spill here instead of onto the GPU node own small CPU.
# A compute node has 16 cores and 15.4 GB against the GPU node 6 cores and
# ~3.4 GB free, so spilled layers get more cores and far more headroom.
#
# Uses build-cpu: this node has no GPU, so the CUDA binary cannot even start
# (it links libcuda.so.1). build-cpu is now built with -DGGML_RPC=ON.
#
#   sbatch /data/llm/jobs/rpc-worker-cpu.sh

set -uo pipefail
export LLM_BUILD=/data/llm/build-cpu
source /data/llm/bin/llm-env.sh
PORT=${LLM_RPC_PORT:-50052}
THREADS=${LLM_RPC_THREADS:-${SLURM_CPUS_ON_NODE:-16}}

echo "cpu worker: $(hostname -s)  ip: $(hostname -I | awk "{print \$1}")  port: $PORT  threads: $THREADS"
free -g | awk '/Mem:/{print "  ram: " $2 " GB total, " $7 " GB available"}'

# Node-local tensor cache (bin/model-cache.py). With --cache, every tensor over
# 10 MB the router sends is kept as $LLAMA_CACHE/rpc/<hash>, and the next load of
# that model sends only the hash. LLAMA_CACHE must be LOCAL: the default is
# ~/.cache, which is the NFS home. The keeper evicts per the router's keep set.
CACHEARG=()
if [ -w /var/cache/slerm-llm ]; then
  export LLAMA_CACHE=/var/cache/slerm-llm
  CACHEARG=(--cache)
  python3 /data/llm/bin/model-cache.py worker >> /data/llm/logs/model-cache-$(hostname -s).log 2>&1 &
fi

exec ggml-rpc-server --host 0.0.0.0 --port "$PORT" --threads "$THREADS" "${CACHEARG[@]}"
