#!/bin/bash
#SBATCH --job-name=llm-rpccpu
#SBATCH --partition=llmcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --output=/data/llm/logs/rpccpu-avx-%j.out

# Same idea as rpc-worker-cpu.sh, but for slerm-node05 (deadlock), which differs
# from node01/02 in two ways that both matter:
#
#  1. ISA: it is a Xeon E5-1620 v2 (Ivy Bridge) with avx + f16c but NO avx2, NO
#     fma and NO bmi2. build-cpu/ is compiled to a Haswell baseline, so every
#     binary in it dies with SIGILL at startup -- ggml_cpu_init() itself emits
#     `shlx` (BMI2). Hence build-cpu-avx/, built with
#     -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF.
#     Note GGML_BMI2 defaults to ${INS_ENB} exactly like AVX2, so turning off
#     AVX2/FMA alone is NOT enough.
#
#  2. Shape: 8 threads (4 real cores) but 31.7 GB RAM, versus node01/02's 16
#     threads and 15.4 GB. It is the RAM node, not a compute node -- it holds
#     more layers than either compute node but computes each one more slowly.
#
#   sbatch -w slerm-node05 /data/llm/jobs/rpc-worker-cpu-avx.sh

set -uo pipefail
export LLM_BUILD=/data/llm/build-cpu-avx
source /data/llm/bin/llm-env.sh
PORT=${LLM_RPC_PORT:-50052}
THREADS=${LLM_RPC_THREADS:-${SLURM_CPUS_ON_NODE:-8}}

echo "cpu worker (avx-only build): $(hostname -s)  ip: $(hostname -I | awk "{print \$1}")  port: $PORT  threads: $THREADS"
free -g | awk '/Mem:/{print "  ram: " $2 " GB total, " $7 " GB available"}'
grep -m1 "model name" /proc/cpuinfo | sed 's/^/  cpu: /'

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
