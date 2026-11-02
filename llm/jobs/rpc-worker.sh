#!/bin/bash
#SBATCH --job-name=llm-rpcworker
#SBATCH --partition=llm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --output=/data/llm/logs/rpcworker-%j.out

# Exposes this node GPU to a llama-server running on the OTHER node.
# Submitted on its own rather than as an srun step of the server job: a step
# launched from node03 onto node04 never starts, because slurmd on the worker
# has to connect BACK to srun over ephemeral ports and firewalld only permits
# 22/8080/50052 between the nodes. Two independent jobs talk solely over 50052,
# which is open, so none of that matters.
#
#   sbatch --gres=gpu:maxwell:1 /data/llm/jobs/rpc-worker.sh

set -uo pipefail
source /data/llm/bin/llm-env.sh
PORT=${LLM_RPC_PORT:-50052}

echo "worker host: $(hostname -s)  ip: $(hostname -I | awk "{print \$1}")  port: $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Keep publishing telemetry so the fan controller still sees this card.
python3 /data/llm/bin/gpu-publish.py &

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

# Restart on crash instead of letting the job die. 2026-09-27 11:53 the 980 Ti took an Xid 31 MMU fault at ~30k
# context (gpt-oss-20b), the worker aborted, and every load failed on "Failed to connect :50052" until it was
# resubmitted by hand -- --requeue only covers node failure, not the program exiting.
while true; do
  ggml-rpc-server --host 0.0.0.0 --port "$PORT" "${CACHEARG[@]}"
  echo "$(date -Is) ggml-rpc-server exited with status $? -- restarting in 5s"
  sleep 5   # TIME_WAIT bind race
done
