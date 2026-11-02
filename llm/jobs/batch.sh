#!/bin/bash
#SBATCH --job-name=llm-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=/data/llm/logs/batch-%A_%a.out

# Offline inference over a prompt file. Starts a private llama-server inside
# the job, drains the file through it, then exits and frees the card.
#
#   sbatch /data/llm/jobs/batch.sh prompts.jsonl out.jsonl
#   sbatch --array=0-1 /data/llm/jobs/batch.sh prompts.jsonl out.jsonl
#
# With --array, each task takes every Nth prompt and writes out.<task>.jsonl,
# so both cards can chew the same file at once. Slurm will not place a task on
# a node whose GPU the persistent server already holds, so batch and endpoint
# coexist without any preemption config.

set -euo pipefail
source /data/llm/bin/llm-env.sh

IN=${1:?usage: batch.sh <prompts-file> <out-file>}
OUT=${2:?usage: batch.sh <prompts-file> <out-file>}
CTX=${LLM_CTX:-4096}

# Private port so this never collides with a persistent server on the same node.
PORT=$(( 18000 + (SLURM_JOB_ID % 1000) ))
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && OUT="${OUT%.jsonl}.${SLURM_ARRAY_TASK_ID}.jsonl"

echo "host: $(hostname)  port: $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi on this node)"

llama-server --model "$LLM_MODEL" --host 127.0.0.1 --port "$PORT" \
  --n-gpu-layers "${LLM_NGL:-99}" --ctx-size "$CTX" --jinja \
  > "/data/llm/logs/batch-server-${SLURM_JOB_ID}.log" 2>&1 &
SRV=$!
trap "kill $SRV 2>/dev/null || true" EXIT

# Wait for the model to finish loading (cold NFS read of a 4.7G GGUF is slow).
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "server up after ${i}s"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "server died during load; see batch-server-${SLURM_JOB_ID}.log" >&2
    tail -20 "/data/llm/logs/batch-server-${SLURM_JOB_ID}.log" >&2
    exit 1
  fi
  sleep 1
done

# Extra driver flags (e.g. --no-think, --max-tokens N) via LLM_INFER_ARGS.
python3 /data/llm/bin/batch_infer.py "$IN" "$OUT" --url "http://127.0.0.1:$PORT" ${LLM_INFER_ARGS:-}
