#!/bin/bash
#SBATCH --job-name=p100-loads
#SBATCH --partition=gpu
#SBATCH --nodelist=slerm-node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=/data/llm/logs/p100loads-%j.out

# Real workloads on ONE Tesla P100 (2026-10-04), build-pascal, with the same
# server options the router presets use. Client: jobs/p100-loads.py.
# Everything fits in VRAM: no CPU spillover until the second P100 is in (user).
#   1. gpt-oss-20b as opencode uses it: 64k ctx, prompts of ~20k/45k/60k tokens
#      (the old 980 Ti pool crashed with Xid 31 at ~30.2k, twice)
#   2. qwen3-14b, 4 slots: concurrency the 6 GB cards never had room for
#   3. gemma-3-4b + mmproj: Firstlook's image path

set -uo pipefail
export LLM_BUILD=/data/llm/build-pascal
source /data/llm/bin/llm-env.sh
M=$LLM_ROOT/models
J=${SLURM_JOB_ID:-x}
PORT=8099
CLIENT="python3 $LLM_ROOT/jobs/p100-loads.py"

python3 /data/llm/bin/gpu-publish.py &
nvidia-smi dmon -s pucm -d 5 -o T > "$LLM_ROOT/logs/p100loads-$J.dmon" &

serve() {  # name args... -> starts llama-server, waits for /health
    local name=$1; shift
    echo; echo "== $(date +%T) $name"
    llama-server --host 127.0.0.1 --port $PORT -ngl 99 --flash-attn on --jinja \
        --reasoning-format deepseek --metrics "$@" > "$LLM_ROOT/logs/p100loads-$J.$name.log" 2>&1 &
    SRV=$!
    local t0=$SECONDS
    until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
        kill -0 $SRV 2>/dev/null || { echo "llama-server died:"; tail -15 "$LLM_ROOT/logs/p100loads-$J.$name.log"; return 1; }
        sleep 2
    done
    echo "loaded in $((SECONDS - t0))s; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3; }

serve gptoss -m "$M/gpt-oss-20b-mxfp4.gguf" -c 65536 --cache-type-k q8_0 --cache-type-v q8_0 \
    --chat-template-file /data/llm/etc/gpt-oss-20b-slerm.jinja --parallel 1 \
    && { $CLIENT longctx $PORT; echo "VRAM at end $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
stop

serve qwen14b-par4 -m "$M/qwen3-14b.gguf" -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 --parallel 4 \
    && $CLIENT parallel $PORT
stop

serve gemma-vision -m "$M/gemma-3-4b-it-q4km.gguf" --mmproj "$M/gemma-3-4b-it-mmproj-f16.gguf" -c 8192 --parallel 1 \
    && $CLIENT vision $PORT
stop

echo; echo "== $(date +%T) done"
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv
dmesg 2>/dev/null | grep -i xid | tail -3
