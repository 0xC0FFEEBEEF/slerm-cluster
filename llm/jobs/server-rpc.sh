#!/bin/bash
#SBATCH --job-name=llm-rpc
#SBATCH --partition=llm
#SBATCH --nodes=1
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --output=/data/llm/logs/rpc-%j.out

# Run one model across BOTH GPUs so a model that fits on neither card alone
# runs entirely in VRAM. Usable budget is ~5.2 GB per card (~10.4 GB combined),
# which covers a 14B Q4 but not the 20B or 27B.
#
#   sbatch /data/llm/jobs/server-rpc.sh
#   LLM_MODEL=/data/llm/models/qwen3-14b.gguf sbatch /data/llm/jobs/server-rpc.sh
#
# CAPACITY, NOT SPEED: layer split is pipelined, so the cards process a token in
# sequence rather than in parallel. For a model that already fits on one card
# this is SLOWER than jobs/server.sh.
#
# This job runs on the turing node and submits a SEPARATE worker job for the
# maxwell card. It is deliberately not an srun step: a step launched from one
# node onto the other never starts, because the worker slurmd must connect back
# to srun over ephemeral ports and firewalld only permits 22/8080/50052 here.

set -uo pipefail
source /data/llm/bin/llm-env.sh

CTX=${LLM_CTX:-8192}
CACHE=${LLM_CACHE:-q8_0}
FA=${LLM_FA:-on}
# Thinking is ON (models that support it decide for themselves). --reasoning-format
# deepseek routes thoughts into message.reasoning_content, which Open WebUI renders
# as a collapsible Thinking block instead of leaking <think> into the reply.
# NOTE: whatever generates Open WebUI titles/tags must NOT be a thinking model --
# it JSON.parses that reply. Task model is llama3.2:1b on cryoai Ollama.
# LLM_TKWARGS='{"enable_thinking":false}' forces thinking off.
TKWARGS=${LLM_TKWARGS:-'{}'}
RPC_PORT=${LLM_RPC_PORT:-50052}
# MEASURED device order: the RPC device sorts FIRST, the local card second.
# Verified by VRAM -- "60,40" put 5585 MiB on the remote 980 Ti and only 4120 MiB
# on the local 1660 SUPER, i.e. backwards. So "40,60" is what favours the faster
# card: 40% to the remote maxwell, 60% to the local turing. Measured 33.5 vs
# 22.0 tok/s on identical single-card work, and a pipelined split wants more
# layers on the faster card, so it tracks throughput rather than VRAM.
SPLIT=${LLM_SPLIT:-40,60}
WORKER_GRES=${LLM_WORKER_GRES:-gpu:maxwell:1}

echo "main  : $(hostname -s)  model: $LLM_MODEL"
echo "ctx   : $CTX  split: $SPLIT  cache: $CACHE  fa: $FA"

WJOB=$(sbatch --parsable --gres="$WORKER_GRES" /data/llm/jobs/rpc-worker.sh)
echo "submitted rpc worker job $WJOB (gres $WORKER_GRES)"
cleanup() { echo "cancelling worker $WJOB"; scancel "$WJOB" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the worker to land on a node, then for its port to answer.
WORKER_IP=""
for i in $(seq 1 60); do
  N=$(squeue -h -j "$WJOB" -o "%N" 2>/dev/null | tr -d " ")
  if [ -n "$N" ]; then
    WORKER_IP=$(getent hosts "$N" | awk "{print \$1}" | head -1)
    [ -n "$WORKER_IP" ] && { echo "worker running on $N ($WORKER_IP)"; break; }
  fi
  sleep 2
done
[ -n "$WORKER_IP" ] || { echo "ERROR: worker job $WJOB never started" >&2; exit 1; }

echo "waiting for rpc-server on $WORKER_IP:$RPC_PORT ..."
ok=0
for i in $(seq 1 120); do
  if (exec 3<>/dev/tcp/$WORKER_IP/$RPC_PORT) 2>/dev/null; then ok=1; echo "worker up after ${i}s"; break; fi
  sleep 1
done
[ "$ok" = 1 ] || { echo "ERROR: no RPC worker on $WORKER_IP:$RPC_PORT after 120s" >&2; exit 1; }

# llama-server does not set SO_REUSEADDR, so a socket still in TIME_WAIT from a
# previous server blocks the bind and it exits instantly. Wait for a real bind.
python3 - "$LLM_PORT" <<'PORTWAIT'
import socket, sys, time
port = int(sys.argv[1])
for attempt in range(90):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port)); s.close()
        if attempt:
            print("port %d free after %ds" % (port, attempt))
        break
    except OSError as e:
        s.close()
        if attempt == 0:
            print("port %d busy (%s), waiting..." % (port, e.strerror), flush=True)
        time.sleep(1)
else:
    print("port %d never freed" % port); sys.exit(1)
PORTWAIT

python3 /data/llm/bin/gpu-publish.py &

# --tensor-split maps POSITIONALLY onto this device order, and whether RPC
# devices sort before or after the local one varies by llama.cpp version.
echo "=== device order (tensor-split maps onto this order) ==="
llama-server --list-devices --rpc "$WORKER_IP:$RPC_PORT" 2>&1 | head -20

exec llama-server \
  --model "$LLM_MODEL" \
  --alias "$(basename "$LLM_MODEL" .gguf)-rpc" \
  --host 0.0.0.0 --port "$LLM_PORT" \
  --rpc "$WORKER_IP:$RPC_PORT" \
  --n-gpu-layers 999 \
  --tensor-split "$SPLIT" \
  --ctx-size "$CTX" \
  --cache-type-k "$CACHE" --cache-type-v "$CACHE" \
  --flash-attn "$FA" \
  --chat-template-kwargs "$TKWARGS" \
  --reasoning-format deepseek \
  --parallel 1 --jinja --metrics
