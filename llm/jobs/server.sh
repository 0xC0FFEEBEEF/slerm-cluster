#!/bin/bash
#SBATCH --job-name=llm-server
#SBATCH --partition=llm
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --output=/data/llm/logs/server-%j.out

# Persistent OpenAI-compatible endpoint on one GPU node.
#
# Submit with the card you want, e.g.
#   sbatch --gres=gpu:turing:1  /data/llm/jobs/server.sh     # node03, faster
#   sbatch --gres=gpu:maxwell:1 /data/llm/jobs/server.sh     # node04
#
# Deliberately takes only 3 of the node6 cores and 6G of 9.4G, so the NEXRAD
# array (which runs on compute,gpu WITHOUT --gres) still has CPU room here.

set -euo pipefail
source /data/llm/bin/llm-env.sh

CTX=${LLM_CTX:-8192}
NGL=${LLM_NGL:-99}
ALIAS=$(basename "$LLM_MODEL" .gguf)
IP=$(hostname -I | awk "{print \$1}")

echo "host:  $(hostname)  ip: $IP  port: $LLM_PORT"
echo "model: $LLM_MODEL"
echo "ctx:   $CTX   ngl: $NGL"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi on this node)"

# Publish where this landed so clients do not have to guess the node.
EP=$LLM_ROOT/run/endpoint-${SLURM_JOB_ID}.json
cat > "$EP" <<JSON
{"job_id":"$SLURM_JOB_ID","host":"$(hostname -s)","ip":"$IP","port":$LLM_PORT,
 "url":"http://$IP:$LLM_PORT/v1","model":"$ALIAS","ctx":$CTX}
JSON
ln -sfn "$EP" "$LLM_ROOT/run/endpoint-$(hostname -s).json"
trap "rm -f $EP $LLM_ROOT/run/endpoint-$(hostname -s).json" EXIT

# 6 GB VRAM vs a 4.7 GB model leaves ~1 GB for KV. Quantising KV to q8_0 roughly
# halves it, which is what buys a usable context. Quantised V needs flash attn,
# so LLM_CACHE and LLM_FA move together.
CACHE=${LLM_CACHE:-q8_0}
FA=${LLM_FA:-on}
# Thinking is ON (models that support it decide for themselves). --reasoning-format
# deepseek routes thoughts into message.reasoning_content, which Open WebUI renders
# as a collapsible Thinking block instead of leaking <think> into the reply.
# NOTE: whatever generates Open WebUI titles/tags must NOT be a thinking model --
# it JSON.parses that reply. Task model is llama3.2:1b on cryoai Ollama.
# LLM_TKWARGS='{"enable_thinking":false}' forces thinking off.
TKWARGS=${LLM_TKWARGS:-'{}'}

# llama-server does not set SO_REUSEADDR, so a socket lingering in TIME_WAIT
# blocks the bind and it exits instantly. Parsing ss is unreliable here, so
# test the real thing: try the same bind llama-server will do, WITHOUT
# SO_REUSEADDR, and wait until it actually succeeds. TIME_WAIT is ~60s.
python3 - "$LLM_PORT" <<'PORTWAIT'
import socket, sys, time
port = int(sys.argv[1])
for attempt in range(90):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))   # deliberately no SO_REUSEADDR
        s.close()
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

# Publish GPU telemetry to NFS for the panel. The panel cannot ssh here and
# slurmapi.gpu_probe costs a job slot, so it can never be polled -- but this
# job is already sitting on the GPU, so sampling it is free. Dies with the job.
python3 /data/llm/bin/gpu-publish.py &

# Slurm can start a replacement server before the previous one has released
# port 8080, and llama-server exits instantly rather than waiting. The bind probe
# above narrows that window but cannot close it: the race is between the probe
# and llama-server own bind, microseconds later. So retry instead of dying. A
# genuine misconfiguration still fails every attempt and the job ends.
ARGS=(
  --model "$LLM_MODEL"
  --alias "$ALIAS"
  --host 0.0.0.0 --port "$LLM_PORT"
  --n-gpu-layers "$NGL"
  --ctx-size "$CTX"
  --cache-type-k "$CACHE" --cache-type-v "$CACHE"
  --flash-attn "$FA"
  --chat-template-kwargs "$TKWARGS"
  --reasoning-format deepseek
  --parallel 1 --jinja --metrics
)
for attempt in 1 2 3 4 5 6; do
  llama-server "${ARGS[@]}" && break
  rc=$?
  echo "llama-server exited rc=$rc (attempt $attempt of 6); retrying in 15s"
  sleep 15
done
