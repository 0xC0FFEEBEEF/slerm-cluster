#!/bin/bash
#SBATCH --job-name=llm-tasks
#SBATCH --partition=llmcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=3G
#SBATCH --time=INFINITE
#SBATCH --output=/data/llm/logs/tasks-%j.out
# Open WebUI's task model (titles, tags, follow-ups) as its own job on node02 or node01 (2026-10-21).
# slerm-orch submits it when a task request arrives and nothing else serves one (a router job still runs its own
# copy beside the router), and cancels it after task_idle_s without a request; a node woken for it then powers off.
# It used to run beside Strata on node04, inside Strata's 41 GiB memory cap, where its 1.6 GiB left too little RAM
# to park a big conversation. Do not sbatch it by hand: the gateway only routes to the job slerm-orch knows about.
TASK_GGUF=${LLM_TASK_GGUF:-/data/llm/models/servable/llama3.2-1b.gguf}
BUILD=/data/llm/build-cpu
echo "=== $(date -Is) task model $(basename "$TASK_GGUF") on $(hostname):8081"
export CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="$BUILD/bin"
exec nice -n 10 "$BUILD/bin/llama-server" -m "$TASK_GGUF" --alias tasks-llama3.2-1b --host 0.0.0.0 --port 8081 \
  -c 8192 --parallel 1 -t 6 -tb 6 -lm dio --no-webui
