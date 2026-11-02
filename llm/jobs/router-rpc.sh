#!/bin/bash
#SBATCH --job-name=llm-router
#SBATCH --partition=llm
#SBATCH --nodes=1
#SBATCH --gres=gpu:pascal:2
#SBATCH --cpus-per-task=34
#SBATCH --mem=41G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --output=/data/llm/logs/router-%j.out

# The llama.cpp router for every slerm model. Since 2026-10-04 it runs on
# slerm-node04, which holds BOTH Tesla P100s (16 GB each, ~31 GB usable), and
# optionally adds compute-node CPUs over RPC for models that overflow the cards.
# slerm-orch submits it (--gres=gpu:pascal:2); do not sbatch it by hand.
#
# HISTORY (before 2026-10-04): node03's 1660 SUPER was the router and node04's
# 980 Ti joined as a separate RPC worker job, ~10.4 GB pooled.
#
#   sbatch /data/llm/jobs/server-rpc.sh
#   LLM_MODEL=/data/llm/models/qwen3-14b.gguf sbatch /data/llm/jobs/server-rpc.sh
#
# CAPACITY, NOT SPEED: layer split is pipelined, so the cards process a token in
# sequence rather than in parallel. For a model that already fits on one card
# this is SLOWER than jobs/server.sh.
#
# With LLM_GPU_WORKER set, this job submits a SEPARATE worker job for another
# node's GPU (the old 980 Ti path). It is deliberately not an srun step: a step launched from one
# node onto the other never starts, because the worker slurmd must connect back
# to srun over ephemeral ports and firewalld only permits 22/8080/50052 here.

set -uo pipefail
# 2026-10-07 P100 work: GPU-only routers run llama.cpp v0.5.0 + local P100 changes (build-v050-p100: HFMA2 mat-vec,
# Pascal tensor-split AllReduce, vmad dp4a). Routers with RPC workers keep build-pascal (b10729): the workers run
# b10729 too and the RPC protocol differs (6 vs 7). Notes: /data/llm/bench/qwen38-20261007, memory slerm-p100-kernel-work.
if [ -z "${LLM_BUILD:-}" ] && [ -z "${LLM_CPU_WORKER:-}" ] && [ "${LLM_GPU_WORKER:-none}" = none ]; then
  export LLM_BUILD=/data/llm/build-v050-p100d   # 2026-10-08 brick 8 (v5 verify kernel for Q4_K/Q5_K, v4 for Q6_K); brick 6 = p100c, brick 5 = p100b
fi
source /data/llm/bin/llm-env.sh
# 2026-10-06: cuBLAS defaults to FP16 ACCUMULATION on the P100s (fast-fp16 cards). FP32 is the precise path and
# was no slower for prompts (Ornith 90k: 624 vs 571 tok/s at ub2048, KLD 0.0070 vs 0.0084). The router passes its
# environment to every child model, so this is router-wide. Bench: /data/llm/bench/ornith-20261006/prec/.
export GGML_CUDA_CUBLAS_COMPUTE_TYPE=${GGML_CUDA_CUBLAS_COMPUTE_TYPE:-f32}
# 2026-10-07 P100 switches (build-pascal ignores them): AllReduce on Pascal, FP32 for decode-sized reductions and BF16
# from 1 MB (prompt; KLD 0.000004), HFMA2 mat-vec for Q4_K/Q5_K/Q6_K one-token matmuls (2 rows per row set).
export GGML_CUDA_AR_PASCAL=${GGML_CUDA_AR_PASCAL:-1}
export GGML_CUDA_AR_BF16_THRESHOLD=${GGML_CUDA_AR_BF16_THRESHOLD:-1048576}
export GGML_CUDA_MMVH_PASCAL=${GGML_CUDA_MMVH_PASCAL:-2}
# multi-token (MTP verify) batches stay on MMVQ: the mmvh multi-column path measured 26 vs 36.5 tok/s (brick 3)
# 2026-10-08 brick 6: MTP verify (2-4 tokens) through the v4 HFMA2 kernel with 4 rows per row set; A/B on the production
# settings: thinking 24.2 -> 28.9 tok/s, code 34.4 -> 42.4, 90k context 20.2 -> 21.9. 0 = stock MMVQ.
export GGML_CUDA_MMVH_MC=${GGML_CUDA_MMVH_MC:-104}
# 2026-10-08 brick 8: v5 (activations in shared memory, 4 rows) for Q4_K/Q5_K, Q6_K stays on v4 (4 rows): vs brick 6
# thinking 27.9 -> 29.1 tok/s, code 39.7 -> 42.9. IQ4_XS stays on stock MMVQ (through v4/v5 it was slower end to end).
export GGML_CUDA_MMVH_MC_Q6=${GGML_CUDA_MMVH_MC_Q6:-4}
# 2026-10-08: live knobs for the P100 build (PROF=N profiles N graph segments to /var/tmp/ggml-prof-<pid>.txt).
# The file normally does not exist; writing it changes the running server without a restart.
export GGML_CUDA_TUNE_FILE=${GGML_CUDA_TUNE_FILE:-/var/tmp/ggml-tune.txt}

CTX=${LLM_CTX:-8192}
# Default: this node's link farm (bin/model-cache.py), where each model resolves to
# a local copy when one is cached (~586 MB/s) and to NFS otherwise (57-217 MB/s).
# Prints models/servable itself on a node with no /var/cache/slerm-llm.
MODELS_DIR=${LLM_MODELS_DIR:-$(python3 /data/llm/bin/model-cache.py init-farm 2>/dev/null || echo /data/llm/models/servable)}
MODELS_MAX=${LLM_MODELS_MAX:-1}   # one model resident at a time (presets assume all of the VRAM)
MODELS_PRESET=${LLM_MODELS_PRESET:-/data/llm/etc/models.ini}
# Preset precedence is: CLI > per-model section > [*]. So when a preset is in
# use we must NOT pass ctx-size / tensor-split on the command line, or the CLI
# value silently overrides every per-model setting in the file.
PRESETARG=()
[ -f "$MODELS_PRESET" ] && PRESETARG=(--models-preset "$MODELS_PRESET")

# NOTE: --n-gpu-layers is deliberately NOT set here. llama.cpp auto-fits layers
# to free device memory, but only when ngl is left unset -- passing 999 makes it
# print "n_gpu_layers already set by user to 999, abort" and skip the fit, which
# OOMs any model larger than the pooled VRAM. Leaving it unset lets small models
# fully offload and larger ones (gpt-oss-20b) spill the remainder to host RAM.
# Set LLM_NGL to force a value.
CACHE=${LLM_CACHE:-q8_0}
FA=${LLM_FA:-on}
# Thinking is ON (models that support it decide for themselves). --reasoning-format
# deepseek routes thoughts into message.reasoning_content, which Open WebUI renders
# as a collapsible Thinking block instead of leaking <think> into the reply.
# NOTE: whatever generates Open WebUI titles/tags must NOT be a thinking model --
# it JSON.parses that reply. Task model: tasks-llama3.2-1b, started below beside the router (2026-10-09).
# LLM_TKWARGS='{"enable_thinking":false}' forces thinking off.
TKWARGS=${LLM_TKWARGS:-'{}'}
RPC_PORT=${LLM_RPC_PORT:-50052}
# MEASURED device order: the RPC device sorts FIRST, the local card second.
# Verified by VRAM -- "60,40" put 5585 MiB on the remote 980 Ti and only 4120 MiB
# on the local 1660 SUPER, i.e. backwards. So "40,60" is what favours the faster
# card: 40% to the remote maxwell, 60% to the local turing. Measured 33.5 vs
# 22.0 tok/s on identical single-card work, and a pipelined split wants more
# layers on the faster card, so it tracks throughput rather than VRAM.
SPLIT=${LLM_SPLIT:-}   # unset: let llama.cpp auto-fit across the devices
WORKER_GRES=${LLM_WORKER_GRES:-gpu:pascal:1}

echo "main  : $(hostname -s)  models-dir: $MODELS_DIR (max $MODELS_MAX resident)"
echo "ctx   : $CTX  split: $SPLIT  cache: $CACHE  fa: $FA"

# A worker that is LISTENING is not necessarily SERVING. ggml-rpc-server takes
# one client at a time and never times out a read, so when this job's node dies
# without closing its sockets (node03 rebooted 2026-09-13), every worker stays
# blocked on that dead connection forever -- while the kernel keeps accepting
# new connections into the listen backlog. A TCP connect still succeeds, the
# requeued router adopted all of them, and llama-server hung in the RPC
# handshake without ever binding its HTTP port, so Open WebUI showed no slerm
# models at all. Only an answered HELLO (bin/rpc-hello.py) proves a worker works.
rpc_ready() {  # host port seconds -> 0 as soon as the worker answers HELLO
  local end=$((SECONDS + $3))
  while :; do
    python3 /data/llm/bin/rpc-hello.py "$1" "$2" 3 >/dev/null && return 0
    [ "$SECONDS" -ge "$end" ] && return 1
    sleep 1
  done
}
retire_job() {  # scancel, then wait until the job is gone so its port is free
  scancel "$1" 2>/dev/null || true
  for _ in $(seq 1 30); do
    [ -z "$(squeue -h -j "$1" -o "%i" 2>/dev/null)" ] && return 0
    sleep 2
  done
  echo "WARNING: job $1 still queued 60s after scancel" >&2
}

# On preemption this job is REQUEUED, and the EXIT trap does not reliably get to
# cancel the workers first -- so a restart would submit a second worker while the
# first is still running, leaking an orphan job every time. Adopt a live worker
# instead of blindly submitting one -- but only one that still answers HELLO.
# A separate GPU worker (another node's card over RPC) is OFF by default since
# 2026-10-04: both P100s are local to the router. LLM_GPU_WORKER=yes brings the
# old path back if a second GPU node ever appears.
WJOB=""
WORKER_IP=""
if [ "${LLM_GPU_WORKER:-none}" != none ]; then
  WJOB=$(squeue -h -u "$USER" -n llm-rpcworker -t R -o "%i" 2>/dev/null | head -1)
  if [ -n "$WJOB" ]; then
    echo "adopting existing rpc worker job $WJOB"
    N=$(squeue -h -j "$WJOB" -o "%N" 2>/dev/null | tr -d " ")
    WIP=$(getent hosts "$N" | awk "{print \$1}" | head -1)
    if ! rpc_ready "$WIP" "$RPC_PORT" 15; then
      echo "worker $WJOB on $N does not answer HELLO (wedged) -- replacing it"
      retire_job "$WJOB"
      WJOB=""
    fi
  fi
  if [ -z "$WJOB" ]; then
    WJOB=$(sbatch --parsable --gres="$WORKER_GRES" /data/llm/jobs/rpc-worker.sh)
    echo "submitted rpc worker job $WJOB (gres $WORKER_GRES)"
  fi
  cleanup() { echo "cancelling worker $WJOB"; scancel "$WJOB" 2>/dev/null || true; }
  trap cleanup EXIT

  # Wait for the worker to land on a node, then for it to answer.
  WORKER_IP=""
  for i in $(seq 1 60); do
    N=$(squeue -h -j "$WJOB" -t R -o "%N" 2>/dev/null | tr -d " ")
    if [ -n "$N" ]; then
      WORKER_IP=$(getent hosts "$N" | awk "{print \$1}" | head -1)
      [ -n "$WORKER_IP" ] && { echo "worker running on $N ($WORKER_IP)"; break; }
    fi
    sleep 2
  done
  [ -n "$WORKER_IP" ] || { echo "ERROR: worker job $WJOB never started" >&2; exit 1; }

  echo "waiting for rpc-server on $WORKER_IP:$RPC_PORT ..."
  rpc_ready "$WORKER_IP" "$RPC_PORT" 120 \
    || { echo "ERROR: rpc-server on $WORKER_IP:$RPC_PORT never answered HELLO in 120s" >&2; exit 1; }
  echo "worker answers HELLO"
fi

# llama-server does not set SO_REUSEADDR, so a socket still in TIME_WAIT from a
# previous server blocks the bind and it exits instantly. Wait for a real bind.
# Optional third device: a compute node CPU + RAM, exposed over RPC. For models
# that overflow the two GPUs this is a better landing spot than the GPU node own
# CPU -- 16 cores and ~14 GB free versus 6 cores and ~3.4 GB. Latency is not the
# obstacle people assume: node03->node01 measures 0.606 ms against 0.548 ms for
# node03->node04 on the same hypervisor, and a layer boundary only ships a few
# KB per token.
#   LLM_CPU_WORKER=slerm-node01 sbatch /data/llm/jobs/router-rpc.sh
RPC_LIST="${WORKER_IP:+$WORKER_IP:$RPC_PORT}"
# LLM_CPU_WORKER takes a comma-separated list, e.g. "slerm-node01,slerm-node02".
CJOBS=""
CPU_IPS=""
for W in $(echo "${LLM_CPU_WORKER:-}" | tr "," " "); do
  [ -z "$W" ] && continue
  CIP=$(getent hosts "$W" | awk "{print \$1}" | head -1)
  CJ=$(squeue -h -u "$USER" -n llm-rpccpu -t R -w "$W" -o "%i" 2>/dev/null | head -1)
  if [ -n "$CJ" ]; then
    echo "adopting existing cpu rpc worker job $CJ on $W"
    if ! rpc_ready "$CIP" "$RPC_PORT" 15; then
      echo "cpu worker $CJ on $W does not answer HELLO (wedged) -- replacing it"
      retire_job "$CJ"
      CJ=""
    fi
  fi
  if [ -z "$CJ" ]; then
    # Pick the CPU worker build that matches this node's ISA. node05
    # (deadlock) is Ivy Bridge -- no avx2/fma/bmi2 -- so build-cpu/ binaries
    # SIGILL there at startup. build-cpu-avx/ is the AVX-only build.
    CPUSCRIPT=/data/llm/jobs/rpc-worker-cpu.sh
    case "$W" in
      slerm-node05|deadlock) CPUSCRIPT=/data/llm/jobs/rpc-worker-cpu-avx.sh ;;
    esac
    CJ=$(sbatch --parsable --nodelist="$W" "$CPUSCRIPT")
    echo "  (using $(basename "$CPUSCRIPT") for $W)"
    echo "submitted cpu rpc worker job $CJ on $W"
  fi
  CJOBS="$CJOBS $CJ"
  trap "echo cancelling workers $WJOB$CJOBS; scancel $WJOB$CJOBS 2>/dev/null || true" EXIT
  if rpc_ready "$CIP" "$RPC_PORT" 120; then
    echo "cpu worker $W answers HELLO"
    RPC_LIST="${RPC_LIST:+$RPC_LIST,}$CIP:$RPC_PORT"
    CPU_IPS="$CPU_IPS $CIP"
  else
    echo "WARNING cpu worker $W never answered HELLO -- skipping it" >&2
    scancel "$CJ" 2>/dev/null || true
    # A generated preset (slerm_topo.py) has one split value per device, in
    # order. Skipping a device shifts every later share onto the wrong one, so
    # under the orchestrator a missing worker is fatal and it re-plans instead.
    if [ "${LLM_STRICT_DEVICES:-0}" = 1 ]; then
      echo "ERROR: strict device set and $W is missing -- exiting" >&2
      exit 3
    fi
  fi
done
echo "rpc devices: $RPC_LIST"

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

# 2026-10-10 NUMA. byteme has 64 GB in two unequal halves: CPU1 = one 32 GB DIMM (ONE memory channel), CPU2 = 2x 16 GB
# (TWO channels). VM 304 mirrors that as two guest NUMA nodes: node 0 = CPU1 (vCPUs 0-17, 21.5 GB), node 1 = CPU2
# (vCPUs 18-35, 23.5 GB); byteme's hookscript slerm-numa-pin pins the vCPU threads to the matching host CPU. Layers a
# model keeps on this node's CPU are read from RAM once per token, so the channel count sets the speed. Measured with
# llama-bench (bench/numa-bench-20261010.sh job 4544, numa-final-20261010.sh job 4546), tok/s written:
#                                   llama3.2-1b (CPU)   qwen3.6-35b Q6, 6 layers on CPU   coder-next Q4, 21 on CPU
#   threads + memory on node 1            21.3                    27.4                            12.1
#   threads + memory on node 0            11.6                    20.6
#   no placement (what you get unbound)   16.5 (11.7-20.2 by luck)  21.0-23.3                     10.4
#   spread 1:2 over both CPUs             18.1                    27.8                            10.4
# So the router and every model it starts run on node 1's CPUs with memory PREFERRED there -- not bound: a model
# bigger than node 1 spills to node 0 instead of being OOM-killed. llama.cpp's own --numa distribute needs mmap
# (see load-mode in models.ini) and fell to 1.2 tok/s at 35 threads. numactl must not see the CUDA toolchain's older
# libnuma on LD_LIBRARY_PATH ("libnuma_2.1 not found"), hence the env juggling. Other nodes: nothing changes.
NUMA_PRE=(); NUMA_T=(); NUMA_CPUS=""
if [ "$(hostname -s)" = slerm-node04 ] && [ -r /sys/devices/system/node/node1/cpulist ] && command -v numactl >/dev/null; then
  # this job's CPUs on node 1 = the thread count (llama.cpp would otherwise start one thread per core of the whole VM)
  NUMA_N=$(python3 -c '
import os
def cl(s):
    out = set()
    for p in s.strip().split(","):
        a, _, b = p.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out
print(len(os.sched_getaffinity(0) & cl(open("/sys/devices/system/node/node1/cpulist").read())))' 2>/dev/null || echo 0)
  if [ "${NUMA_N:-0}" -ge 4 ]; then
    NUMA_CPUS=$(cat /sys/devices/system/node/node1/cpulist)
    NUMA_PRE=(env -u LD_LIBRARY_PATH numactl --cpunodebind=1 --preferred=1 env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}")
    NUMA_T=(--threads "$NUMA_N" --threads-batch "$NUMA_N")
    echo "numa  : router on node 1 (host CPU2, cpus $NUMA_CPUS), $NUMA_N threads"
    # "Preferred" falls back to node 0 BEFORE the kernel drops any page cache, so cache left on node 1 (an earlier job,
    # a model read through the page cache) pushes model weights onto the one-channel half: coder-next Q4 got 9.7 of its
    # 21 GiB there and wrote 10.9 instead of 12.1 tok/s. vm.zone_reclaim_mode=1 did not help (tested). So: (1) sweep
    # node 1 now -- a BOUND allocation does make the kernel drop cache, so take what is free or cached there and give
    # it straight back; (2) models load with direct I/O (load-mode = dio in models.ini), which leaves no page cache;
    # (3) the cache keeper below prefers node 0 for its copies. Nothing long-lived is strictly bound: a bound
    # allocation that cannot be met OOM-kills the biggest process on the node, i.e. the model.
    env -u LD_LIBRARY_PATH numactl --membind=1 python3 -c '
import re
m = {k: int(v) for k, v in re.findall(r"Node 1 (\w+):\s+(\d+) kB", open("/sys/devices/system/node/node1/meminfo").read())}
n = (m["MemFree"] + m["FilePages"] - m.get("Shmem", 0) - m.get("Mapped", 0) - 2500000) * 1024
if n > 0:
    b = bytearray(n)
    b[::4096] = b"\1" * (len(b) // 4096)' 2>/dev/null
    echo "numa  : node 1 free after sweep: $(awk '/MemFree/{print int($4/1024)}' /sys/devices/system/node/node1/meminfo) MB"
  else
    echo "numa  : only ${NUMA_N:-0} of this job's CPUs are on node 1; running unbound"
  fi
fi

# 2026-10-09 (user: "move the tags and such on another node"): Open WebUI's task model -- chat titles, tags and
# follow-up suggestions -- ran on cryoai's own CPU (Ollama llama3.2:1b) and froze Open WebUI for minutes on long
# chats. It now runs HERE, on this node's CPU beside the router, as its own small llama-server that lives and dies
# with this job (Slurm kills the whole job, so it needs no cleanup). The gateway sends model "tasks-llama3.2-1b" to
# it and never brings slerm up for it. CPU-only build, so it cannot touch the GPUs; nice'd so the router wins the
# CPU. It restarts a few times if it dies (a requeued job can find its port still in TIME_WAIT).
# LLM_TASK_MODEL=none turns it off. Log: /data/llm/logs/tasks-<job>.out
TASK_GGUF=${LLM_TASK_MODEL:-/data/llm/models/servable/llama3.2-1b.gguf}
TASK_PORT=${LLM_TASK_PORT:-8081}
if [ "$TASK_GGUF" != none ] && [ -f "$TASK_GGUF" ]; then
  TASK_BUILD=/data/llm/build-cpu
  grep -qw avx2 /proc/cpuinfo || TASK_BUILD=/data/llm/build-cpu-avx   # deadlock (Ivy Bridge) has no avx2
  echo "task model: $(basename "$TASK_GGUF") on :$TASK_PORT ($(basename "$TASK_BUILD"))"
  # Threads follow the job size (2026-10-10, node04 went to 36 vCPU on 2x E5-2695 v4): reading the prompt scales
  # with threads (pp512 54 tok/s at 3, 135 at 8, 284 at 18, ~470 at 35), writing does not (memory-bound from 8 up).
  # A small job (CPU-only router on node05, 8 CPUs) keeps the old 3/4.
  # On node04 (NUMA block above) its memory goes to node 1 and the 8 writing threads stay on node 1's CPUs (21 tok/s
  # there, 11.6 on node 0), while prompt reading may use every CPU of the job.
  TASK_T=3; TASK_TB=4
  if [ "${SLURM_CPUS_PER_TASK:-0}" -ge 16 ]; then
    TASK_T=8; TASK_TB=$(( SLURM_CPUS_PER_TASK - 2 )); [ "$TASK_TB" -gt 32 ] && TASK_TB=32
  fi
  TASK_NUMA=(); TASK_CPU=()
  if [ ${#NUMA_PRE[@]} -gt 0 ]; then
    TASK_NUMA=(env -u LD_LIBRARY_PATH numactl --preferred=1 env "LD_LIBRARY_PATH=$TASK_BUILD/bin")
    # -lm dio: the default (mmap) maps whatever page-cache copy of the GGUF already exists, wherever it sits; on
    # 2026-10-10 that was node 0 and it wrote 7 tok/s from the CPUs of node 1. Direct I/O gives it its own memory.
    TASK_CPU=(-lm dio)
    case "$NUMA_CPUS" in *,*) ;; *-*) TASK_CPU+=(-Cr "$NUMA_CPUS" -Crb "0-$(( $(nproc --all) - 1 ))") ;; esac
  fi
  ( export CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="$TASK_BUILD/bin"
    for try in 1 2 3 4 5; do
      echo "=== $(date -Is) start $try"
      nice -n 10 "${TASK_NUMA[@]}" "$TASK_BUILD/bin/llama-server" -m "$TASK_GGUF" --alias tasks-llama3.2-1b \
        --host 0.0.0.0 --port "$TASK_PORT" -c 8192 --parallel 1 -t "$TASK_T" -tb "$TASK_TB" "${TASK_CPU[@]}" --no-webui
      sleep 10
    done ) > "/data/llm/logs/tasks-${SLURM_JOB_ID:-manual}.out" 2>&1 &
fi

command -v nvidia-smi >/dev/null && python3 /data/llm/bin/gpu-publish.py &
# Keeps the last LLM_CACHE_KEEP (2) models on local disk for LLM_CACHE_TTL (2h)
# and publishes run/model-activity.json, which the workers' keepers follow.
case "$MODELS_DIR" in
  /var/cache/slerm-llm/*) KEEP_NUMA=()
                          [ ${#NUMA_PRE[@]} -gt 0 ] && KEEP_NUMA=(env -u LD_LIBRARY_PATH numactl --preferred=0 env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}")
                          "${KEEP_NUMA[@]}" python3 /data/llm/bin/model-cache.py router "$LLM_PORT" \
                            >> /data/llm/logs/model-cache-$(hostname -s).log 2>&1 & ;;
esac

# --tensor-split maps POSITIONALLY onto this device order, and whether RPC
# devices sort before or after the local one varies by llama.cpp version.
echo "=== device order (tensor-split maps onto this order) ==="
RPCARG=()
[ -n "$RPC_LIST" ] && RPCARG=(--rpc "$RPC_LIST")
llama-server --list-devices "${RPCARG[@]}" 2>&1 | head -20

NGLARG=()
[ -n "${LLM_NGL:-}" ] && NGLARG=(--n-gpu-layers "$LLM_NGL")
# --tensor-split ALSO disables auto-fit ("model_params::tensor_split already set
# by user, abort"), exactly like --n-gpu-layers does. Both must stay unset for
# llama.cpp to size each model to the free VRAM, which is what lets a model
# bigger than the pooled cards load at all. Set LLM_SPLIT to force a ratio and
# accept that only fully-fitting models will then load.
# When CPU workers are attached, auto-fit is actively HARMFUL: it allocates in
# proportion to FREE MEMORY, and a CPU node advertises ~14 GB against ~5.7 GB per
# GPU, so it grabs most of the layers and leaves the GPUs idle. Measured on
# gpt-oss-20b: auto-fit gave 8.76 tok/s with only 1.7/2.0 GB on the GPUs, while
# an explicit GPU-heavy split gave 22.94 tok/s with 5.2/5.5 GB on them -- 2.6x.
# So derive a GPU-heavy split whenever CPU devices are present: the GPUs take
# the bulk and the CPU nodes absorb only the overflow.
# Device order is the RPC list in order, then the local GPUs (CUDA0, CUDA1) last.
SPLITARG=()
if [ -n "${LLM_SPLIT:-}" ]; then
  SPLITARG=(--tensor-split "$LLM_SPLIT")
elif [ ${#PRESETARG[@]} -gt 0 ]; then
  # A preset is in use and supplies per-model splits. Passing a derived
  # split here would outrank every one of them, since CLI wins.
  echo "using per-model splits from $MODELS_PRESET"
elif [ -n "$CPU_IPS" ]; then
  ncpu=$(echo "$CPU_IPS" | wc -w)
  per_cpu=5
  ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
  [ "$ngpu" -ge 1 ] || ngpu=1
  gpu_share=$(( (100 - ncpu * per_cpu) / ngpu ))
  auto=""
  for _ in $(seq 1 "$ncpu"); do auto="${auto:+$auto,}$per_cpu"; done   # rpc CPU devices first
  for _ in $(seq 1 "$ngpu"); do auto="$auto,$gpu_share"; done           # then the local GPUs
  echo "derived tensor-split for $ngpu local GPU(s) + $ncpu CPU device(s): $auto"
  SPLITARG=(--tensor-split "$auto")
fi

exec "${NUMA_PRE[@]}" llama-server "${NUMA_T[@]}" "${NGLARG[@]}" "${SPLITARG[@]}" "${PRESETARG[@]}" \
  --models-dir "$MODELS_DIR" \
  --models-max "$MODELS_MAX" \
  --host 0.0.0.0 --port "$LLM_PORT" \
  "${RPCARG[@]}" \
  --chat-template-kwargs "$TKWARGS" \
  --parallel 1 --jinja --metrics
