#!/bin/bash
#SBATCH --job-name=llm-strata
#SBATCH --partition=llm
#SBATCH --nodes=1
#SBATCH --nodelist=slerm-node04
#SBATCH --gres=gpu:pascal:2
#SBATCH --cpus-per-task=34
#SBATCH --mem=41G
#SBATCH --time=INFINITE
#SBATCH --output=/data/llm/logs/strata-%j.out

# One Strata server (github.com/Niko1221/Strata, install in /data/llm/strata) for the STRATA-* models in the model
# picker (2026-10-17, user: "make a STRATA-* model selection in the model picker and wire it up"). slerm-orch submits
# it with STRATA_MODEL=<registry name> when a request names a STRATA model, after it has stopped the llama.cpp router:
# both hold both P100s, so only one runs at a time. Do not sbatch it by hand. Registry: /data/llm/etc/strata-models.json.
#
# Start-up (measured 10-17, trial jobs 4587-4589 and jobs 4591-4593): node04's 43 GiB of RAM must hold the experts the
# cards do not (12.5 GiB pinned for the Coder) AND as much of the 27 GB n-gram table as possible, which Strata reads
# from the page cache on every token (a miss = an 11 ms NFS read; 8 GB cached -> 12-17 tok/s, 21 GB -> 27-39). So:
#   1. drop_before: posix_fadvise(DONTNEED) on files nobody needs cached any more (shard 1 once a pack has experts.bin;
#      a stale 9 GB copy from the previous start pushed the table out)
#   2. warm_before: the pack's experts.bin, sequentially (170-770 MB/s; the fill's own profile-order reads get ~32 MB/s
#      from NFS). Packs are built with tools/iq_pack.py --experts-bin: from experts.bin Strata releases each layer's
#      pages once it is in VRAM / pinned RAM; from the GGUF in place it keeps shard 1 mapped and the table loses.
#   3. the server; READY when /health reports a context
#   4. warm_after (skipped with --ple-io direct, see below; the table range of the MERGED file llama.cpp serves, so both engines share one cached copy), and only
#      then the ready marker /data/llm/run/strata-ready.json that slerm-orch waits for: warmed in the background, it
#      competed with the first answer for the head's disks (4-8 tok/s).
# --host 0.0.0.0: Strata listens on 127.0.0.1 unless told (the gateway on the head must reach it; firewalld lets only
# slerm-head in on 18100). Beside it, like the router: the task model (Open WebUI titles/tags) on :8081, on NUMA node 0 so it never competes with
# the expert pool on node 1, and the GPU publisher for the panel.
set -uo pipefail
MODEL=${STRATA_MODEL:?STRATA_MODEL is not set (slerm-orch sets it)}
REG=${STRATA_REGISTRY:-/data/llm/etc/strata-models.json}
# Local copies (2026-10-17): the registry's "local_copies" maps NFS paths to copies on node04's M.2 (/ssd, a VM 304 disk
# on byteme's internal USB 3.0 port): a table miss there is a ~0.3 ms read instead of an 11-19 ms one from byteme's raidz1.
# A path in the config or in the warm/drop lists that has a COMPLETE copy (same size; for a directory, every file) is
# swapped for it in a per-job copy of the config; anything else stays on NFS, so a missing or half-copied SSD only costs
# speed. 'requires' stays on the NFS paths: slerm-orch checks them on the head, which cannot see /ssd.
VARS=$(python3 - "$REG" "$MODEL" "${SLURM_JOB_ID:-manual}" <<'PY'
import json, os, shlex, sys
reg = json.load(open(sys.argv[1]))
m = reg["models"][sys.argv[2]]
def files(p):
    if os.path.isfile(p):
        return [p]
    return [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs]
def complete(src, dst):
    srcs = files(src)
    return bool(srcs) and all(os.path.isfile(dst + s[len(src):]) and os.path.getsize(dst + s[len(src):]) == os.path.getsize(s)
                              for s in srcs)
copies = reg.get("local_copies", {})
local = {s: d for s, d in copies.items() if complete(s, d)}
for s in sorted(set(copies) - set(local)):
    print("local copy missing or incomplete, using NFS:", s, file=sys.stderr)
used = set()
def sub(p, mark=True):
    for s, d in local.items():
        if p == s or p.startswith(s + "/"):
            if mark:
                used.add(s)
            return d + p[len(s):]
    return p
def walk(o):
    if isinstance(o, dict):
        return {k: walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(v) for v in o]
    return sub(o) if isinstance(o, str) else o
def spec(w):
    return "%s:%d:%d" % (sub(w["file"]), w["offset"], w["length"]) if isinstance(w, dict) else sub(w)
config = m["config"]
cfg = json.load(open(config))
run_cfg = walk(cfg)
warm_before = [spec(w) for w in m.get("warm_before", [])]
warm_after = [spec(w) for w in m.get("warm_after", [])]
drop = [sub(w, mark=False) for w in m.get("drop_before", [])]
for s in sorted(used):   # the NFS originals' cached pages would only take RAM from the local copies
    drop += [f for f in files(s) if f not in drop]
# --ple-io direct (2026-10-21): Strata reads table rows unbuffered, several at once, into its own row cache, so the page
# cache is no help: no table warm, and the table's cached pages are dropped (RAM for conversation parking). Only from the
# SSD copy: on NFS each miss is an 11-19 ms read with nothing cached behind it, so there it falls back to mmap.
a = run_cfg.get("args", [])
if "--ple-io" in a[:-1] and a[a.index("--ple-io") + 1] == "direct":
    ple = a[a.index("--ple-gguf") + 1] if "--ple-gguf" in a[:-1] else ""
    if ple.startswith("/ssd/"):
        warm_after = []
        drop += [ple] if ple not in drop else []
    else:
        a[a.index("--ple-io") + 1] = "mmap"
        print("table not on the local SSD (%s): --ple-io mmap instead of direct" % ple, file=sys.stderr)
if run_cfg != cfg:
    config = "/data/llm/run/strata-config-%s.json" % sys.argv[3]
    json.dump(run_cfg, open(config, "w"), indent=1)
for s in sorted(used):
    print("using the local copy", local[s], "for", s, file=sys.stderr)
print("CONFIG=%s" % shlex.quote(config))
print("PORT=%d" % int(reg.get("port", 18100)))
print("READY_S=%d" % int(m.get("ready_s", 2400)))
print("WARM_BEFORE=(%s)" % " ".join(shlex.quote(w) for w in warm_before))
print("WARM_AFTER=(%s)" % " ".join(shlex.quote(w) for w in warm_after))
print("DROP_BEFORE=(%s)" % " ".join(shlex.quote(w) for w in drop))
PY
) || { echo "ERROR: $MODEL is not in $REG" >&2; exit 2; }
eval "$VARS"

STRATA=/data/llm/strata/Strata
LOG=/data/llm/logs/strata-server-${SLURM_JOB_ID:-manual}.log
echo "=== $(date -Is) job ${SLURM_JOB_ID:-manual} on $(hostname -s): $MODEL"
echo "config $CONFIG  port $PORT  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
free -g | sed -n 1,2p

# Same bind wait as router-rpc.sh: a socket in TIME_WAIT from the previous server blocks the bind.
python3 - "$PORT" <<'PORTWAIT'
import socket, sys, time
port = int(sys.argv[1])
for attempt in range(90):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port)); s.close()
        break
    except OSError as e:
        s.close()
        if attempt == 0:
            print("port %d busy (%s), waiting..." % (port, e.strerror), flush=True)
        time.sleep(1)
else:
    print("port %d never freed" % port); sys.exit(1)
PORTWAIT

command -v nvidia-smi >/dev/null && python3 /data/llm/bin/gpu-publish.py &

# 2026-10-21: off by default. The task model is now an on-demand job on node02/node01 (jobs/tasks.sh, started by
# slerm-orch): here it sat inside Strata's 41 GiB cap and left too little RAM to park big conversations.
# LLM_TASK_MODEL=<gguf> brings the old sidecar back.
TASK_GGUF=${LLM_TASK_MODEL:-none}
TASK_PORT=${LLM_TASK_PORT:-8081}
if [ "$TASK_GGUF" != none ] && [ -f "$TASK_GGUF" ]; then
  TASK_BUILD=/data/llm/build-cpu
  echo "task model: $(basename "$TASK_GGUF") on :$TASK_PORT, NUMA node 0"
  ( export CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="$TASK_BUILD/bin"
    for try in 1 2 3 4 5; do
      echo "=== $(date -Is) start $try"
      nice -n 10 env -u LD_LIBRARY_PATH numactl --cpunodebind=0 --preferred=0 env "LD_LIBRARY_PATH=$TASK_BUILD/bin" \
        "$TASK_BUILD/bin/llama-server" -m "$TASK_GGUF" --alias tasks-llama3.2-1b --host 0.0.0.0 --port "$TASK_PORT" \
        -c 8192 --parallel 1 -t 6 -tb 16 -lm dio --no-webui
      sleep 10
    done ) > "/data/llm/logs/tasks-${SLURM_JOB_ID:-manual}.out" 2>&1 &
fi

rm -f /data/llm/run/strata-ready.json
if [ ${#DROP_BEFORE[@]} -gt 0 ]; then
  python3 - "${DROP_BEFORE[@]}" <<'PY'
import os, sys
for p in sys.argv[1:]:
    try:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        print("dropped the page cache of", os.path.basename(p))
    except OSError as e:
        print("drop", p, e)
PY
fi
if [ ${#WARM_BEFORE[@]} -gt 0 ]; then
  echo "=== $(date -Is) warming before start"
  python3 /data/llm/bin/page-warm.py "${WARM_BEFORE[@]}"
fi

cd "$STRATA"
export XDG_CONFIG_HOME=/data/llm/strata/config
export STRATA_ENGINE_READY_S=$READY_S
# The engine aborts a request when one step makes no progress for STRATA_WATCHDOG_S (default 60). On node04 the first
# prompt layer reads n-gram table rows that are not cached one page fault at a time from NFS (11 ms each), so a long
# prompt with half the table cached (IQ2_XS, a 52,914-token chat, 10-17 15:11) tripped it after 60 s and the request
# failed. Slow is not stuck here: allow 15 min per step.
export STRATA_WATCHDOG_S=${STRATA_WATCHDOG_S:-900}
# Strata refuses Host names it does not know (DNS rebinding protection); the gateway and slerm-orch call it by name
export STRATA_ALLOWED_HOSTS="$(hostname -s),$(hostname -f),$(hostname -I | awk '{print $1}')"
echo "=== $(date -Is) starting the server (log $LOG)"
env -u LD_LIBRARY_PATH numactl --cpunodebind=1 --preferred=1 \
  .venv/bin/python serve/server.py --engine strata --config "$CONFIG" --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
SP=$!

READY=0
END=$((SECONDS + READY_S + 120))
while kill -0 "$SP" 2>/dev/null && [ "$SECONDS" -lt "$END" ]; do
  if python3 - "$PORT" <<'PY'
import json, sys, urllib.request
try:
    h = json.load(urllib.request.urlopen("http://127.0.0.1:%s/health" % sys.argv[1], timeout=5))
except Exception:
    sys.exit(1)
sys.exit(0 if h.get("loaded") and (h.get("max_context") or 0) > 0 else 1)
PY
  then READY=1; break; fi
  sleep 5
done
if [ "$READY" = 1 ]; then
  echo "=== $(date -Is) READY after ${SECONDS}s"
  free -g | sed -n 1,2p
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  if [ ${#WARM_AFTER[@]} -gt 0 ]; then
    python3 /data/llm/bin/page-warm.py "${WARM_AFTER[@]}"
    free -g | sed -n 1,2p
  fi
  python3 -c "import json,sys,time; json.dump({'job': sys.argv[1], 'model': sys.argv[2], 'at': int(time.time())}, open('/data/llm/run/strata-ready.json', 'w'))" \
    "${SLURM_JOB_ID:-manual}" "$MODEL"
  echo "=== $(date -Is) ready marker written after ${SECONDS}s"
else
  echo "=== $(date -Is) server did not become ready (alive: $(kill -0 "$SP" 2>/dev/null && echo yes || echo no)); last log lines:"
  tail -20 "$LOG"
  kill "$SP" 2>/dev/null
fi
wait "$SP"
RC=$?
echo "=== $(date -Is) server exited ($RC)"
exit "$RC"
