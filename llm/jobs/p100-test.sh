#!/bin/bash
#SBATCH --job-name=p100-test
#SBATCH --partition=gpu
#SBATCH --nodelist=slerm-node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=/data/llm/logs/p100test-%j.out

# Acceptance test for a Tesla P100 (2026-10-04): correctness through
# llama-server, then llama-bench over models that fit on one 16 GB card, then a
# 32k-deep run (nothing had ever run past ~30.2k on the old 980 Ti pool).
# Uses build-pascal (sm_60). Named without the llm- prefix on purpose: slerm-orch
# then treats node04 as busy and leaves it alone.

set -uo pipefail
export LLM_BUILD=/data/llm/build-pascal
source /data/llm/bin/llm-env.sh
M=$LLM_ROOT/models
J=${SLURM_JOB_ID:-x}

# Telemetry for gpu-fan-control on the head (the P100 is passively cooled and
# iDRAC cannot see it), plus a 5 s power/temp/clock log for this report.
python3 /data/llm/bin/gpu-publish.py &
nvidia-smi dmon -s pucm -d 5 -o T > "$LLM_ROOT/logs/p100test-$J.dmon" &

echo "== $(date -Is) $(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,clocks.max.mem --format=csv
llama-server --list-devices 2>&1 | tail -3

echo; echo "== correctness (qwen3-8b, thinking off)"
llama-server -m "$M/qwen3-8b.gguf" -ngl 99 -fa on -c 8192 --host 127.0.0.1 --port 8099 \
    --jinja --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":false}' \
    > "$LLM_ROOT/logs/p100test-$J.server.log" 2>&1 &
SRV=$!
for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:8099/health >/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "llama-server died"; tail -20 "$LLM_ROOT/logs/p100test-$J.server.log"; break; }
    sleep 2
done
for q in "What is 17 times 23? Reply with only the number." \
         "What is the capital of Australia? Reply with one word."; do
    curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":32,\"temperature\":0}" |
        python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("timings",{});
print(repr(d["choices"][0]["message"]["content"].strip()), " gen %.1f tok/s" % t.get("predicted_per_second",0))'
done
kill $SRV; wait $SRV 2>/dev/null
grep -E "offloaded|CUDA0 model buffer|compute capability" "$LLM_ROOT/logs/p100test-$J.server.log" | head -5

echo; echo "== llama-bench pp512 / tg128, all layers on the P100, flash attention on"
for m in llama3.2-1b qwen3-8b ornith-1.5-9b-q4km qwen3-14b gpt-oss-20b-mxfp4 qwen3-30b-a3b-2507-q3km; do
    echo "-- $m ($(du -h "$M/$m.gguf" | cut -f1))"
    llama-bench -m "$M/$m.gguf" -ngl 99 -fa 1 -p 512 -n 128 -r 2 -o md 2>&1 | grep -E "^\|" | tail -2
done

echo; echo "== long context: gpt-oss-20b at 32k depth"
llama-bench -m "$M/gpt-oss-20b-mxfp4.gguf" -ngl 99 -fa 1 -p 512 -n 64 -d 32768 -r 1 -o md 2>&1 | grep -E "^\||error|fail" | tail -3

echo; echo "== $(date -Is) done"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv
dmesg 2>/dev/null | grep -i xid | tail -3
