#!/bin/bash
#SBATCH --job-name=p100-thermal2
#SBATCH --partition=gpu
#SBATCH --nodelist=slerm-node04
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=00:35:00
#SBATCH --output=/data/llm/logs/p100thermal2-%j.out

# Two passive P100s at full load at once, cooled by iDRAC alone (2026-10-04).
# gpu-fan-control is disabled; this job runs no gpu-publish.py, so nothing but
# iDRAC reacts. One card alone plateaued at 72 C (p100-thermal.sh, job 4421).
# The cards slow down at 82 C and SHUT DOWN at 85 C, so the load is killed at
# P100_ABORT_C (75) on either card or on any thermal slowdown flag.

set -uo pipefail
export LLM_BUILD=/data/llm/build-pascal
source /data/llm/bin/llm-env.sh
LIMIT=${P100_ABORT_C:-75}
DUR=${P100_LOAD_S:-900}
M=$LLM_ROOT/models/qwen3-14b.gguf

CUDA_VISIBLE_DEVICES=0 llama-bench -m "$M" -ngl 99 -fa 1 -p 2048 -n 0 -r 2000 >/dev/null 2>&1 &
L0=$!
CUDA_VISIBLE_DEVICES=1 llama-bench -m "$M" -ngl 99 -fa 1 -p 2048 -n 0 -r 2000 >/dev/null 2>&1 &
L1=$!
trap 'kill $L0 $L1 2>/dev/null' EXIT
start=$SECONDS
peak=0
echo "time      +s   gpu0: temp power clock | gpu1: temp power clock | throttle0 throttle1"
while kill -0 $L0 2>/dev/null || kill -0 $L1 2>/dev/null; do
    mapfile -t rows < <(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits)
    IFS=', ' read -r t0 p0 c0 r0 <<< "${rows[0]}"
    IFS=', ' read -r t1 p1 c1 r1 <<< "${rows[1]}"
    el=$((SECONDS - start))
    hot=$(( t0 > t1 ? t0 : t1 ))
    (( hot > peak )) && peak=$hot
    printf "%s %4d   %3sC %5sW %4s | %3sC %5sW %4s | %s %s\n" "$(date +%T)" "$el" "$t0" "$p0" "$c0" "$t1" "$p1" "$c1" "$r0" "$r1"
    # 0x8 HW slowdown, 0x20 SW thermal slowdown, 0x40 HW thermal slowdown
    if (( hot >= LIMIT )) || (( (r0 & 0x68) != 0 )) || (( (r1 & 0x68) != 0 )); then
        echo "ABORT at ${hot}C (throttle $r0 $r1) after ${el}s"; break
    fi
    (( el >= DUR )) && { echo "held ${DUR}s without reaching ${LIMIT}C"; break; }
    sleep 5
done
kill $L0 $L1 2>/dev/null; wait 2>/dev/null
echo "peak ${peak}C"
for i in 1 2 3 4 5 6; do
    sleep 10
    echo "cooldown +$((i*10))s $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader | tr '\n' ' ')"
done
