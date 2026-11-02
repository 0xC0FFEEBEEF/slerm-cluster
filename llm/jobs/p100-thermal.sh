#!/bin/bash
#SBATCH --job-name=p100-thermal
#SBATCH --partition=gpu
#SBATCH --nodelist=slerm-node04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/data/llm/logs/p100thermal-%j.out

# Can iDRAC's own fan curve cool a passive P100 with no help? (2026-10-04)
# iDRAC has no sensor for the card (inlet/exhaust/CPU only), so this measures
# indirect cooling. Run it with gpu-fan-control STOPPED on the head; this job
# deliberately does not run gpu-publish.py, so nothing but iDRAC reacts.
#
# Worst case load: sustained prompt processing (compute-bound, sits at the
# 250 W cap). The card slows down at 82 C and SHUTS DOWN at 85 C, so the load
# is killed at P100_ABORT_C (75) or on any thermal slowdown flag.

set -uo pipefail
export LLM_BUILD=/data/llm/build-pascal
source /data/llm/bin/llm-env.sh
LIMIT=${P100_ABORT_C:-75}
DUR=${P100_LOAD_S:-900}

llama-bench -m "$LLM_ROOT/models/qwen3-14b.gguf" -ngl 99 -fa 1 -p 2048 -n 0 -r 2000 >/dev/null 2>&1 &
LOAD=$!
trap 'kill $LOAD 2>/dev/null' EXIT
start=$SECONDS
peak=0
echo "time      +s   temp  power  sm-clock  throttle"
while kill -0 $LOAD 2>/dev/null; do
    IFS=', ' read -r t p sm thr < <(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits)
    el=$((SECONDS - start))
    (( t > peak )) && peak=$t
    printf "%s %4d  %3sC  %5sW  %5sMHz  %s\n" "$(date +%T)" "$el" "$t" "$p" "$sm" "$thr"
    # 0x8 HW slowdown, 0x20 SW thermal slowdown, 0x40 HW thermal slowdown
    if (( t >= LIMIT )) || (( (thr & 0x68) != 0 )); then
        echo "ABORT at ${t}C (throttle $thr) after ${el}s"; break
    fi
    (( el >= DUR )) && { echo "held ${DUR}s without reaching ${LIMIT}C"; break; }
    sleep 5
done
kill $LOAD 2>/dev/null; wait $LOAD 2>/dev/null
echo "peak ${peak}C"
for i in 1 2 3 4 5 6; do
    sleep 10
    echo "cooldown +$((i*10))s $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader)"
done
