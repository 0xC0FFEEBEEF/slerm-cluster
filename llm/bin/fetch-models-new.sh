#!/usr/bin/env bash
# Fetch the 2026-09 MoE candidates. Same conventions as bin/fetch-models.sh:
# detached, resumable, skips anything already at its CDN-reported size.
#
# WHY THESE: tonight's benchmarks showed the split on this cluster is by
# ARCHITECTURE, not size -- a 46 GiB MoE ran at 5.47 tok/s while a 16 GiB DENSE
# model managed 1.72. Only ~3-4B params are active per token in an MoE, so the
# CPU nodes only move the active slice. Every pick below is MoE, and every arch
# is confirmed present in this build's llama-arch.cpp (148 archs supported):
#
#   Qwen3.6-35B-A3B   -> arch qwen35moe   (256 experts, ~3B active)
#   gemma-4-26B-A4B   -> arch gemma4      (~4B active)
#
# Qwen3.6-35B-A3B is specifically the MoE counterpart of the DENSE Qwen3.6-27B
# that measured 1.72 tok/s and got archived. Same model generation, the
# architecture that actually suits this hardware.
#
# Two quants of the Qwen are fetched deliberately: Q4_K_XL is the daily driver,
# Q6_K_XL tests whether deadlock's added capacity buys real quality headroom, and
# the pair answers what that quality costs in tok/s. That is the same question
# the README already asks of Q3 vs Q4.
set -uo pipefail
DEST=/data/llm/models

# repo | remote filename | local name | expected bytes (0 = ask the CDN)
SPECS=(
  "unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf|gemma4-26b-a4b-ud-q6kxl.gguf|0"
  "unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|qwen3.6-35b-a3b-ud-q4kxl.gguf|0"
  "unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf|qwen3.6-35b-a3b-ud-q6kxl.gguf|0"
)

for spec in "${SPECS[@]}"; do
  IFS="|" read -r repo src dst want <<< "$spec"
  url="https://huggingface.co/${repo}/resolve/main/${src}"
  out="${DEST}/${dst}"

  remote=$(curl -sIL "$url" | awk -F": " 'tolower($1)=="content-length"{n=$2} END{gsub(/\r/,"",n); print n}')
  [ -n "$remote" ] && want="$remote"
  echo "=== $(date -Is) ${dst}  target=${want} bytes"

  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$want" != "0" ] && [ "$have" = "$want" ]; then
    echo "--- already complete, skipping"
    continue
  fi

  for attempt in $(seq 1 20); do
    curl -L --fail --retry 5 --retry-delay 10 --retry-all-errors \
         -C - --speed-limit 20480 --speed-time 180 \
         -o "$out" "$url"
    rc=$?
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$rc" = "0" ] && { [ "$want" = "0" ] || [ "$have" = "$want" ]; }; then
      echo "=== $(date -Is) OK ${dst} ${have} bytes"
      break
    fi
    echo "--- attempt ${attempt} rc=${rc} have=${have}/${want}, retrying"
    sleep 15
  done
done

echo "=== $(date -Is) FETCH COMPLETE"
ls -la "$DEST" | grep -E "gemma4|qwen3.6-35b"
df -h /export | tail -1
