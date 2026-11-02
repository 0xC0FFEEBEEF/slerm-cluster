#!/usr/bin/env bash
# Detached, resumable model fetcher. Re-running continues any partial file and
# skips anything already at its expected size.
# Ordered smallest-first so something usable lands early.
set -uo pipefail
DEST=/data/llm/models

# repo : remote filename : local name : expected bytes
SPECS=(
  "unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-UD-Q4_K_XL.gguf|qwen3.6-27b-ud-q4kxl.gguf|17612793984"
  "unsloth/Qwen3.6-27B-GGUF|mmproj-F16.gguf|qwen3.6-27b-mmproj-f16.gguf|0"
  "unsloth/Qwen3-Coder-Next-GGUF|Qwen3-Coder-Next-UD-Q2_K_XL.gguf|qwen3-coder-next-ud-q2kxl.gguf|0"
  "unsloth/Qwen3-Coder-Next-GGUF|Qwen3-Coder-Next-UD-Q4_K_XL.gguf|qwen3-coder-next-ud-q4kxl.gguf|0"
  "ggml-org/gpt-oss-120b-GGUF|gpt-oss-120b-MXFP4.gguf|gpt-oss-120b-mxfp4.gguf|0"
)

# Optional: a list file (one repo|remote|local|bytes per line, # comments) replaces SPECS.
if [ -n "${1:-}" ]; then
  mapfile -t SPECS < <(grep -v -e "^#" -e "^[[:space:]]*$" "$1")
fi

for spec in "${SPECS[@]}"; do
  IFS="|" read -r repo src dst want <<< "$spec"
  url="https://huggingface.co/${repo}/resolve/main/${src}"
  out="${DEST}/${dst}"

  # authoritative size from the CDN, so we know when we are actually done
  remote=$(curl -sIL "$url" | awk -F": " "tolower(\$1)==\"content-length\"{n=\$2} END{gsub(/\r/,\"\",n); print n}")
  [ -n "$remote" ] && want="$remote"
  echo "=== $(date -Is) ${dst}  target=${want} bytes"

  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$want" != "0" ] && [ "$have" = "$want" ]; then
    echo "--- already complete, skipping"
    continue
  fi

  # No curl --retry: it restarts from the offset this curl began at (0 on a fresh file), and the HF CDN closes
  # the connection every ~17-20 min, so 2026-10-08 lost 15 GB twice. This loop resumes from the file size.
  for attempt in $(seq 1 20); do
    curl -L --fail \
         -C - --speed-limit 20480 --speed-time 180 \
         -o "$out" "$url"
    rc=$?
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$rc" = "0" ] && { [ "$want" = "0" ] || [ "$have" = "$want" ]; }; then
      echo "=== $(date -Is) OK ${dst} ${have} bytes"
      break
    fi
    echo "--- attempt ${attempt} rc=${rc} have=${have}/${want}; resuming in 20s"
    sleep 20
  done
done

echo "=== $(date -Is) ALL DONE"

