#!/bin/bash
# Stage the two Qwen3-30B-A3B MoEs at both Q3_K_M and Q4_K_M for comparison.
cd /data/llm/models
fetch() {  # repo file outname
  echo "[$(date +%H:%M:%S)] $3"
  curl -sL --retry 5 -C - -o "$3" "https://huggingface.co/$1/resolve/main/$2"
  echo "[$(date +%H:%M:%S)] done $3 -> $(stat -c %s "$3" 2>/dev/null) bytes"
}
fetch unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF  Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf  qwen3-30b-a3b-2507-q3km.gguf
fetch unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf  qwen3-30b-a3b-2507-q4km.gguf
fetch unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf qwen3-coder-30b-a3b-q3km.gguf
fetch unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf qwen3-coder-30b-a3b-q4km.gguf
echo "[$(date +%H:%M:%S)] ALL 4 COMPLETE"
