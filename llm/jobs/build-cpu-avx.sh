#!/bin/bash
#SBATCH --job-name=llm-build-avx
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/data/llm/logs/buildavx-%j.out

# CPU-only build into build-cpu-avx/ with an IVY BRIDGE (AVX-only) ISA baseline.
#
# WHY THIS EXISTS: build-cpu/ is built with -DGGML_AVX2=ON -DGGML_FMA=ON ("same
# Haswell ISA baseline as the CUDA build"). slerm-node05 (deadlock) is a Xeon
# E5-1620 v2 -- Ivy Bridge -- which has avx + f16c but NO avx2 and NO fma.
# Running build-cpu/bin/ggml-rpc-server there dies instantly with SIGILL
# (exit 132, core dumped). Verified 2026-09-09.
#
# So node05 needs its own RPC server binary. Everything else about the pool is
# unchanged; only the ISA baseline differs.
#
#   sbatch -w slerm-node05 /data/llm/jobs/build-cpu-avx.sh
#
# Build it ON node05: GGML_NATIVE=OFF means the build host does not affect the
# output, but node05 is idle and node01/02 are fully allocated to rpccpu workers.

set -euo pipefail
source /data/llm/bin/llm-env.sh

cmake -S "$LLM_ROOT/src/llama.cpp" -B "$LLM_ROOT/build-cpu-avx" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=ON -DGGML_BMI2=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build "$LLM_ROOT/build-cpu-avx" -j "${SLURM_CPUS_ON_NODE:-8}"

echo "=== done ==="
ls -1 "$LLM_ROOT/build-cpu-avx/bin" | grep -E "^(ggml-rpc-server|llama-server|llama-cli|llama-bench)$" || true

echo "=== ISA check: avx2/fma/bmi2 must all be OFF ==="
grep -iE "^GGML_(AVX|AVX2|FMA|F16C|BMI2):" "$LLM_ROOT/build-cpu-avx/CMakeCache.txt"

# BMI2 is the one that bites: it defaults to ${INS_ENB} just like AVX2, and
# ggml_cpu_init() itself emits `shlx`, so EVERY binary SIGILLs at startup on a
# pre-Haswell CPU -- even `llama-cli --version`. Ivy Bridge has no BMI1/BMI2.
echo "=== smoke test on this host ($(hostname -s)) ==="
rc=0
"$LLM_ROOT/build-cpu-avx/bin/ggml-rpc-server" --help >/dev/null 2>&1 || rc=$?
if [ "$rc" -ge 128 ]; then
  echo "FAILED: still dies with signal $(kill -l $((rc-128)) 2>/dev/null) (exit $rc)"
  exit 1
fi
echo "OK: ggml-rpc-server runs on $(hostname -s) (exit $rc)"
