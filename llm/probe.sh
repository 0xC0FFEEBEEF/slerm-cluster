#!/bin/bash
#SBATCH --job-name=gpuprobe
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --output=/data/llm/logs/probe-%j.out

echo "host: $(hostname)"
echo "--- nvidia-smi ---"
nvidia-smi
echo "--- query ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "--- nvcc ---"
which nvcc && nvcc --version || echo "no nvcc"
echo "--- os/python/toolchain ---"
cat /etc/os-release | head -2
python3 -V; gcc --version | head -1; cmake --version 2>/dev/null | head -1 || echo "no cmake"
echo "--- cpu ---"
lscpu | grep -E "Model name|^CPU\(s\)|MHz"
echo "--- mem ---"
free -g
echo "--- local disk /tmp ---"
df -h /tmp /var/tmp 2>/dev/null
