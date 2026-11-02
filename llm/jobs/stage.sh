#!/bin/bash
#SBATCH --job-name=llm-stage
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/data/llm/logs/stage-%j.out

# Staging (the panel's Stage button): copy the models in run/staged.json onto
# this node ahead of use -- a full local copy on the router node, pre-written
# rpc cache files on a worker. Submitted by slerm-orch with --partition and
# --nodelist; exits once this node holds everything staged for it. Small on
# purpose so it fits beside a router or GPU worker; slerm-orch cancels it
# before starting a router whose CPU workers need the whole node (the worker's
# own model-cache loop then carries on staging).
set -uo pipefail
exec python3 /data/llm/bin/model-cache.py stage
