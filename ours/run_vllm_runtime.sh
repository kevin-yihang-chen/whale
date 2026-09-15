#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=whale-vllm-runtime
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/vllm-runtime-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
export TRITON_CACHE_DIR="$PWD/data/runtime-cache/triton"
export VLLM_CACHE_ROOT="$PWD/data/runtime-cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$PWD/data/runtime-cache/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"
exec data/training-runtime-v1/bin/python -m ours.vllm_runtime \
  --plan "${1:-results/chess-vllm-runtime-plan-20260909-v3.json}" \
  --output "data/vllm-runtime-${SLURM_JOB_ID:?}"
