#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=whale-bootstrap
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/chess-bootstrap-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4: native on-policy success sampling; no weight update is claimed here.
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
export TRITON_CACHE_DIR="$PWD/data/runtime-cache/triton"
export VLLM_CACHE_ROOT="$PWD/data/runtime-cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$PWD/data/runtime-cache/flashinfer"
export CHESS_PUZZLE_DEFAULT_MAX_TURNS=9 CHESS_PUZZLE_MAX_TURNS_CAP=18
export CHESS_PUZZLE_FORMAT_RETRIES=1 CHESS_PUZZLE_ILLEGAL_RETRIES=1
export CHESS_PUZZLE_ASSISTANT_TOKEN_BUDGET=8129 CHESS_PUZZLE_POLICY_MAX_TOKENS=8129
exec data/training-runtime-v1/bin/python -m ours.chess_bootstrap \
  --plan "${1:?frozen plan is required}" --chunk "${2:-0}" \
  --output "data/chess-bootstrap-${SLURM_JOB_ID:?}"
