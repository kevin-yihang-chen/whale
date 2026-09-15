#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h800:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:40:00
#SBATCH --job-name=whale-heldout
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/controlled-heldout-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
test "$#" = 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CHESS_PUZZLE_DEFAULT_MAX_TURNS=9 CHESS_PUZZLE_MAX_TURNS_CAP=18
export CHESS_PUZZLE_FORMAT_RETRIES=1 CHESS_PUZZLE_ILLEGAL_RETRIES=1
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
export TRITON_CACHE_DIR="$PWD/data/runtime-cache/triton"
export VLLM_CACHE_ROOT="$PWD/data/runtime-cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$PWD/data/runtime-cache/flashinfer"
exec data/training-runtime-v1/bin/python -m ours.controlled_heldout --plan "$1" --phase evaluate
