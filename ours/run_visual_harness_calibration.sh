#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=veto-visual-calibration
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/visual-calibration-%j.log
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
test "$#" = 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset HARNESS_PATH CHESS_PUZZLE_HARNESS_PATH _HARNESS_SYSTEM_PROMPT _HARNESS_USER_PROMPT_TEMPLATE WHALE_VISUAL_HARNESS_PATH
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE=2 VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S=240
export PYTHONPATH="$PWD/data/visual-runtime-overlay-v1:$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
export TRITON_CACHE_DIR="$PWD/data/runtime-cache/triton"
export VLLM_CACHE_ROOT="$PWD/data/runtime-cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$PWD/data/runtime-cache/flashinfer"
exec data/training-runtime-v1/bin/python -m ours.visual_harness_calibration --phase run --plan "$1"
