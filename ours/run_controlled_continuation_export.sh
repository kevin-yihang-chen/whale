#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=whale-continuation-export
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/continuation-checkpoint-export-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4 canonical serialization and full parameter audit execute on CPU.
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
test "$#" -eq 1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
exec data/training-runtime-v1/bin/python -m ours.controlled_continuation_export --phase run --plan "$1"
