#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h800:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=whale-alternation-rsft
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/alternation-rsft-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
test "$#" = 1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
exec data/training-runtime-v1/bin/python -m ours.alternation_training --plan "$1" --run
