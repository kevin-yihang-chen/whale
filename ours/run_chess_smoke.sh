#!/usr/bin/env bash
#SBATCH --partition=q-h800
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --job-name=whale-chess-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/chess-smoke-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/userhome/cs3/yihangc/Documents/whale-delta/ours/compat:/userhome/cs3/yihangc/Documents/whale-delta/data/runtime-deps/chess-1.11.2:/userhome/cs3/yihangc/Documents/whale-delta/upstream/WHALE/domains/chess_puzzles:/userhome/cs3/yihangc/Documents/whale-delta
exec /userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python -m ours.run_chess_smoke \
  --plan results/chess-smoke-plan-20260909.json \
  --output "data/chess-smoke-${SLURM_JOB_ID:?}"
