#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=whale-optimizer-probe
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/native-optimizer-probe-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4 infrastructure: synthetic optimizer-memory check, no actual model training.
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
exec data/training-runtime-v1/bin/python -m ours.probe_native_optimizer --mode gpu \
  --config results/native-rsft-pilot-resolved-20260909-v4.yaml \
  --graph results/native-checkpoint-graph-preflight-20260909.json \
  --output "results/native-optimizer-probe-${SLURM_JOB_ID:?}.json"
