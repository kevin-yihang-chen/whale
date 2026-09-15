#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --job-name=whale-visual-smoke
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/visual-smoke-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/userhome/cs3/yihangc/Documents/whale-delta
exec /userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python -m ours.run_visual_smoke \
  --manifest data/engineering-chart-pairs-v1/manifest.json \
  --model /userhome/cs3/yihangc/Data/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --output "data/visual-smoke-${SLURM_JOB_ID:?}"
