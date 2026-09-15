#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=whale-chess-decode
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/chess-decode-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/userhome/cs3/yihangc/Documents/whale-delta/ours/compat:/userhome/cs3/yihangc/Documents/whale-delta/data/runtime-deps/chess-1.11.2:/userhome/cs3/yihangc/Documents/whale-delta/upstream/WHALE/domains/chess_puzzles:/userhome/cs3/yihangc/Documents/whale-delta
decode_python=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python
decode_plans=data/chess-decode-study-v1
decode_output="data/chess-decode-${SLURM_JOB_ID:?}"
mkdir "$decode_output"
decode_pids=()
for shard in 0 1; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:h100:1 --mem=24G \
    "$decode_python" -m ours.run_chess_smoke --plan "$decode_plans/plan-${shard}.json" \
    --output "$decode_output/shard-${shard}" > "$decode_output/shard-${shard}.log" 2>&1 &
  decode_pids+=("$!")
done
decode_failed=0
for pid in "${decode_pids[@]}"; do
  if ! wait "$pid"; then decode_failed=1; fi
done
exit "$decode_failed"
