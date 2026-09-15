#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=whale-rsft-pilot
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/native-rsft-pilot-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4: original trainer/actor, fresh on-policy collection, one optimizer phase.
# The bootstrap archive establishes feasibility; it is not silently replayed
# as new model calls here. All new sampling and any empty step count as costs.
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export RSFT_BASE_MODEL="$PWD/data/models/qwen3.5-4b-851bf6e"
export RSFT_TRAIN_FILES="$PWD/data/chess-bootstrap-v1/chunk-0.parquet"
export RSFT_VAL_FILES="$RSFT_TRAIN_FILES"
export RSFT_ROLLOUT_N=8 RSFT_AGENT_NUM_WORKERS=4
# All64 native tasks are submitted together to eight generation slots. The
# response timeout includes queue waiting; the Slurm90-minute cap still applies.
export RSFT_GENERATE_TIMEOUT_S=4800 RSFT_TEST_FREQ=-1
# Restore the native launcher's 3072 MiB transport bucket. Its NCCL engine
# sends complete fp32 tensors, including the 2425 MiB embedding matrix.
export RSFT_WEIGHT_BUCKET_MB=3072
export RSFT_RUN_NAME="pilot-${SLURM_JOB_ID:-cpu-preflight}"
# E4: use the native AdamW scalar path to avoid foreach temporary-list peaks.
# Record exact online batches so a failed optimizer need not lose replay evidence.
rsft_pilot_overrides=(
  'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}'
  'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.recorded_training_bootstrap.prepare_worker'
)
if [[ "${1:-}" == --cfg ]]; then
  exec bash ours/run_native_rsft.sh "${rsft_pilot_overrides[@]}" "$@"
fi
if [[ $# != 1 ]]; then
  echo 'Usage: run_native_rsft_pilot.sh FROZEN_PLAN.json (or --cfg job --resolve for CPU configuration)' >&2
  exit 2
fi
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
rsft_effective_config="results/native-rsft-pilot-effective-${SLURM_JOB_ID:?}.log"
bash ours/run_native_rsft.sh "${rsft_pilot_overrides[@]}" --cfg job --resolve > "$rsft_effective_config" 2>&1
data/training-runtime-v1/bin/python -m ours.rsft_pilot_gate --plan "$1" --effective-config "$rsft_effective_config"
exec bash ours/run_native_rsft.sh "${rsft_pilot_overrides[@]}"
