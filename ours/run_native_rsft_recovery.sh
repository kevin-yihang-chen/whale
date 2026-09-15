#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:20:00
#SBATCH --job-name=whale-rsft-recovery
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/native-rsft-recovery-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4: one cached interrupted step, original actor/optimizer/checkpoint transport.
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export RSFT_BASE_MODEL="$PWD/data/models/qwen3.5-4b-851bf6e"
export RSFT_TRAIN_FILES="$PWD/data/chess-bootstrap-v1/chunk-0.parquet"
export RSFT_VAL_FILES="$RSFT_TRAIN_FILES"
export RSFT_ROLLOUT_N=8 RSFT_AGENT_NUM_WORKERS=4
export RSFT_GENERATE_TIMEOUT_S=4800 RSFT_TEST_FREQ=-1 RSFT_WEIGHT_BUCKET_MB=3072
export RSFT_RUN_NAME="replay-${SLURM_JOB_ID:-cpu-preflight}"
export WHALE_RSFT_REPLAY_PLAN="$PWD/results/native-rsft-recovery-plan-20260909-v3.json"
rsft_recovery_overrides=(
  'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}'
  'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.recovered_training_bootstrap.prepare_worker'
  "+ray_kwargs.ray_init.runtime_env.env_vars.WHALE_RSFT_REPLAY_PLAN=$WHALE_RSFT_REPLAY_PLAN"
)
if [[ "${1:-}" == --cfg ]]; then
  exec bash ours/run_native_rsft.sh "${rsft_recovery_overrides[@]}" "$@"
fi
if [[ $# != 0 ]]; then
  echo 'Usage: run_native_rsft_recovery.sh (or --cfg job --resolve)' >&2
  exit 2
fi
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
rsft_effective_config="results/native-rsft-recovery-effective-${SLURM_JOB_ID:?}.log"
bash ours/run_native_rsft.sh "${rsft_recovery_overrides[@]}" --cfg job --resolve > "$rsft_effective_config" 2>&1
data/training-runtime-v1/bin/python -m ours.recovery_gate --plan "$WHALE_RSFT_REPLAY_PLAN" --effective-config "$rsft_effective_config"
exec bash ours/run_native_rsft.sh "${rsft_recovery_overrides[@]}"
