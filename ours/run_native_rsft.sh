#!/usr/bin/env bash
#SBATCH --partition=q-hgpu-small
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=whale-native-rsft
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/native-rsft-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4: original success-filtered response-token SFT and native weight transport.
# Defaults retain the earlier engineering configuration. Explicit RSFT_* inputs
# select a frozen pilot; an empty update must not be counted as a pass.
set -euo pipefail
rsft_root=/userhome/cs3/yihangc/Documents/whale-delta
export RSFT_RUNTIME_ROOT="$rsft_root"
export PATH="$rsft_root/data/training-runtime-v1/bin:$PATH"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$rsft_root/ours/compat:$rsft_root/upstream/WHALE/domains/chess_puzzles:$rsft_root"
export TRITON_CACHE_DIR="$rsft_root/data/runtime-cache/triton"
export VLLM_CACHE_ROOT="$rsft_root/data/runtime-cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$rsft_root/data/runtime-cache/flashinfer"
export HF_HOME="$rsft_root/data/runtime-cache/huggingface"
export PROJECT_DIR="$rsft_root/upstream/WHALE/domains/chess_puzzles"
export OUTPUT_ROOT="$rsft_root/data/native-rsft"
export RUN_NAME="${RSFT_RUN_NAME:-engineering-${SLURM_JOB_ID:-cpu-config}}"
export BASE_MODEL="${RSFT_BASE_MODEL:-$rsft_root/data/models/qwen3.5-2b-15852e8}"
export TRAIN_FILES="${RSFT_TRAIN_FILES:-$rsft_root/data/chess-engineering-v3/engineering.parquet}"
export VAL_FILES="${RSFT_VAL_FILES:-$TRAIN_FILES}"
export HARNESS_PATH="${RSFT_HARNESS_PATH:-$PROJECT_DIR/environments/chess_puzzle/base_harness.py}"
export CHESS_PUZZLE_ALGO=rsft NNODES=1 N_GPUS_PER_NODE=1 ROLLOUT_NNODES=1 ROLLOUT_GPUS_PER_NODE=1
export ROLLOUT_N="${RSFT_ROLLOUT_N:-1}" VAL_ROLLOUT_N=1
export AGENT_NUM_WORKERS="${RSFT_AGENT_NUM_WORKERS:-2}" REWARD_NUM_WORKERS=1
export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 PPO_MICRO_PER_GPU=1
export ENABLE_ONLINE_RSFT=true RSFT_SFT_EPOCHS=1 RSFT_SFT_MINI_BATCH_SIZE=8 RSFT_SFT_MICRO_PER_GPU=1
export RSFT_SCORE_THRESHOLD=0.5 ACTOR_LR=1e-7
export MAX_PROMPT_LENGTH=4096 MAX_TOTAL_RESPONSE_LENGTH=16384 ASSISTANT_TOKEN_BUDGET=8129 POLICY_MAX_TOKENS=8129
export MAX_MODEL_LEN=32768 MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=8
export CHESS_PUZZLE_DEFAULT_MAX_TURNS=9 CHESS_PUZZLE_MAX_TURNS_CAP=18 MAX_TURNS=9
export CHESS_PUZZLE_ENABLE_THINKING=True FORMAT_RETRIES=1 ILLEGAL_RETRIES=1
export ROLLOUT_TEMPERATURE=1.0 ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=20
export ENABLE_CHUNKED_PREFILL=True ENABLE_PREFIX_CACHING=True ENFORCE_EAGER=True
export USE_REMOVE_PADDING=False ENABLE_GRADIENT_CHECKPOINTING=True
export GPU_MEM_UTIL=0.6 ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB="${RSFT_WEIGHT_BUCKET_MB:-128}"
export TRAINER_SAVE_FREQ=1 TRAINER_TEST_FREQ="${RSFT_TEST_FREQ:-1}" TRAINER_TOTAL_EPOCHS=1
export TRAINER_VAL_BEFORE_TRAIN=False TRAINER_LOGGER=console RESUME_MODE=disable
export DUMP_ROLLOUTS=true DUMP_VAL_ROLLOUTS=true
export VERL_TASK_RUNNER_START_TIMEOUT_S=240 RAY_WAIT_REGISTER_CENTER_TIMEOUT_S=240
export VERL_AGENT_LOOP_WORKER_STARTUP_TIMEOUT_S=240 RAY_worker_register_timeout_seconds=240
export VERL_VLLM_GENERATE_TIMEOUT_S="${RSFT_GENERATE_TIMEOUT_S:-240}"

# Redirect only the original trainer entrypoint to the declared namespace repair.
# Keep all native shell defaults/overrides and trainer/actor source unchanged.
python3() {
  if [[ "${1:-}" == "$PROJECT_DIR/scripts/chess_puzzle_preamble_disagg_rsft.py" ]]; then
    shift
    "$RSFT_RUNTIME_ROOT/data/training-runtime-v1/bin/python" -m ours.training_bootstrap "$@"
  else
    "$RSFT_RUNTIME_ROOT/data/training-runtime-v1/bin/python" "$@"
  fi
}
export -f python3
bash "$PROJECT_DIR/scripts/train_chess_puzzle_multinode_disagg.sh" \
  +trainer.online_rsft.iterations=1 \
  +ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.training_bootstrap.prepare_worker \
  +ray_kwargs.ray_init.num_cpus=16 \
  +ray_kwargs.ray_init.include_dashboard=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  "hydra.searchpath=[file://$rsft_root/ours/config]" \
  "hydra.run.dir=$OUTPUT_ROOT/$RUN_NAME/hydra" \
  "$@"
