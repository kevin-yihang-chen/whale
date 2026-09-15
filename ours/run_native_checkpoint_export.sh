#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=whale-checkpoint-export
#SBATCH --output=/userhome/cs3/yihangc/Documents/whale-delta/results/native-checkpoint-export-%j.log
#SBATCH --mail-user=kevin.yihang.chan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
# E4 merger/audit execute on CPU. QOSMinGRES requires one allocated GPU, which
# remains hidden from the process; its entire allocation is counted as a cost.
set -euo pipefail
cd /userhome/cs3/yihangc/Documents/whale-delta
if [[ $# != 2 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo 'Usage: run_native_checkpoint_export.sh SOURCE_JOB RECOVERY_PLAN.json' >&2
  exit 2
fi
native_source_job="$1"
native_recovery_plan="$2"
native_export_root="$PWD/data/native-rsft/replay-$native_source_job"
native_actor_dir="$native_export_root/global_step_1/actor"
native_export_dir="$native_export_root/hf-checkpoint-canonical"
test -f "$native_actor_dir/model_world_size_1_rank_0.pt"
test ! -e "$native_export_dir"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN ZHIPU_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY WANDB_API_KEY
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/ours/compat:$PWD/upstream/WHALE/domains/chess_puzzles:$PWD"
data/training-runtime-v1/bin/python - "$native_source_job" "$native_recovery_plan" <<'PY'
import json, math, os, sys
from pathlib import Path
from ours.visual_task import file_sha256
job, plan_path = sys.argv[1], Path(sys.argv[2])
result = json.loads(Path(f'results/native-rsft-recovery-{job}/result.json').read_text())
assert result['completed_optimizer_steps'] == 1
assert math.isfinite(result['gradient_norm']) and result['gradient_norm'] > 0
assert result['plan_sha256'] == file_sha256(plan_path)
state = Path(f'data/native-rsft/replay-{job}/global_step_1/actor/model_world_size_1_rank_0.pt')
report = {'kind': 'native_checkpoint_export_start', 'source_job_id': job,
          'job_id': os.environ['SLURM_JOB_ID'], 'recovery_plan_sha256': file_sha256(plan_path),
          'native_checkpoint_sha256': file_sha256(state), 'new_model_calls': 0,
          'source_sha256': {p: file_sha256(Path(p)) for p in
                           ['ours/run_native_checkpoint_export.sh', 'ours/verify_native_transition.py',
                            'ours/canonical_native_export.py']}}
with Path(f"results/native-checkpoint-export-start-{report['job_id']}.json").open('x') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps(report), flush=True)
PY
data/training-runtime-v1/bin/python -m ours.canonical_native_export \
  --base "$PWD/data/models/qwen3.5-4b-851bf6e" --native "$native_actor_dir" --target "$native_export_dir" \
  --report "results/native-canonical-export-${native_source_job}.json"
data/training-runtime-v1/bin/python -m ours.verify_native_transition \
  --base "$PWD/data/models/qwen3.5-4b-851bf6e" --native "$native_actor_dir" \
  --exported "$native_export_dir" --recovery-plan "$native_recovery_plan" \
  --output "results/native-checkpoint-transition-${native_source_job}.json"
data/training-runtime-v1/bin/python - "$native_source_job" <<'PY'
import json, os, sys
from pathlib import Path
start = json.loads(Path(f"results/native-checkpoint-export-start-{os.environ['SLURM_JOB_ID']}.json").read_text())
result = json.loads(Path(f'results/native-checkpoint-transition-{sys.argv[1]}.json').read_text())
assert start['native_checkpoint_sha256'] == result['native_checkpoint_sha256']
print('Native checkpoint input unchanged throughout export and value audit.', flush=True)
PY
