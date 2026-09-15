"""Freeze and launch fresh E4 controlled-pilot training from common theta0.

The weight-only control runs its two native online batches continuously under h0.
This entrypoint does not implement search or silently substitute an HF restart
for native phase resume. All task data, sampling order and shared overrides are
fixed before sampling. Empty accepted batches remain measured empty outcomes.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from .alternation_training import ROOT, environment, write_new
from .audit_native_training_batch import require
from .controlled_conditions import native_training_overrides
from .experiment_randomization import TRIAL_SEEDS, TrialRandomization
from .local_completion import checkpoint_manifest
from .preflight import PIN
from .visual_task import file_sha256

INITIALIZATION = Path('results/canonical-initialization-20260910.json')
SPLIT = Path('results/chess-pilot-manifest-20260909.json')
CPU_NAME = 'controlled-pilot-cpu-preflight'
SOURCES = ('ours/controlled_pilot.py', 'ours/run_controlled_pilot.sh',
    'ours/controlled_conditions.py', 'ours/experiment_randomization.py',
    'ours/controlled_training_bootstrap.py', 'ours/native_fused_chunk.py',
    'ours/native_transport_memory.py', 'ours/compact_recorded_training_bootstrap.py',
    'ours/native_training_trace.py', 'ours/compact_native_batch.py', 'ours/run_native_rsft.sh',
    'ours/training_bootstrap.py', 'ours/alternation_training.py', 'ours/rsft_pilot_gate.py',
    'ours/local_completion.py', 'ours/visual_task.py', 'ours/audit_native_training_batch.py',
    'ours/audit_alternation_training.py', 'ours/controlled_pilot_protocol.md')


def training_overrides(seed):
    # Disable the engineering wrapper's one-batch override. These are native
    # batch steps; accepted-row count determines actual optimizer minibatches.
    return [*native_training_overrides(seed), 'trainer.online_rsft.iterations=0',
            'trainer.total_training_steps=2', 'trainer.max_actor_ckpt_to_keep=2']


def launch_environment(plan, run_name):
    env = environment(plan['model'], plan['harness'], plan['dataset'], run_name)
    env.update(TrialRandomization(plan['seed']).environment())
    return env


def resolve(plan, run_name):
    from omegaconf import OmegaConf
    raw = subprocess.check_output(['bash', str(ROOT / 'ours/run_native_rsft.sh'),
        *training_overrides(plan['seed']), '--cfg', 'job', '--resolve'], cwd=ROOT,
        env=launch_environment(plan, run_name), text=True, stderr=subprocess.STDOUT)
    marker = 'model_engine: dp\n'
    require(marker in raw, 'Missing native resolved configuration')
    config = OmegaConf.create(raw[raw.index(marker):])
    return raw, OmegaConf.to_container(config, resolve=True)


def native_order(config):
    from unittest.mock import patch
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    cfg = OmegaConf.create(config)
    model = cfg.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    harness = cfg.ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_HARNESS_PATH
    with patch.dict(os.environ, HARNESS_PATH=harness, CHESS_PUZZLE_HARNESS_PATH=harness):
        dataset = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, processor,
                                    is_train=True, max_samples=cfg.data.train_max_samples)
        ids = [info['puzzle_id'] for info in dataset.dataframe['extra_info']]
        indices = list(create_rl_sampler(cfg.data, dataset))
    require(len(ids) == len(set(ids)) == len(indices) == 128, 'Incomplete pilot dataset after native filtering')
    return {'native_dataset_class': type(dataset).__name__, 'native_filtered_ids': ids,
            'native_sampler_indices': indices, 'ordered_ids': [ids[index] for index in indices],
            'planned_batch_ids': [[ids[index] for index in indices[begin:begin+8]] for begin in (0, 8)]}


def prepare(path, seed):
    TrialRandomization(seed)
    require(seed in TRIAL_SEEDS, 'Outside declared pilot seeds')
    init = json.loads(INITIALIZATION.read_text())
    require(init['status'] == 'PASS' and init['optimizer_steps'] == init['new_model_calls'] == 0 and
            init['audit']['exact_source_bf16_cast'], 'No verified untrained common initialization')
    split = json.loads(SPLIT.read_text())['splits']
    plan = {'kind': 'controlled_weight_only_pilot_plan', 'condition': 'weight_only',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
        'model': init['exported']['path'], 'initial_manifest': init['exported'],
        'harness': str(ROOT / 'upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py'),
        'dataset': split['train']['path'], 'initialization_report': str(INITIALIZATION),
        'split_manifest': str(SPLIT), 'veto_mode': 'off', 'native_batch_steps': 2,
        'fresh_trajectories': 128, 'rollouts_per_prompt': 8,
        'max_assistant_output_tokens': 128 * 8129,
        'max_policy_calls': 128 * 18, 'proposer_calls': 0,
        'randomization': TrialRandomization(seed).metadata(),
        'resources': {'gpus': 2, 'gpu_type': 'H800', 'cpus': 16, 'ram_gib': 160,
                      'time_limit_seconds': 5400, 'max_gpu_hours': 3.},
        'resource_decision': {'forecast_hkt': '2026-09-10T00:52:49+08:00',
            'single_gpu': 'Not supported by this validated disaggregated actor/rollout configuration.',
            'two_gpus': 'Immediate forecast; two native batches estimated 60-75 minutes / 2-2.5GPUh.',
            'four_gpus': 'Forecast2026-09-11T18:57:22; independent-trial Ray isolation is not yet validated.',
            'merge_risk': 'One trainer owns both sequential batches; no generation shards or prefix reuse.'},
        'limitations': ['128/32/64 pilot split and two online batches, not original paper-scale reproduction.',
            'Two batch steps do not imply two optimizer steps; empty accepted batches are valid measured outcomes.',
            'Weight-only keeps optimizer and sampling process alive between batches; no phase restart.',
            'No heldout scores or F0 success are inferred from training metrics.',
            'Seed propagation into a complete Ray training run is first exercised by this pilot.']}
    raw, config = resolve(plan, CPU_NAME)
    config_path = path.with_suffix('.yaml')
    with config_path.open('x') as stream:
        stream.write(raw)
    plan['resolved_config'] = str(config_path)
    plan['native_dataset_order'] = native_order(config)
    require(set(plan['native_dataset_order']['native_filtered_ids']) == set(split['train']['puzzle_ids']),
            'Native dataset differs from declared training split')
    old = json.loads(Path('results/alternation-recovery-plan-20260909-v3.json').read_text())
    sources = set(SOURCES) | {str(INITIALIZATION), str(SPLIT), str(config_path), plan['harness'], plan['dataset']}
    sources.update(name for name in old['source_sha256'] if name.startswith(('ours/', 'upstream/')))
    sources.update(str(p) for p in Path('ours/compat').rglob('*.py'))
    plan['source_sha256'] = {name: file_sha256(ROOT / name) for name in sorted(sources)}
    plan['versions'] = {n: version(n) for n in ('torch', 'vllm', 'transformers', 'ray', 'numpy', 'chess', 'pandas', 'pyarrow', 'safetensors', 'cupy-cuda13x')}
    write_new(path, plan)


def check(path):
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'controlled_weight_only_pilot_plan' and plan['condition'] == 'weight_only', 'Wrong trial kind')
    require(plan['seed'] in TRIAL_SEEDS and plan['veto_mode'] == 'off', 'Wrong trial controls')
    require(plan['native_batch_steps'] == 2 and plan['fresh_trajectories'] == 128 and
            plan['proposer_calls'] == 0 and plan['max_assistant_output_tokens'] == 1040512, 'Changed trial budget')
    require(subprocess.check_output(['git', '-C', 'upstream/WHALE', 'rev-parse', 'HEAD'], text=True).strip() == PIN,
            'Wrong native revision')
    require(not subprocess.check_output(['git', '-C', 'upstream/WHALE', 'status', '--porcelain'], text=True).strip(),
            'Modified upstream')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, f'Changed controlled-trial source: {name}')
    require({n: version(n) for n in plan['versions']} == plan['versions'], 'Changed runtime')
    require(checkpoint_manifest(Path(plan['model'])) == plan['initial_manifest'], 'Changed common initialization')
    split = json.loads(Path(plan['split_manifest']).read_text())['splits']
    train, mh, test = [set(split[key]['puzzle_ids']) for key in ('train', 'mh_val', 'test')]
    require(not (train & mh or train & test or mh & test), 'Overlapping data roles')
    require(set(plan['native_dataset_order']['native_filtered_ids']) == train, 'Wrong filtered train coverage')
    from .rsft_pilot_gate import check_tensor_bucket
    check_tensor_bucket(plan['model'], 'float32', 3072)
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    require(not (args.prepare and args.run), 'Separate preparation from execution')
    if args.prepare:
        prepare(args.plan, args.seed)
    else:
        from omegaconf import OmegaConf
        plan = check(args.plan)
        run_name = f"controlled-weight-only-seed{plan['seed']}-{os.environ['SLURM_JOB_ID']}" if args.run else CPU_NAME
        if args.run:
            import torch
            require(torch.cuda.device_count() == 2 and all('H800' in torch.cuda.get_device_name(i) for i in range(2)),
                    'Expected two H800 devices')
            require(not (ROOT / 'data/native-rsft' / run_name).exists(), 'Preserve existing training directory')
        raw, config = resolve(plan, run_name)
        expected_raw = Path(plan['resolved_config']).read_text().replace(CPU_NAME, run_name)
        marker = 'model_engine: dp\n'
        expected = OmegaConf.to_container(OmegaConf.create(expected_raw[expected_raw.index(marker):]), resolve=True)
        require(config == expected, 'Actual native Hydra configuration differs from frozen trial')
        require(native_order(config) == plan['native_dataset_order'], 'Actual data filtering or sampling order differs')
        report = {'kind': 'controlled_pilot_preflight', 'status': 'PASS', 'plan_sha256': file_sha256(args.plan),
                  'run_name': run_name, 'seed': plan['seed'], 'native_batch_steps': 2, 'new_model_calls': 0}
        if args.run:
            output = Path(f"results/controlled-pilot-{os.environ['SLURM_JOB_ID']}")
            output.mkdir(exist_ok=False)
            write_new(output / 'start.json', report)
            (output / 'effective-config.yaml').write_text(raw)
            os.execvpe('bash', ['bash', str(ROOT / 'ours/run_native_rsft.sh'), *training_overrides(plan['seed'])],
                       launch_environment(plan, run_name))
        print(json.dumps(report), flush=True)
