"""One real E4 visual RSFT batch through WHALE's native disaggregated trainer.

This engineering run uses the working direct-answer condition and 32 seeded W
examples, not development trajectories. It is not a formal training-budget or
learning-rate selection. Native filtering, FP32 AdamW and model/extra saving are
retained, including the upstream omission of Adam moments from checkpoints.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from . import native_visual_service as native
from .visual_task import file_sha256

ROOT = native.ROOT
TEMPLATE = ROOT / 'results/controlled-whale-seed42-phase1-plan-20260910-v1.yaml'
DIRECT_PLAN = ROOT / 'data/visual-development-screen-20260910-v1/direct-plan.json'
DIRECT_RESULT = ROOT / 'data/visual-development-screen-20260910-v1/direct/result.json'
DATA = ROOT / 'data/plotqa-evidence-native-20260910-v1/result.json'
SOURCES = ('ours/native_visual_training.py', 'ours/visual_training_bootstrap.py',
    'ours/visual_training_identity.py', 'ours/visual_training_agents.yaml',
    'ours/run_native_visual_training.sh', 'ours/experiment_randomization.py',
    'ours/native_fused_chunk.py', 'ours/native_transport_memory.py', 'ours/native_training_trace.py',
    'upstream/WHALE/domains/chess_puzzles/verl/trainer/main_textarena_disagg_rsft.py',
    'upstream/WHALE/domains/chess_puzzles/verl/trainer/ppo/ray_trainer.py',
    'upstream/WHALE/domains/chess_puzzles/verl/workers/disagg_fsdp_workers.py',
    'upstream/WHALE/domains/chess_puzzles/verl/workers/actor/dp_actor.py',
    'upstream/WHALE/domains/chess_puzzles/verl/utils/checkpoint/fsdp_checkpoint_manager.py')


def configuration(output, plan_path):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from omegaconf import OmegaConf
    raw = TEMPLATE.read_text()
    cfg = OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
    direct = json.loads(DIRECT_PLAN.read_text())
    shared = OmegaConf.create(direct['config'])
    transport = deepcopy(cfg.actor_rollout_ref.rollout.checkpoint_engine)
    cfg.actor_rollout_ref.rollout = OmegaConf.merge(cfg.actor_rollout_ref.rollout, shared.actor_rollout_ref.rollout)
    rollout = cfg.actor_rollout_ref.rollout
    rollout.checkpoint_engine = transport
    rollout.enable_sleep_mode = True
    rollout.n, rollout.temperature, rollout.top_k, rollout.top_p = 8, 1., 20, 1.
    rollout.calculate_log_probs = False
    rollout.log_prob_micro_batch_size_per_gpu = 1
    rollout.agent.agent_loop_manager_class = 'ours.native_visual_agent_observation.ObservedVisualAgentManager'
    rollout.agent.default_agent_loop = 'visual_training_agent'
    rollout.agent.agent_loop_config_path = str(ROOT / 'ours/visual_training_agents.yaml')
    rollout.trace.project_name, rollout.trace.experiment_name = 'veto-visual-engineering', output.name
    cfg.actor_rollout_ref.model.trust_remote_code = False
    cfg.data = OmegaConf.merge(cfg.data, shared.data)
    data = json.loads(DATA.read_text())['partitions']['W']
    cfg.data.train_files = [data['parquet']]
    cfg.data.val_files = [data['parquet']]
    cfg.data.train_max_samples, cfg.data.val_max_samples = 32, 32
    cfg.data.dataloader_num_workers = 0
    cfg.data.cache_dir = str(output / 'dataset-cache')
    cfg.trainer.project_name, cfg.trainer.experiment_name = 'veto-visual-engineering', output.name
    cfg.trainer.default_local_dir = str(output / 'checkpoints')
    cfg.trainer.rollout_data_dir = str(output / 'rollouts')
    cfg.trainer.validation_data_dir = str(output / 'unused-validation')
    cfg.trainer.max_actor_ckpt_to_keep = None  # No automatic checkpoint deletion.
    cfg.trainer.log_val_generations = 0
    cfg.reward.custom_reward_function = shared.reward.custom_reward_function
    env = {k: v for k, v in cfg.ray_kwargs.ray_init.runtime_env.env_vars.items()
           if not k.startswith('CHESS_') and k != 'WANDB_PROJECT'}
    env.update(PYTHONPATH=':'.join(str(ROOT / p) for p in (
        'data/visual-runtime-overlay-v1', 'ours/compat', 'upstream/WHALE/domains/chess_puzzles', '.')),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost', VLLM_NO_USAGE_STATS='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', TRITON_CACHE_DIR=str(ROOT / 'data/runtime-cache/triton'),
        VLLM_CACHE_ROOT=str(ROOT / 'data/runtime-cache/vllm'),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / 'data/runtime-cache/flashinfer'))
    env.update(VETO_NATIVE_EVALUATION_PLAN=str(plan_path), VETO_NATIVE_EVALUATION_OUTPUT=str(output),
               VERL_AGENT_LOOP_WORKER_STARTUP_BATCH_SIZE='2')
    cfg.ray_kwargs.ray_init.num_cpus = 24
    cfg.ray_kwargs.ray_init.runtime_env.env_vars = env
    cfg.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook = 'ours.visual_training_bootstrap.prepare_worker'
    return cfg


def inspect_dataset(cfg):
    from transformers import AutoProcessor, AutoTokenizer
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.utils.config import validate_config
    validate_config(config=cfg, use_reference_policy=False, use_critic=False)
    model = cfg.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    dataset = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, processor,
                                is_train=True, max_samples=cfg.data.train_max_samples)
    assert len(dataset) == 32
    ids = list(dataset.dataframe['visual_sample_id'])
    assert len(set(ids)) == 32
    loader = StatefulDataLoader(dataset=dataset, batch_size=8, num_workers=0, drop_last=True,
                               collate_fn=collate_fn, sampler=create_rl_sampler(cfg.data, dataset))
    first = next(iter(loader))
    assert len(first['visual_sample_id']) == 8
    assert all(any(part.get('type') == 'image' for message in prompt
                   if isinstance(message['content'], list) for part in message['content'])
               for prompt in first['raw_prompt'])
    return {'dataset_class': type(dataset).__name__, 'examples': len(dataset), 'all_sample_ids': ids,
            'first_native_batch_sample_ids': first['visual_sample_id'].tolist(),
            'generation_batch_size': 8, 'trajectories_per_prompt': 8,
            'model_calls': 0, 'optimizer_steps': 0,
            'limitation': 'Actual training order must still match the recorded native batch; initialization advances sampler state.'}


def prepare(path, output):
    from omegaconf import OmegaConf
    assert not path.exists() and not output.exists()
    native.check(ROOT / 'results/native-visual-service-plan-20260910-v1.json')
    reference = json.loads(DIRECT_RESULT.read_text())
    direct = json.loads(DIRECT_PLAN.read_text())
    assert reference['status'] == 'COMPLETE_DEVELOPMENT_VISUAL_CASE'
    assert reference['plan_sha256'] == file_sha256(DIRECT_PLAN)
    assert reference['marginal_accuracy'] > .5
    output.mkdir()
    cfg = configuration(output, path)
    preflight = inspect_dataset(cfg)
    plan = deepcopy(direct)
    plan.update(kind='native_visual_rsft_engineering', role='W', output=str(output),
                config=OmegaConf.to_container(cfg, resolve=True), cpu_preflight=preflight)
    for key in ('manifest', 'manifest_sha256', 'audit_data_sha256', 'decode_sha256'):
        plan.pop(key, None)
    data = json.loads(DATA.read_text())['partitions']['W']
    assert file_sha256(Path(data['parquet'])) == data['parquet_sha256']
    plan['training_data'] = data
    plan['inputs_sha256'] = {str(p): file_sha256(p) for p in (TEMPLATE, DIRECT_PLAN, DIRECT_RESULT, DATA)}
    plan['source_sha256'].update({name: file_sha256(ROOT / name) for name in SOURCES})
    plan['bounds'] = {'gpus': 2, 'cpus': 24, 'time_limit_seconds': 3600, 'native_batches': 1,
        'trajectories': 64, 'maximum_generation_calls': 192, 'maximum_generated_assistant_tokens': 64 * 1024,
        'new_native_checkpoints': 1, 'api_calls': 0}
    plan['limitations'] = ['Direct-answer engineering condition; crop tools disabled for this update only.',
        'The native seed42 subsample selects 32 W rows without model results; its sampler chooses eight prompts and eight fresh trajectories each.',
        'Original 1e-7 learning rate retained for execution verification; formal visual LR calibration remains pending.',
        'No development examples used for training; no formal VETO/WHALE result or visual h0 selection.',
        'Native model/extra checkpoint excludes Adam moments; this run does not yet prove resume or changed weights.']
    native.write_new(path, plan)
    native.write_new(output / 'cpu-preflight.json', preflight)


def run(path):
    from .visual_training_bootstrap import prepare_worker
    prepare_worker()
    import ray
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.main_textarena_disagg_rsft import run_disaggregated_rsft
    plan = json.loads(path.read_text())
    assert plan['kind'] == 'native_visual_rsft_engineering' and plan['role'] == 'W'
    for name, sha in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        assert file_sha256(ROOT / name) == sha, name
    assert file_sha256(Path(plan['training_data']['parquet'])) == plan['training_data']['parquet_sha256']
    assert plan['config'] == OmegaConf.to_container(configuration(Path(plan['output']), path), resolve=True)
    assert torch.cuda.device_count() == 2 and all('H800' in torch.cuda.get_device_name(i) for i in range(2))
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == 24
    output = Path(plan['output'])
    for name in ('requests', 'workers', 'checkpoints'):
        (output / name).mkdir()
    for name in plan['source_sha256']:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (output / 'plan.json').write_bytes(path.read_bytes())
    native.write_new(output / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path)})
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path), VETO_NATIVE_EVALUATION_OUTPUT=str(output))
    try:
        run_disaggregated_rsft(OmegaConf.create(plan['config']))
        actor = output / 'checkpoints/global_step_1/actor'
        assert (actor / 'model_world_size_1_rank_0.pt').exists()
        native.write_new(output / 'execution-result.json', {'status': 'NATIVE_VISUAL_RSFT_RETURNED',
            'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path),
            'native_checkpoint': str(actor), 'parameter_change_verified': False,
            'resume_verified': False, 'scientific_method_verified': False})
    except BaseException as error:
        native.write_new(output / 'failure.json', {'status': 'INCOMPLETE', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.plan.resolve(), args.output.resolve())
    else:
        run(args.plan.resolve())
