"""Compact E4 stages with native RSFT, pixel gradients and state-bound restoration."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from . import native_visual_training as base
from .chart_answer_protocol import PROTOCOL
from .fast_chart_protocol import ROOT, OUTPUT, storage_check
from .local_completion import checkpoint_manifest
from .visual_task import file_sha256

DATA = OUTPUT / 'native-data/result.json'
MODEL = ROOT / 'data/models/qwen3.5-4b-common-bf16-v1'
RATES = (1e-7, 1e-6, 1e-5)
SOURCES = ('ours/fast_chart_training.py', 'ours/fast_chart_training_bootstrap.py', 'ours/local_completion.py',
    'ours/fast_chart_transport.py',
    'ours/fast_chart_training_worker.py', 'ours/run_fast_chart_training.sh',
    'ours/visual_pixel_bootstrap.py', 'ours/qwen35_visual_fused.py',
    'ours/chart_answer_protocol.py', 'ours/fast_chart_protocol.py',
    'ours/native_resume_observation.py', 'ours/visual_sampler_reference.py')


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint(directory, step):
    directory = Path(directory).resolve()
    if directory.name != f'global_step_{step}' or step not in (4, 8):
        raise ValueError('Unexpected compact native checkpoint step')
    required = ('actor/model_world_size_1_rank_0.pt', 'actor/extra_state_world_size_1_rank_0.pt', 'data.pt', 'actor/fsdp_config.json')
    if any(not (directory/name).is_file() for name in required) or read(directory/'actor/fsdp_config.json')['world_size'] != 1:
        raise ValueError('Incomplete or incompatible single-actor native checkpoint')
    paths = [directory/'data.pt', *sorted((directory/'actor').rglob('*'))]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Unaccounted checkpoint symlink')
    files = [str(p.relative_to(directory)) for p in paths if p.is_file()]
    if list((directory/'actor').glob('optim*')):
        raise ValueError('Compact branches require the same native Adam-reset convention')
    return {'directory': str(directory), 'step': step,
        'artifact_sha256': {name: file_sha256(directory/name) for name in files},
        'adam_moments_saved': False}


def configuration(output, path, *, harness, seed, rate, parent=None, train_data=None):
    cfg = base.configuration(output, path)
    cfg.data.train_files = [(train_data or read(DATA)['partitions']['W'])['parquet']]
    cfg.data.val_files = cfg.data.train_files
    cfg.data.train_max_samples, cfg.data.val_max_samples = 2048, 8
    cfg.data.visual_harness_path = str(harness)
    cfg.data.visual_answer_protocol = PROTOCOL
    cfg.data.seed = seed
    cfg.trainer.total_training_steps = 8 if parent else 4
    cfg.trainer.total_epochs = 1
    cfg.trainer.save_freq = 4
    cfg.trainer.max_actor_ckpt_to_keep = None
    cfg.trainer.resume_mode = 'resume_path' if parent else 'disable'
    cfg.trainer.resume_from_path = parent['directory'] if parent else None
    cfg.trainer.del_local_ckpt_after_load = False
    cfg.trainer.project_name = 'veto-compact-chart'
    cfg.trainer.online_rsft.iterations = 0
    cfg.actor_rollout_ref.actor.optim.lr = rate
    cfg.actor_rollout_ref.actor.data_loader_seed = seed
    cfg.actor_rollout_ref.actor.fsdp_config.seed = seed
    rollout = cfg.actor_rollout_ref.rollout
    rollout.engine_kwargs.vllm.seed = seed
    rollout.engine_kwargs.vllm.worker_cls = 'ours.fast_chart_training_worker.CompactChartModelWorker'
    rollout.agent.num_workers = 2
    rollout.max_num_seqs = 8
    rollout.enable_prefix_caching = False
    rollout.enable_chunked_prefill = False
    rollout.trace.project_name = 'veto-compact-chart'
    env = cfg.ray_kwargs.ray_init.runtime_env.env_vars
    env['WHALE_TRIAL_SEED'], env['PYTHONHASHSEED'] = str(seed), str(seed)
    env['VETO_COMPACT_TRAINING_PLAN'] = str(path)
    env['VETO_VISUAL_FORWARD_AUDIT'] = str(output/'checkpoints/audit')
    env['HF_DATASETS_CACHE'] = str(OUTPUT/'runtime-cache/datasets')
    env['XDG_CACHE_HOME'] = str(OUTPUT/'runtime-cache/xdg')
    # All new task caches are charged to the compact storage root.
    for key, name in (('TRITON_CACHE_DIR','triton'), ('VLLM_CACHE_ROOT','vllm'),
                      ('FLASHINFER_WORKSPACE_BASE','flashinfer')):
        env[key] = str(OUTPUT/'runtime-cache'/name)
    cfg.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook = 'ours.fast_chart_training_bootstrap.prepare_worker'
    return cfg


def dataset_loader(cfg):
    from transformers import AutoProcessor, AutoTokenizer
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.utils.config import validate_config
    validate_config(config=cfg, use_reference_policy=False, use_critic=False)
    model = cfg.actor_rollout_ref.model.path
    dataset = create_rl_dataset(cfg.data.train_files, cfg.data,
        AutoTokenizer.from_pretrained(model, local_files_only=True),
        AutoProcessor.from_pretrained(model, local_files_only=True), is_train=True, max_samples=2048)
    ids = list(dataset.dataframe['visual_sample_id'])
    if len(ids) != 2048 or len(set(ids)) != 2048:
        raise ValueError('Harness filtering changed the common W population')
    loader = StatefulDataLoader(dataset, batch_size=8, num_workers=0, drop_last=True,
        collate_fn=collate_fn, sampler=create_rl_sampler(cfg.data, dataset))
    return dataset, loader


def inspect_dataset(cfg, parent=None):
    import torch
    from omegaconf import OmegaConf
    from .native_resume_observation import state_digest
    cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    cfg.data.cache_dir=str(OUTPUT/'preflight-cache')
    dataset, loader = dataset_loader(cfg)
    saved = None
    if parent:
        saved = torch.load(Path(parent['directory'])/'data.pt', map_location='cpu', weights_only=False)
        loader.load_state_dict(saved)
        if state_digest(loader.next_iter_state) != state_digest(saved):
            raise ValueError('Native sampler restoration differs from the actual parent checkpoint')
    batches = []
    with torch.random.fork_rng(devices=[]):
        iterator = iter(loader)
        for _ in range(4):
            batch = next(iterator)
            if not all(any(part.get('type') == 'image' for message in prompt
                    if isinstance(message['content'], list) for part in message['content'])
                    for prompt in batch['raw_prompt']):
                raise ValueError('Training prompt lost its chart image')
            batches.append(batch['visual_sample_id'].tolist())
    if len(set(x for batch in batches for x in batch)) != 32:
        raise ValueError('Unexpected repeated chart in the four native batches')
    return {'all_sample_ids': list(dataset.dataframe['visual_sample_id']),
        'next_four_batches': batches, 'examples': len(dataset),
        'saved_state_digest': state_digest(saved) if saved is not None else None,
        'saved_state_sha256': parent['artifact_sha256']['data.pt'] if parent else None,
        'model_calls': 0, 'optimizer_steps': 0}


def prepare(path, output, *, calibration, seed, rate, parent_plan=None, selection=None, augmentation=None):
    from omegaconf import OmegaConf
    path, output, calibration = map(lambda p: Path(p).resolve(), (path, output, calibration))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require fresh compact training paths')
    if seed not in (42,43,44) or rate not in RATES:
        raise ValueError('Unregistered training seed or learning rate')
    shared = read(calibration)
    if shared['status'] != 'COMPLETE_SHARED_H0_CALIBRATION':
        raise ValueError('Common strong initial harness calibration is incomplete')
    harness, parent, incoming = Path(shared['harness']), None, None
    inputs = {str(calibration): file_sha256(calibration), str(DATA): file_sha256(DATA)}
    if parent_plan:
        parent_plan = Path(parent_plan).resolve()
        incoming = read(parent_plan)
        if incoming['kind'] != 'compact_chart_rsft_stage' or incoming['stage'] != 1 or incoming['seed'] != seed or incoming['rate'] != rate:
            raise ValueError('Incompatible common first-stage plan')
        result_path = Path(incoming['output'])/'execution-result.json'
        result = read(result_path)
        if result['status'] != 'COMPLETE_COMPACT_NATIVE_STAGE' or result['plan_sha256'] != file_sha256(parent_plan):
            raise ValueError('First-stage execution not verified')
        parent = checkpoint(Path(incoming['output'])/'checkpoints/global_step_4', 4)
        if selection is None:
            raise ValueError('A second stage requires a recorded selection handoff')
        selection = Path(selection).resolve()
        choice = read(selection)
        if choice.get('status') != 'COMPLETE_COMPACT_SELECTION' or choice['parent_plan_sha256'] != file_sha256(parent_plan):
            raise ValueError('Selection does not belong to these first-stage weights')
        harness = Path(choice['selected_harness'])
        if file_sha256(harness) != choice['harness_sha256']:
            raise ValueError('Selected harness code changed')
        inputs.update({str(parent_plan): file_sha256(parent_plan), str(result_path): file_sha256(result_path),
                       str(selection): file_sha256(selection)})
    elif selection is not None:
        raise ValueError('Selection cannot precede the first weight stage')
    training_data=read(DATA)['partitions']['W']
    augmentation_record=None
    if augmentation:
        augmentation=Path(augmentation).resolve();augmentation_record=read(augmentation)
        if not parent or augmentation_record['kind']!='compact_counterfactual_augmentation' or augmentation_record['parent_plan_sha256']!=file_sha256(parent_plan):
            raise ValueError('Augmentation belongs to another continuation')
        if augmentation_record['selection_sha256']!=file_sha256(selection) or choice['condition']!='whale':
            raise ValueError('Augmentation must use the exact WHALE-selected harness')
        if (augmentation_record['ratio'],augmentation_record['pairs'],augmentation_record['total_trajectories'])!=(.5,8,256):
            raise ValueError('Augmentation budget differs from the frozen control')
        training_data={'manifest':str(augmentation),'manifest_sha256':file_sha256(augmentation),
            'parquet':augmentation_record['parquet'],'parquet_sha256':augmentation_record['parquet_sha256']}
        inputs[str(augmentation)]=file_sha256(augmentation)
    cfg = configuration(output, path, harness=harness, seed=seed, rate=rate, parent=parent,train_data=training_data)
    if Path(cfg.actor_rollout_ref.model.path).resolve() != MODEL:
        raise ValueError('Native initialization is not the common4B model')
    preflight = inspect_dataset(cfg, parent)
    if incoming:
        # Independent reference uses the actual incoming plan/h0 and saved sampler.
        reference = inspect_dataset(OmegaConf.create(incoming['config']), parent)
        if augmentation_record:
            if reference!=augmentation_record['actual_sampler_reference']:
                raise ValueError('Augmentation reference does not match the actual parent sampler')
            mapping=augmentation_record['mapping']
            expected=deepcopy(reference)
            expected['all_sample_ids']=[mapping.get(ident,ident) for ident in reference['all_sample_ids']]
            expected['next_four_batches']=[[mapping.get(ident,ident) for ident in batch] for batch in reference['next_four_batches']]
            reference=expected
        if preflight != reference:
            raise ValueError('Candidate harness changed the incoming native sampler population/order')
    original = read(ROOT/'results/visual-pixel-rsft-plan-20260910-v3.json')
    sources = set(original['source_sha256']) | set(base.SOURCES) | set(SOURCES)
    sources.add(str(harness.relative_to(ROOT)))
    model = checkpoint_manifest(MODEL)
    plan = {'kind': 'compact_chart_rsft_stage', 'role': 'W', 'stage': 2 if parent else 1,
        'output': str(output), 'seed': seed, 'rate': rate,
        'harness': str(harness), 'harness_sha256': file_sha256(harness),
        'h0_calibration': str(calibration), 'parent_plan': str(parent_plan) if parent else None,
        'selection': str(selection) if parent else None, 'resume_checkpoint': parent,
        'selection_protocol':choice.get('selection_protocol','gate_v1') if parent else None,
        'training_data': training_data, 'cpu_preflight': preflight,'augmentation':str(augmentation) if augmentation else None,
        'config': OmegaConf.to_container(cfg, resolve=True), 'model': model,
        'worker_coordinates': base.native.embedding_coordinates(MODEL),
        'inputs_sha256': inputs, 'source_sha256': {name: file_sha256(ROOT/name) for name in sorted(sources)},
        'bounds': {'gpus': 2, 'cpus': 24, 'time_limit_seconds': 3600, 'native_batches': 4,
            'trajectories': 256, 'maximum_generation_calls': 768,
            'maximum_generated_assistant_tokens': 256*1024, 'new_native_checkpoints': 1, 'api_calls': 0},
        'limitations': ['One limited-budget weight stage; no performance claim from execution alone.',
            'Native model/extra restoration omits Adam moments in every branch.',
            'Checkpoint and complete receiver values are checked separately from task accuracy.']}
    storage_check(35)
    base.native.write_new(path, plan)
    return plan


def check(path):
    plan = read(path)
    if plan['kind'] != 'compact_chart_rsft_stage' or Path(plan['output']).exists():
        raise ValueError('Wrong or already executed training plan')
    for name, digest in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f'Frozen training input changed: {name}')
    if checkpoint_manifest(MODEL) != plan['model']:
        raise ValueError('Initial model identity changed')
    for key in ('manifest', 'parquet'):
        if file_sha256(Path(plan['training_data'][key])) != plan['training_data'][key+'_sha256']:
            raise ValueError('Training data identity changed')
    if plan['resume_checkpoint'] and checkpoint(Path(plan['resume_checkpoint']['directory']),4) != plan['resume_checkpoint']:
        raise ValueError('Parent checkpoint identity changed')
    storage_check(35)
    return plan


def audit_output(plan, path):
    out = Path(plan['output'])
    start = 1 if plan['stage'] == 1 else 5
    batches = []
    for step, ids in zip(range(start,start+4), plan['cpu_preflight']['next_four_batches'], strict=True):
        r = read(out/f'checkpoints/audit/rollout-step{step}.json')
        if r['examples'] != 64 or r['sample_ids'] != [sample for sample in ids for _ in range(8)]:
            raise ValueError('Actual native generation differs from the frozen sampler order/budget')
        if any(score not in (0,1) for score in r['scores']):
            raise ValueError('Invalid native visual reward')
        accepted = sum(score == 1 for score in r['scores'])
        if accepted:
            update = read(out/f'checkpoints/audit/update-step{step}.json')
            if update['accepted_receipt']['examples'] != accepted:
                raise ValueError('Native success filtering lost or added trajectories')
        batches.append({'step': step, 'successful_trajectories': accepted,
                        'rollout_receipt_sha256': file_sha256(out/f'checkpoints/audit/rollout-step{step}.json')})
    records = [json.loads(line) for p in (out/'checkpoints/audit').glob('visual-backbone-*.jsonl') for line in p.read_text().splitlines()]
    inputs = {r['event'] for r in records if r['kind'] == 'visual_backbone_input'}
    gradients = [r for r in records if r['kind'] == 'visual_feature_gradient']
    if sum(b['successful_trajectories'] for b in batches) and (not gradients or
            not any(r['nonzero_elements'] > 0 for r in gradients) or
            not all(r['event'] in inputs and r['finite'] for r in gradients)):
        raise ValueError('No verified image-loss gradient for successful trajectories')
    for step in range(start-1,start+4):
        if read(out/f'state/transport-step{step}.json')['plan_sha256'] != file_sha256(path):
            raise ValueError('Missing actual native transport observation')
    return {'batches': batches, 'input_forwards': len(inputs), 'gradient_events': len(gradients),
        'nonzero_gradient_events': sum(r['nonzero_elements']>0 for r in gradients),
        'checkpoint': checkpoint(out/f'checkpoints/global_step_{start+3}',start+3)}


def run(path):
    plan = check(path)
    if os.environ.get('WHALE_TRIAL_SEED') != str(plan['seed']) or os.environ.get('PYTHONHASHSEED') != str(plan['seed']):
        raise ValueError('Seed must be set before launching this interpreter')
    from .fast_chart_training_bootstrap import prepare_worker
    from omegaconf import OmegaConf
    import torch
    if torch.cuda.device_count() != 2 or any('H800' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise ValueError('Require the frozen two-H800 native allocation')
    out = Path(plan['output']); out.mkdir(parents=True)
    for name in ('requests','workers','checkpoints','state'):
        (out/name).mkdir()
    (out/'plan.json').write_bytes(path.read_bytes())
    for name in plan['source_sha256']:
        target = out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes((ROOT/name).read_bytes())
    os.environ.update(VETO_COMPACT_TRAINING_PLAN=str(path), VETO_NATIVE_EVALUATION_PLAN=str(path),
        VETO_NATIVE_EVALUATION_OUTPUT=str(out), VETO_VISUAL_FORWARD_AUDIT=str(out/'checkpoints/audit'))
    prepare_worker()
    import ray
    from verl.trainer.main_textarena_disagg_rsft import run_disaggregated_rsft
    started = time.monotonic()
    try:
        run_disaggregated_rsft(OmegaConf.create(plan['config']))
        report = audit_output(plan,path)
        report.update(status='COMPLETE_COMPACT_NATIVE_STAGE', plan_sha256=file_sha256(path),
            job_id=os.environ['SLURM_JOB_ID'], timing_seconds=time.monotonic()-started,
            parameter_change_verified=False, scientific_method_verified=False)
        base.native.write_new(out/'execution-result.json',report)
    except BaseException as exc:
        base.native.write_new(out/'failure.json',{'status':'INCOMPLETE','error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','check','run'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--seed',type=int,choices=(42,43,44));p.add_argument('--rate',type=float,choices=RATES)
    p.add_argument('--parent-plan',type=Path);p.add_argument('--selection',type=Path)
    p.add_argument('--augmentation',type=Path)
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.plan,a.output,calibration=a.calibration,seed=a.seed,rate=a.rate,parent_plan=a.parent_plan,selection=a.selection,augmentation=a.augmentation)
    elif a.action=='check': check(a.plan)
    else: run(a.plan.resolve())
