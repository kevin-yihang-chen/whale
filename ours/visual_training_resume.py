"""Resume visual E4 only from a complete, identity-bound E2-E3 decision.

Keep native RSFT sampling, success filtering, loss and model/extra checkpoint
semantics. The released checkpoint does not contain Adam moments. Initial model
assets construct both graphs; the original native loader restores theta1 before
the first weight synchronization and the next eight W questions.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

from .acceptance import EvidenceConstrainedAcceptance
from .adapter import WHALEAcceptanceAdapter
from .evidence import EvaluationIdentity, fingerprint
from .local_completion import checkpoint_manifest
from .native_phase_resume import check_configuration, check_layout
from .native_visual_service import ROOT, write_new
from .visual_resume_bootstrap import observe_visual_loader
from .visual_search_bridge import boundary, certify, load as load_search
from .visual_task import file_sha256

SOURCES = ('ours/visual_training_resume.py', 'ours/visual_resume_bootstrap.py',
    'ours/visual_resume_model_worker.py', 'ours/run_visual_training_resume.sh',
    'ours/native_resume_observation.py', 'ours/native_phase_resume.py', 'ours/visual_resume_data_probe.py',
    'ours/visual_sampler_reference.py',
    'upstream/WHALE/domains/chess_puzzles/verl/checkpoint_engine/base.py',
    'upstream/WHALE/domains/chess_puzzles/verl/workers/fsdp_workers.py',
    'upstream/WHALE/domains/chess_puzzles/verl/workers/rollout/vllm_rollout/vllm_async_server.py')


def read(path):
    return json.loads(Path(path).read_text())


def certify_visual_training_lineage(first, first_path):
    if first.get('execution_variant') != 'qwen35_visual_pixel_restart':
        raise ValueError('Visual resume requires a corrected image-conditioned training lineage')
    directory = Path(first['output'])
    receipt_path = directory / 'visual-backbone-audit.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'AUDITED_NATIVE_IMAGE_CONSUMPTION_AND_GRADIENTS'
    assert receipt['plan_sha256'] == file_sha256(first_path)
    assert receipt['actual_images_match_native_success_subset'] and receipt['nonzero_gradient_events'] > 0
    assert receipt['paired_gradient_events'] == receipt['actual_backbone_forwards'] == receipt['accepted_examples']
    evidence = {str(receipt_path): file_sha256(receipt_path), **receipt['evidence_sha256']}
    observed = read(directory / 'visual-forward-result.json')
    assert observed['plan_sha256'] == receipt['plan_sha256'] and observed['job_id'] == receipt['job_id']
    evidence.update(observed['artifacts_sha256'])
    for name, digest in evidence.items():
        assert file_sha256(Path(name)) == digest, name
    return evidence


def certify_handoff(search_root, first_path, followup_path):
    from meta_harness import meta_harness_chess_puzzle as native
    selection_path = search_root / 'accepted-for-training.json'
    selection = read(selection_path)  # A bare accepted_harness.txt cannot pass.
    search = load_search(search_root)
    assert selection['status'] == 'COMPLETE_SEARCH_AND_TRAINING_ACCEPTANCE'
    assert selection['plan_sha256'] == file_sha256(search_root / 'plan.json')
    base = certify(search['baseline']['plan'], search['baseline']['allocation'])
    candidate_ref = read(search_root / 'candidate-evaluation.json')
    candidate = certify(candidate_ref['plan'], candidate_ref['allocation'])
    assert file_sha256(Path(candidate['plan']['output']) / 'result.json') == candidate_ref['result_sha256']
    archive = [base, candidate]
    assert [x['candidate'].name for x in archive] == ['h0', 'h1']
    assert all(x['result']['identity'] == search['identity'] for x in archive)
    assert selection['evaluations'] == {x['candidate'].name: {'candidate': asdict(x['candidate']),
        'audit': json.loads(json.dumps(asdict(x['audit']))), 'repeatability_passed': x['result']['repeatability_passed']}
        for x in archive}
    identity = EvaluationIdentity(**search['identity'])
    comparison = read(search_root / 'search/logs/iteration_001/comparison.json')
    restored = boundary(WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance(search['mode'], epsilon=search['epsilon'])),
        native, search_root / 'search', archive, identity, stage='resume', frontier=comparison['frontier'],
        rows=comparison['summary'], valid_names=['h1'], resume=selection['receipt'])
    assert fingerprint(restored) == fingerprint(read(search_root / 'acceptance-resume.json'))
    assert restored['accepted_harness'] == selection['accepted_harness'] == selection['stop']['training_harness']
    selected = search_root / 'search/harnesses' / selection['accepted_harness'] / 'harness.py'
    assert str(selected) == selection['accepted_harness_path']
    assert file_sha256(selected) == selection['accepted_harness_sha256']
    first, followup = read(first_path), read(followup_path)
    assert first['kind'] == 'native_visual_rsft_engineering' and first['role'] == 'W'
    visual_evidence = certify_visual_training_lineage(first, first_path)
    assert checkpoint_manifest(Path(first['model']['path'])) == first['model']
    assert Path(followup['training_plan']) == first_path and followup['base'] == first['model']
    assert followup['inputs_sha256'][str(first_path)] == file_sha256(first_path)
    directory = check_layout(Path(first['output']) / 'checkpoints/global_step_1')
    assert Path(followup['native']) == directory / 'actor'
    transition_path = Path(followup['output']) / 'transition.json'
    transition = read(transition_path)
    assert transition['status'] == 'PASS' and transition['resume_artifacts_unchanged'] and transition['export_exact_native_bf16_cast']
    assert transition['plan_sha256'] == file_sha256(followup_path)
    assert transition['exported'] == base['plan']['model'] == candidate['plan']['model']
    assert transition['native_checkpoint_sha256'] == file_sha256(directory / 'actor/model_world_size_1_rank_0.pt')
    artifacts = {str(Path(p).relative_to(directory)): sha for p, sha in followup['resume_sha256'].items()}
    assert all(file_sha256(directory / name) == sha for name, sha in artifacts.items())
    coordinates = base['plan']['worker_coordinates']
    assert len(coordinates) == 8 and all(p['base'] != p['expected'] for p in coordinates)
    return first, {'selection_path': str(selection_path), 'selection_sha256': file_sha256(selection_path),
        'search_root': str(search_root), 'search_plan_sha256': file_sha256(search_root / 'plan.json'),
        'first_plan': str(first_path), 'first_plan_sha256': file_sha256(first_path),
        'followup_plan': str(followup_path), 'followup_plan_sha256': file_sha256(followup_path),
        'transition_path': str(transition_path), 'transition_sha256': file_sha256(transition_path),
        'search_receipt_sha256': {name: file_sha256(search_root / name) for name in (
            'plan.json', 'candidate-evaluation.json', 'accepted-for-training.json',
            'acceptance-resume.json', 'search/logs/iteration_001/comparison.json')},
        'accepted_harness': str(selected), 'harness_sha256': file_sha256(selected),
        'visual_training_evidence_sha256': visual_evidence,
        'identity': search['identity'], 'resume_coordinates': coordinates,
        'resume_checkpoint': {'directory': str(directory), 'step': 1, 'artifact_sha256': artifacts}}


def configuration(first, output, plan_path, harness):
    config = deepcopy(first['config'])
    data, trainer = config['data'], config['trainer']
    rollout = config['actor_rollout_ref']['rollout']
    data.update(visual_harness_path=str(harness), cache_dir=str(output / 'dataset-cache'))
    trainer.update(total_training_steps=2, resume_mode='resume_path',
        resume_from_path=str(Path(first['output']) / 'checkpoints/global_step_1'),
        experiment_name=output.name, default_local_dir=str(output / 'checkpoints'),
        rollout_data_dir=str(output / 'rollouts'), validation_data_dir=str(output / 'unused-validation'),
        del_local_ckpt_after_load=False, max_actor_ckpt_to_keep=None, max_critic_ckpt_to_keep=None)
    rollout['trace']['experiment_name'] = output.name
    rollout['engine_kwargs']['vllm']['worker_cls'] = 'ours.visual_resume_model_worker.VisualResumeModelWorker'
    ray = config['ray_kwargs']['ray_init']
    ray['runtime_env']['worker_process_setup_hook'] = 'ours.visual_resume_bootstrap.prepare_worker'
    ray['runtime_env']['env_vars'].update(VETO_NATIVE_EVALUATION_PLAN=str(plan_path),
        VETO_NATIVE_EVALUATION_OUTPUT=str(output), VETO_VISUAL_RESUME_PLAN=str(plan_path),
        VETO_VISUAL_FORWARD_AUDIT=str(output / 'checkpoints/audit'))
    return config


def data_preflight(config, first):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.dataset.rl_dataset import collate_fn
    cfg = OmegaConf.create(config)
    for key in ('manifest', 'parquet'):
        assert file_sha256(Path(first['training_data'][key])) == first['training_data'][key + '_sha256']
    directory = check_layout(Path(first['output']) / 'checkpoints/global_step_1')
    check_configuration(cfg, directory)
    model = cfg.actor_rollout_ref.model.path
    dataset = create_rl_dataset(cfg.data.train_files, cfg.data,
        AutoTokenizer.from_pretrained(model, local_files_only=True), AutoProcessor.from_pretrained(model, local_files_only=True),
        is_train=True, max_samples=cfg.data.train_max_samples)
    ids = list(dataset.dataframe['visual_sample_id'])
    assert ids == first['cpu_preflight']['all_sample_ids']
    calls = []
    class ActorBoundary:
        def load_checkpoint(self, path, *, del_local_after_load):
            calls.append({'path': path, 'del_local_after_load': del_local_after_load})
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config, trainer.global_steps, trainer.use_critic = cfg, 0, False
    trainer.actor_rollout_wg = ActorBoundary()
    trainer._create_dataloader(dataset, dataset, collate_fn, create_rl_sampler(cfg.data, dataset))
    trainer._load_checkpoint()
    pending = observe_visual_loader(trainer, directory, ids)
    batch = next(iter(trainer.train_dataloader))
    next_ids = batch['visual_sample_id'].tolist()
    from .visual_sampler_reference import reference as sampler_reference
    reference = sampler_reference(first)
    assert reference['first_plan_sha256'] == file_sha256(Path(first['output']) / 'plan.json')
    assert reference['saved_state_sha256'] == pending['saved_state_sha256']
    assert reference['saved_state_digest'] == pending['pending_state_digest']
    assert next_ids == reference['next_eight_ids']
    assert calls == [{'path': str(directory / 'actor'), 'del_local_after_load': False}]
    assert all(any(p.get('type') == 'image' for message in prompt if isinstance(message['content'], list)
                   for p in message['content']) for prompt in batch['raw_prompt'])
    assert all(digest == file_sha256(Path(cfg.data.visual_harness_path)) for digest in batch['visual_harness_sha256'])
    return {'status': 'PASS_SELECTED_VISUAL_HARNESS_NATIVE_DATA_RESUME_CPU', 'all_sample_ids': ids,
        'next_eight_ids': next_ids, 'pending_loader': pending, 'actor_load_rpc': calls,
        'actor_rpc_executed': False, 'model_calls': 0, 'optimizer_steps': 0,
        'harness_sha256': file_sha256(Path(cfg.data.visual_harness_path)),
        'sampler_reference': reference}


def prepare(path, output, search_root, first_path, followup_path):
    assert not path.exists() and not output.exists()
    first, handoff = certify_handoff(search_root, first_path, followup_path)
    plan = deepcopy(first)
    config = configuration(first, output, path, Path(handoff['accepted_harness']))
    preflight = data_preflight(config, first)
    plan.update(kind='native_visual_rsft_resume', output=str(output), handoff=handoff, config=config,
        cpu_preflight=preflight, accepted_harness=handoff['accepted_harness'], harness_sha256=handoff['harness_sha256'],
        resume_checkpoint=handoff['resume_checkpoint'], resume_coordinates=handoff['resume_coordinates'])
    plan['source_sha256'].update({p: file_sha256(ROOT / p) for p in SOURCES})
    plan['source_sha256'][handoff['accepted_harness']] = handoff['harness_sha256']
    plan['inputs_sha256'].update({str(p): file_sha256(p) for p in (
        first_path, followup_path, Path(handoff['selection_path']),
        ROOT / 'results/visual-native-data-resume-20260910-v1.json')})
    plan['bounds'] = {'gpus': 2, 'cpus': 24, 'time_limit_seconds': 3600, 'native_batches': 1,
        'trajectories': 64, 'maximum_generation_calls': 192, 'maximum_generated_assistant_tokens': 65536,
        'new_native_checkpoints': 1, 'api_calls': 0}
    plan['storage'] = {'additional_working_space_gib': 40, 'minimum_free_margin_gib': 40}
    assert shutil.disk_usage(ROOT).free >= sum(plan['storage'].values()) * 1024**3
    plan['limitations'] = ['One engineering resumed batch using the completed visual search selection; not a formal method result.',
        'Released model/extra resume restores actor/scheduler/RNG/data; Adam moments start empty in every controlled branch.',
        'The original shared W order and success filtering are retained; no evaluation trajectories are trained on.',
        'Receiver checks cover eight coordinates; full parameter changes and independent performance require subsequent evaluation.']
    write_new(path, plan)


def check(path):
    plan = read(path)
    assert plan['kind'] == 'native_visual_rsft_resume' and plan['role'] == 'W'
    handoff = plan['handoff']
    first, verified = certify_handoff(Path(handoff['search_root']), Path(handoff['first_plan']), Path(handoff['followup_plan']))
    assert handoff == verified
    assert plan['config'] == configuration(first, Path(plan['output']), path, Path(handoff['accepted_harness']))
    for name, sha in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        assert file_sha256(ROOT / name) == sha, name
    assert plan['cpu_preflight'] == data_preflight(plan['config'], first)
    assert not Path(plan['output']).exists()
    assert shutil.disk_usage(ROOT).free >= sum(plan['storage'].values()) * 1024**3
    return plan


def run(path):
    plan = check(path)
    output = Path(plan['output'])
    output.mkdir()
    for name in ('requests', 'workers', 'checkpoints', 'resume'):
        (output / name).mkdir()
    for name in plan['source_sha256']:
        source = ROOT / name
        target = output / 'source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(path, output / 'plan.json')
    write_new(output / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path)})
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path), VETO_NATIVE_EVALUATION_OUTPUT=str(output), VETO_VISUAL_RESUME_PLAN=str(path))
    from .visual_resume_bootstrap import prepare_worker
    prepare_worker()
    import ray
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.main_textarena_disagg_rsft import run_disaggregated_rsft
    assert torch.cuda.device_count() == 2 and all('H800' in torch.cuda.get_device_name(i) for i in range(2))
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == 24
    try:
        run_disaggregated_rsft(OmegaConf.create(plan['config']))
        actor = output / 'checkpoints/global_step_2/actor'
        assert (actor / 'model_world_size_1_rank_0.pt').exists()
        rollout = read(output / 'checkpoints/audit/rollout-step2.json')
        assert rollout['examples'] == 64 and list(dict.fromkeys(rollout['sample_ids'])) == plan['cpu_preflight']['next_eight_ids']
        assert len(list((output / 'checkpoints').glob('global_step_*'))) == 1
        for name in ('actor', 'loader', 'transport-step1', 'transport-step2'):
            assert read(output / f'resume/{name}.json')['plan_sha256'] == file_sha256(path)
        write_new(output / 'execution-result.json', {'status': 'NATIVE_VISUAL_RSFT_RESUME_RETURNED',
            'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path), 'native_checkpoint': str(actor),
            'parameter_change_verified': False, 'full_trajectory_audit_pending': True, 'scientific_method_verified': False})
    except BaseException as error:
        write_new(output / 'failure.json', {'status': 'INCOMPLETE', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run'), required=True)
    for name in ('plan', 'output', 'search-root', 'first-plan', 'followup-plan'):
        parser.add_argument('--' + name, type=Path, required=name == 'plan')
    args = parser.parse_args()
    path = args.plan.resolve()
    if args.phase == 'prepare':
        prepare(path, args.output.resolve(), args.search_root.resolve(), args.first_plan.resolve(), args.followup_plan.resolve())
    elif args.phase == 'check':
        check(path)
    else:
        run(path)
