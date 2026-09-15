"""Measure E4 parameter changes and E1 development answers after native RSFT.

Reuse the original FSDP merger, full active-parameter comparison, visual loop,
and pair evaluator. This checkpoint follow-up performs no search or training;
unchanged parameters or predictions remain valid measured outcomes.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import native_visual_service as native
from .canonical_native_export import validate_inference_config
from .controlled_checkpoint_export import restore_generation_asset
from .evidence import EvaluationIdentity, fingerprint
from .local_completion import checkpoint_manifest
from .probe_updated_vllm import changed_embedding_coordinates, inference_contract
from .research_budget import terminal_allocation
from .verify_alternation_transition import compare_active_parameters
from .visual_native_evaluation import evaluate_pairs, pair_inputs, write_pair_parquet
from .visual_task import binary_answer_verifier, file_sha256

ROOT = native.ROOT
EXTRA = ('ours/native_visual_followup.py', 'ours/run_native_visual_followup.sh',
    'ours/canonical_native_export.py', 'ours/controlled_checkpoint_export.py',
    'ours/verify_alternation_transition.py', 'ours/verify_native_transition.py',
    'ours/probe_updated_vllm.py', 'ours/local_completion.py', 'ours/research_budget.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/base_model_merger.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/fsdp_model_merger.py')


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(mapping):
    for name, digest in mapping.items():
        assert file_sha256(ROOT / name) == digest, f'Changed frozen file: {name}'


def prepare(path, output, training_path, baseline_path, batch_audit_path):
    assert not path.exists() and not output.exists()
    training, baseline, audit = map(read, (training_path, baseline_path, batch_audit_path))
    root = Path(training['output'])
    execution = read(root / 'execution-result.json')
    terminal = terminal_allocation((root / 'slurm-terminal.txt').read_text(), execution['job_id'])
    assert terminal and terminal['state'] == 'COMPLETED'
    assert execution['status'] == 'NATIVE_VISUAL_RSFT_RETURNED'
    assert execution['plan_sha256'] == file_sha256(training_path) == audit['plan_sha256']
    assert audit['status'] == 'AUDITED_NATIVE_VISUAL_RSFT_BATCH' and audit['job_id'] == execution['job_id']
    assert audit['input_pixels_and_tokens_replayed'] and audit['native_success_subset_exact']
    assert file_sha256(ROOT / 'ours/audit_native_visual_training.py') == audit['audit_source_sha256']
    assert baseline['mode'] == 'direct' and baseline['role'] == 'V'
    assert baseline['model'] == training['model'] == checkpoint_manifest(Path(training['model']['path']))
    before = read(Path(baseline['output']) / 'result.json')
    assert before['status'] == 'COMPLETE_DEVELOPMENT_VISUAL_CASE'
    assert before['plan_sha256'] == file_sha256(baseline_path)
    manifest, pairs, _ = pair_inputs(Path(baseline['manifest']))
    assert manifest['role'] == 'V' and len(pairs) == 64
    initial_path = ROOT / 'results/canonical-initialization-20260910.json'
    initial = read(initial_path)
    assert initial['exported'] == training['model']
    native_dir = Path(execution['native_checkpoint'])
    resume = {str(p.resolve()): file_sha256(p) for p in native_dir.parent.rglob('*') if p.is_file()}
    assert any(p.endswith('/data.pt') for p in resume)
    assert any(p.endswith('/extra_state_world_size_1_rank_0.pt') for p in resume)
    assert not any('optim_world_size' in p for p in resume)
    inputs = (training_path, baseline_path, batch_audit_path, initial_path,
        root / 'execution-result.json', root / 'slurm-terminal.txt', root / 'allocation-result.json',
        root / 'checkpoints/audit/update-step1.json', Path(baseline['output']) / 'result.json',
        Path(baseline['output']) / 'evaluation/result.json', Path(baseline['manifest']))
    sources = {**training['source_sha256'], **baseline['source_sha256']}
    sources.update({n: file_sha256(ROOT / n) for n in EXTRA})
    verify_files(sources)
    plan = {'kind': 'native_visual_checkpoint_followup', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_plan': str(training_path), 'training_job_id': execution['job_id'],
        'baseline_plan': str(baseline_path), 'baseline_output': baseline['output'],
        'base': training['model'], 'native': str(native_dir), 'resume_sha256': resume,
        'output': str(output), 'target': str(output / 'hf-step-1-canonical'),
        'source_sha256': sources, 'inputs_sha256': {str(p): file_sha256(p) for p in inputs},
        'aliases': initial['audit']['verified_aliases'],
        'active_tensors': initial['audit']['comparison']['tensor_count'],
        'reviewed_native_config_differences': validate_inference_config(Path(training['model']['path']), native_dir / 'huggingface'),
        'resources': {'gpus': 1, 'cpus': 12, 'ram_gib': 96, 'time_limit_seconds': 3600},
        'storage': {'additional_working_space_gib': 32, 'minimum_free_margin_gib': 40,
                    'new_model_exports': 1, 'delete_existing_files': False},
        'bounds': {'development_pairs': 64, 'maximum_generation_calls': 384,
                   'maximum_generated_assistant_tokens': 131072, 'new_optimizer_steps': 0, 'api_calls': 0},
        'limitations': ['One engineering W batch and one development subset, not a formal condition or seed comparison.',
            'No VETO acceptance, harness search, test-set access, or native optimizer resume in this follow-up.',
            'All outcomes, including unchanged weights and predictions, are retained.',
            'Native model/extra/data.pt remain intact; Adam moments were not saved by the original protocol.']}
    native.write_new(path, plan)


def check(path):
    plan = read(path)
    assert plan['kind'] == 'native_visual_checkpoint_followup'
    verify_files(plan['source_sha256'])
    verify_files(plan['inputs_sha256'])
    verify_files(plan['resume_sha256'])
    assert checkpoint_manifest(Path(plan['base']['path'])) == plan['base']
    assert not Path(plan['output']).exists()
    needed = sum(plan['storage'][k] for k in ('additional_working_space_gib', 'minimum_free_margin_gib')) * 1024**3
    assert shutil.disk_usage(ROOT).free >= needed, 'Insufficient existing space; expansion must pause'
    return plan


def make_case(plan, transition):
    case = deepcopy(read(plan['baseline_plan']))
    output = Path(plan['output']) / 'direct'
    case.update(kind='native_visual_post_training_case', output=str(output), model=transition['exported'],
                limitations=plan['limitations'])
    case['config']['actor_rollout_ref']['model']['path'] = plan['target']
    case['config']['data']['cache_dir'] = str(output / 'dataset-cache')
    case['config']['actor_rollout_ref']['rollout']['trace']['experiment_name'] = output.parent.name
    case['config']['trainer']['experiment_name'] = output.parent.name
    embedding = next(r for r in transition['exported_bf16']['tensors']
                     if r['name'] == 'model.language_model.embed_tokens.weight')
    if embedding['changed_elements'] >= 8:
        changed = changed_embedding_coordinates(Path(plan['base']['path']), Path(plan['target']))
        case['worker_coordinates'] = [{**r, 'expected': r['updated']} for r in changed]
        case['coordinates_distinguish_initial_model'] = True
    else:
        case['worker_coordinates'] = native.embedding_coordinates(Path(plan['target']))
        case['coordinates_distinguish_initial_model'] = False
    case['source_sha256'] = plan['source_sha256']
    case['decode_sha256'] = fingerprint({'config': case['config'], 'mode': case['mode'],
                                         'model_assets': case['model']['assets']})
    return case


async def evaluate(case, case_path):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import ray
    import torch
    from omegaconf import OmegaConf
    from .native_visual_agent_observation import ObservedVisualAgentManager
    assert torch.cuda.device_count() == 1 and 'H800' in torch.cuda.get_device_name(0)
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == 12
    output = Path(case['output'])
    output.mkdir()
    for name in ('requests', 'workers'):
        (output / name).mkdir()
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(case_path), VETO_NATIVE_EVALUATION_OUTPUT=str(output))
    native.write_new(output / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(case_path)})
    parquet = output / 'pairs.parquet'
    native.write_new(output / 'materialization.json', write_pair_parquet(Path(case['manifest']), parquet))
    try:
        dataset = native.dataset_for(case, parquet)
        env = {k: v for k, v in os.environ.items() if k.startswith(('VETO_', 'HF_', 'TRANSFORMERS_', 'VERL_', 'VLLM_')) or
               k in ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'NO_PROXY', 'no_proxy')}
        ray.init(num_cpus=12, num_gpus=1, include_dashboard=False, object_store_memory=2*1024**3,
                 runtime_env={'env_vars': env, 'worker_process_setup_hook': 'ours.training_bootstrap.prepare_worker'})
        manager = await ObservedVisualAgentManager.create(OmegaConf.create(case['config']))
        identity = EvaluationIdentity(case['model']['weights_sha256'], case['audit_data_sha256'],
            case['audit_data_sha256'], case['decode_sha256'], case['verifier_sha256'], 'development-after-native-rsft')
        receipt, result = await evaluate_pairs(manager, dataset, manifest_path=Path(case['manifest']),
            identity=identity, output=output / 'evaluation', batch_size=8)
        observed = native.audit_requests(case, output, result)
        native.write_new(output / 'result.json', {'status': 'COMPLETE_POST_TRAINING_VISUAL_CASE',
            'role': 'V', 'mode': 'direct', 'job_id': os.environ['SLURM_JOB_ID'],
            'plan_sha256': file_sha256(case_path), 'audit': asdict(receipt),
            'paired_accuracy': receipt.paired_accuracy, 'marginal_accuracy': receipt.marginal_accuracy,
            'observed': observed, 'limitations': case['limitations']})
    finally:
        ray.shutdown()


def matched_results(before_root, after_root, manifest_path):
    """Require identical actual input/decoding multisets; compare by source ID."""
    import numpy as np
    def actual_inputs(root):
        keys = []
        for path in (root / 'requests').glob('*.request.json'):
            request = read(path)
            assert request['images'] and not request['videos_present']
            keys.append(fingerprint({k: request[k] for k in ('prompt_ids', 'images', 'sampling_params')}))
        assert len(keys) == 128
        return Counter(keys)
    assert actual_inputs(before_root) == actual_inputs(after_root), 'Actual pixels, prompt tokens or decoding changed'
    raw = [read(p / 'evaluation/result.json') for p in (before_root, after_root)]
    records = [{r['sample_id']: r for r in data['records']} for data in raw]
    assert len(records[0]) == len(records[1]) == 128 and set(records[0]) == set(records[1])
    manifest, _, _ = pair_inputs(manifest_path)
    evidence, values = [], [[], []]
    for pair in manifest['pairs']:
        sides = [fingerprint({'pair_id': pair['pair_id'], 'side': side}) for side in (0, 1)]
        answers, correct = [], []
        for index, rows in enumerate(records):
            answers.append([rows[k]['committed_answer'] for k in sides])
            correct.append([int(binary_answer_verifier(a, b)) for a, b in zip(answers[-1], pair['answers'])])
            assert correct[-1] == [rows[k]['correct'] for k in sides]
            assert all(rows[k]['policy_calls'] == 1 and not rows[k]['tool_trace'] for k in sides)
            values[index].append(correct[-1])
        evidence.append({'pair_id': pair['pair_id'], 'source_id': pair['source_id'],
                         'before_after_answers': answers, 'before_after_correctness': correct})
    assert len({p['source_id'] for p in evidence}) == len(evidence) == 64
    a, b = map(np.asarray, values)
    draws = np.random.default_rng(20260910).integers(0, 64, size=(20000, 64))
    metrics = {}
    for name, x, y in (('marginal_accuracy', a.mean(axis=1), b.mean(axis=1)),
                       ('paired_accuracy', a.prod(axis=1), b.prod(axis=1))):
        assert float(x.mean()) == raw[0][name] and float(y.mean()) == raw[1][name]
        delta = y - x
        metrics[name] = {'before': float(x.mean()), 'after': float(y.mean()), 'difference': float(delta.mean()),
            'source_paired_bootstrap_95_percentile_interval': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
    return {'actual_input_and_decoding_multisets_equal': True, 'metrics': metrics,
        'changed_answers': sum(records[0][k]['committed_answer'] != records[1][k]['committed_answer'] for k in records[0]),
        'pair_evidence': evidence, 'bootstrap': {'unit': 'source chart', 'replicates': 20000, 'seed': 20260910},
        'limitations': ['Same development subset after one engineering training batch; not VETO efficacy.',
            'Intervals condition on one model pair; identical predictions imply no observed change, not population certainty.']}


def run(path):
    plan = check(path)
    output = Path(plan['output'])
    output.mkdir()
    native.write_new(output / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'],
        'plan_sha256': file_sha256(path), 'free_bytes': shutil.disk_usage(ROOT).free})
    for name in plan['source_sha256']:
        src = ROOT / name
        target = output / 'source' / src.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    try:
        base, native_dir, target = Path(plan['base']['path']), Path(plan['native']), Path(plan['target'])
        subprocess.run([sys.executable, '-m', 'ours.canonical_native_export', '--base', str(base),
            '--native', str(native_dir), '--target', str(target), '--report', str(output / 'serialization.json')],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        asset = restore_generation_asset(base, target, output / 'generated-generation_config.json')
        contract = inference_contract(base, target)
        comparison = compare_active_parameters(base, native_dir, target, aliases=plan['aliases'], tensor_count=plan['active_tensors'])
        verify_files(plan['resume_sha256'])
        transition = {'kind': 'native_visual_checkpoint_transition', 'status': 'PASS', 'base': plan['base'],
            'exported': checkpoint_manifest(target), 'plan_sha256': file_sha256(path),
            'native_checkpoint_sha256': plan['resume_sha256'][str(native_dir / 'model_world_size_1_rank_0.pt')],
            'inference_contract': contract, 'generation_asset_restoration': asset,
            'resume_artifacts_unchanged': True, **comparison, 'limitations': plan['limitations']}
        native.write_new(output / 'transition.json', transition)
        print(json.dumps({k: {n: v for n, v in comparison[k].items() if n != 'tensors'}
                          for k in ('native_fp32', 'exported_bf16')}), flush=True)
        case = make_case(plan, transition)
        case_path = output / 'direct-plan.json'
        native.write_new(case_path, case)
        asyncio.run(evaluate(case, case_path))
        result = matched_results(Path(plan['baseline_output']), Path(case['output']), Path(case['manifest']))
        native.write_new(output / 'comparison.json', result)
        native.write_new(output / 'result.json', {'status': 'COMPLETE_NATIVE_VISUAL_FOLLOWUP',
            'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path),
            'transition_sha256': file_sha256(output / 'transition.json'),
            'comparison_sha256': file_sha256(output / 'comparison.json'), 'metrics': result['metrics'],
            'formal_matrix_started': False, 'scientific_method_verified': False})
        print(json.dumps(result['metrics']), flush=True)
    except BaseException as error:
        native.write_new(output / 'failure.json', {'status': 'INCOMPLETE', 'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('prepare', 'check', 'run'), required=True)
    for name in ('plan', 'output', 'training-plan', 'baseline-plan', 'batch-audit'):
        p.add_argument('--' + name, type=Path, required=name == 'plan')
    a = p.parse_args()
    if a.phase == 'prepare':
        prepare(a.plan.resolve(), a.output.resolve(), a.training_plan.resolve(), a.baseline_plan.resolve(), a.batch_audit.resolve())
    elif a.phase == 'check':
        check(a.plan.resolve())
    else:
        run(a.plan.resolve())
