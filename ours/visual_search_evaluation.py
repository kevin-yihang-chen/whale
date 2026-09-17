"""Evaluate one visual search candidate on H and, when requested, paired C.

Shared native generation supplies WHALE task accuracy/turns on H and E1 on C.
The fixed-weight identity excludes candidate code and output locations; those
have separate hashes. This evaluator never trains, proposes, or selects h.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import shutil

from .audit_native_training_batch import require
from . import native_visual_service as native
from .acceptance import CandidateMetrics
from .evidence import EvaluationIdentity, fingerprint
from .local_completion import checkpoint_manifest
from .visual_dataset_materialization import single_rows
from .visual_harness import load_visual_harness
from .visual_native_evaluation import evaluate_pairs, pair_inputs, verifier_identity
from .visual_task import binary_answer_verifier, file_sha256

ROOT = native.ROOT
EXTRA = ('ours/visual_search_evaluation.py', 'ours/run_visual_search_evaluation.sh',
    'ours/visual_dataset_materialization.py', 'ours/acceptance.py')


def read(path):
    return json.loads(Path(path).read_text())


def partition_image_count(plan, role):
    """Return the frozen, manifest-backed image count for one evaluation role."""
    partition = plan['partitions'][role]
    manifest = read(partition['manifest'])
    actual = len(manifest['image_files'])
    declared = partition.get('examples', actual)
    require(type(declared) is int and declared > 0 and declared == actual,
            f'Changed {role} partition image count')
    return declared


def decode_identity(config, model_assets):
    fixed = deepcopy(config)
    fixed['data'].pop('visual_harness_path')
    fixed['data'].pop('cache_dir')
    fixed.pop('trainer')
    fixed['actor_rollout_ref']['rollout'].pop('trace')
    return fingerprint({'config': fixed, 'model_assets': model_assets})


def prepare(path, output, reference_path, harness_path, phase, name, with_audit, repeatability_check=False):
    require(not path.exists() and (not output.exists()), 'Validation failed: not path.exists() and (not output.exists())')
    require(name and phase and harness_path.is_file(), 'Validation failed: name and phase and harness_path.is_file()')
    reference = read(reference_path)
    require(reference['mode'] == 'direct' and reference['role'] == 'V', "Validation failed: reference['mode'] == 'direct' and reference['role'] == 'V'")
    require(reference['model'] == checkpoint_manifest(Path(reference['model']['path'])), "Validation failed: reference['model'] == checkpoint_manifest(Path(reference['model']['path']))")
    load_visual_harness(harness_path)
    materialization_path = ROOT / 'data/plotqa-evidence-native-20260910-v1/result.json'
    materialization = read(materialization_path)
    partitions = {r: materialization['partitions'][r] for r in ('H', 'C')}
    h_manifest, h_rows = single_rows(partitions['H']['manifest'])
    c_manifest, pairs, _ = pair_inputs(partitions['C']['manifest'])
    require(len(h_rows) == 128 and len(pairs) == 64, 'Validation failed: len(h_rows) == 128 and len(pairs) == 64')
    require(not set(h_manifest['source_tables']) & set(c_manifest['source_tables']), "Validation failed: not set(h_manifest['source_tables']) & set(c_manifest['source_tables'])")
    plan = deepcopy(reference)
    plan.update(kind='visual_search_candidate_evaluation', created_at_utc=datetime.now(timezone.utc).isoformat(),
        role='H+C' if with_audit else 'H', output=str(output), candidate=name, phase=phase,
        reference_plan=str(reference_path), reference_plan_sha256=file_sha256(reference_path),
        partitions=partitions, with_audit=with_audit, manifest=partitions['C']['manifest'],
        manifest_sha256=partitions['C']['manifest_sha256'], audit_data_sha256=c_manifest['audit_data_sha256'],
        materialization=str(materialization_path), materialization_sha256=file_sha256(materialization_path))
    cfg = plan['config']
    cfg['data']['visual_harness_path'] = str(harness_path)
    cfg['data']['cache_dir'] = str(output / 'dataset-cache')
    cfg['trainer']['experiment_name'] = output.name
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name'] = output.name
    # Shared measurement protocol: one request in flight, no prefix or chunk
    # reuse. Verify repeats empirically; greedy decoding alone was not stable.
    cfg['actor_rollout_ref']['rollout'].update(max_num_seqs=1, enable_prefix_caching=False,
        enable_chunked_prefill=False)
    cfg['actor_rollout_ref']['rollout']['agent']['num_workers'] = 1
    require(cfg['data']['visual_harness_execution'] == 'isolated', "Validation failed: cfg['data']['visual_harness_execution'] == 'isolated'")
    require(cfg['data']['tool_config_path'] is None, "Validation failed: cfg['data']['tool_config_path'] is None")  # Current engineering condition, not a formal h0.
    plan['harness_sha256'] = file_sha256(harness_path)
    plan['decode_sha256'] = decode_identity(cfg, plan['model']['assets'])
    plan['identity'] = asdict(EvaluationIdentity(plan['model']['weights_sha256'], partitions['H']['manifest_sha256'],
        c_manifest['audit_data_sha256'], plan['decode_sha256'], verifier_identity(), phase))
    plan['source_sha256'].update({p: file_sha256(ROOT / p) for p in EXTRA})
    plan['source_sha256'][str(harness_path)] = plan['harness_sha256']
    for p, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT / p) == digest, p)
    plan['batch_size'] = 1
    plan['repeatability_check'] = repeatability_check
    images = (256 if with_audit else 128) * (2 if repeatability_check else 1)
    plan['bounds'] = {'images': images, 'maximum_generation_calls': images * 3,
        'maximum_generated_assistant_tokens': images * 1024, 'gpus': 1, 'cpus': 12,
        'time_limit_seconds': 1800, 'new_model_checkpoints': 0, 'new_optimizer_steps': 0, 'api_calls': 0}
    plan['storage'] = {'additional_working_space_gib': 12 if repeatability_check else 8, 'minimum_free_margin_gib': 40}
    plan['limitations'] = ['H and C are optimization data, not independent validation or test performance.',
        'Current direct-answer engineering condition; formal common h0 and learning-rate calibration remain pending.',
        'This evaluator supplies evidence only: no proposer, accepted-harness update, or VETO efficacy claim.',
        'Candidate callbacks run in the existing filesystem/network-restricted subprocesses.']
    plan['measurement_revision'] = {'reason': 'Same trained checkpoint changed 2/128 predictions on repeated batched V evaluation.',
        'config_changes': {'max_num_seqs': 1, 'agent.num_workers': 1, 'batch_size': 1,
                           'enable_prefix_caching': False, 'enable_chunked_prefill': False},
        'selection_pass': 'First complete pass only; repeats diagnose variability, never select the better score.',
        'limitation': 'A matching repeat is empirical evidence for this workload, not a general determinism guarantee.'}
    native.write_new(path, plan)


def check(path):
    plan = read(path)
    require(plan['kind'] == 'visual_search_candidate_evaluation', "Validation failed: plan['kind'] == 'visual_search_candidate_evaluation'")
    require(file_sha256(Path(plan['reference_plan'])) == plan['reference_plan_sha256'], "Validation failed: file_sha256(Path(plan['reference_plan'])) == plan['reference_plan_sha256']")
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT / name) == digest, name)
    require(checkpoint_manifest(Path(plan['model']['path'])) == plan['model'], "Validation failed: checkpoint_manifest(Path(plan['model']['path'])) == plan['model']")
    require(file_sha256(Path(plan['materialization'])) == plan['materialization_sha256'], "Validation failed: file_sha256(Path(plan['materialization'])) == plan['materialization_sha256']")
    for item in plan['partitions'].values():
        for key in ('manifest', 'parquet'):
            require(file_sha256(Path(item[key])) == item[key + '_sha256'], "Validation failed: file_sha256(Path(item[key])) == item[key + '_sha256']")
    require(plan['decode_sha256'] == decode_identity(plan['config'], plan['model']['assets']), "Validation failed: plan['decode_sha256'] == decode_identity(plan['config'], plan['model']['assets'])")
    rollout = plan['config']['actor_rollout_ref']['rollout']
    require(plan['batch_size'] == rollout['max_num_seqs'] == rollout['agent']['num_workers'] == 1, "Validation failed: plan['batch_size'] == rollout['max_num_seqs'] == rollout['agent']['num_workers'] == 1")
    require(not rollout['enable_prefix_caching'] and (not rollout['enable_chunked_prefill']), "Validation failed: not rollout['enable_prefix_caching'] and (not rollout['enable_chunked_prefill'])")
    require(EvaluationIdentity(**plan['identity']).verifier_sha256 == verifier_identity(), "Validation failed: EvaluationIdentity(**plan['identity']).verifier_sha256 == verifier_identity()")
    require(not Path(plan['output']).exists(), "Validation failed: not Path(plan['output']).exists()")
    require(shutil.disk_usage(ROOT).free >= sum(plan['storage'].values()) * 1024 ** 3, "Validation failed: shutil.disk_usage(ROOT).free >= sum(plan['storage'].values()) * 1024 ** 3")
    return plan


async def evaluate_h(manager, dataset, manifest_path, output, batch_size=1):
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn
    manifest, expected_rows = single_rows(manifest_path)
    require(manifest['role'] == 'H', "Validation failed: manifest['role'] == 'H'")
    expected = {r['visual_sample_id']: r for r in expected_rows}
    require(len(dataset) == len(expected) == 128, 'Validation failed: len(dataset) == len(expected) == 128')
    incoming = list(dataset.dataframe)
    require(len({r['visual_sample_id'] for r in incoming}) == 128, "Validation failed: len({r['visual_sample_id'] for r in incoming}) == 128")
    for row in incoming:
        target = expected[row['visual_sample_id']]
        require(all((row[k] == target[k] for k in ('prompt', 'images', 'reward_model', 'data_source'))), "Validation failed: all((row[k] == target[k] for k in ('prompt', 'images', 'reward_model', 'data_source')))")
    output.mkdir()
    records, artifacts = {}, {}
    try:
        for offset in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(offset, offset + batch_size)]
            batch = DataProto.from_single_dict(collate_fn(items))
            batch.meta_info.update(validate=True)
            result = await manager.generate_sequences(batch)
            archive = output / f'batch-{offset // batch_size:04d}.pkl'
            result.save_to_disk(archive)
            artifacts[archive.name] = file_sha256(archive)
            meta = result.non_tensor_batch
            actual_ids = meta['visual_sample_id'].tolist()
            require(len(result) == len(set(actual_ids)) == len(items), 'Validation failed: len(result) == len(set(actual_ids)) == len(items)')
            require(set(actual_ids) == {r['visual_sample_id'] for r in items}, "Validation failed: set(actual_ids) == {r['visual_sample_id'] for r in items}")
            for i, sample in enumerate(actual_ids):
                require(sample not in records and meta['visual_harness_sha256'][i] == dataset.visual_harness.sha256, "Validation failed: sample not in records and meta['visual_harness_sha256'][i] == dataset.visual_harness.sha256")
                raw, committed = meta['visual_final_answer'][i], meta['visual_committed_answer'][i]
                require(dataset.visual_harness.invoke('parse_answer', text=raw) == committed, "Validation failed: dataset.visual_harness.invoke('parse_answer', text=raw) == committed")
                correct = int(binary_answer_verifier(committed, expected[sample]['reward_model']['ground_truth']))
                mask = result.batch['response_mask'][i]
                attention = result.batch['attention_mask'][i, -len(mask):]
                scores = result.batch['rm_scores'][i]
                require(bool(((mask == 0) | (mask == 1)).all()) and (not bool((mask > attention).any())), 'Validation failed: bool(((mask == 0) | (mask == 1)).all()) and (not bool((mask > attention).any()))')
                require(scores.sum().item() == correct, 'Validation failed: scores.sum().item() == correct')
                require(scores.nonzero().flatten().tolist() == ([int(attention.sum()) - 1] if correct else []), 'Validation failed: scores.nonzero().flatten().tolist() == ([int(attention.sum()) - 1] if correct else [])')
                calls, tokens = meta['visual_policy_calls'][i], meta['visual_generated_tokens'][i]
                require(type(calls) is int and calls > 0 and (type(tokens) is int) and (tokens >= int(mask.sum())), 'Validation failed: type(calls) is int and calls > 0 and (type(tokens) is int) and (tokens >= int(mask.sum()))')
                records[sample] = {'sample_id': sample, 'raw_answer': raw, 'committed_answer': committed,
                    'correct': correct, 'native_turns': int(meta['__num_turns__'][i]), 'policy_calls': calls,
                    'generated_tokens': tokens, 'tool_trace': meta['visual_harness_tool_trace'][i]}
            native.write_new(output / f'batch-{offset // batch_size:04d}.json', [records[k] for k in actual_ids])
        require(set(records) == set(expected), 'Validation failed: set(records) == set(expected)')
        dataset.visual_harness.unchanged()
        ordered = [records[r['visual_sample_id']] for r in expected_rows]
        result = {'status': 'COMPLETE_NATIVE_H_EVALUATION', 'role': 'H', 'examples': len(ordered),
            'accuracy': sum(r['correct'] for r in ordered) / len(ordered),
            'mean_native_turns': sum(r['native_turns'] for r in ordered) / len(ordered),
            'policy_calls': sum(r['policy_calls'] for r in ordered),
            'generated_tokens': sum(r['generated_tokens'] for r in ordered),
            'records': ordered, 'batch_artifact_sha256': artifacts}
        native.write_new(output / 'result.json', result)
        return result
    except BaseException as error:
        native.write_new(output / 'failure.json', {'status': 'INCOMPLETE_NO_AGGREGATE_SCORE',
            'error_type': type(error).__name__, 'error': str(error), 'completed_examples': len(records),
            'batch_artifact_sha256': artifacts})
        raise


async def run(plan, path):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import ray
    import torch
    from omegaconf import OmegaConf
    from .native_visual_agent_observation import ObservedVisualAgentManager
    require(torch.cuda.device_count() == 1 and 'H800' in torch.cuda.get_device_name(0), "Validation failed: torch.cuda.device_count() == 1 and 'H800' in torch.cuda.get_device_name(0)")
    require(int(os.environ['SLURM_CPUS_PER_TASK']) == 12, "Validation failed: int(os.environ['SLURM_CPUS_PER_TASK']) == 12")
    output = Path(plan['output'])
    output.mkdir()
    for name in ('requests', 'workers'):
        (output / name).mkdir()
    for name in plan['source_sha256']:
        src = ROOT / name
        dest = output / 'source' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path), VETO_NATIVE_EVALUATION_OUTPUT=str(output))
    native.write_new(output / 'start.json', {'job_id': os.environ['SLURM_JOB_ID'], 'plan_sha256': file_sha256(path)})
    try:
        datasets = {role: native.dataset_for({**plan, 'bounds': {
                **plan['bounds'], 'images': partition_image_count(plan, role)}},
            Path(plan['partitions'][role]['parquet'])) for role in (('H', 'C') if plan['with_audit'] else ('H',))}
        env = {k: v for k, v in os.environ.items() if k.startswith(('VETO_', 'HF_', 'TRANSFORMERS_', 'VERL_', 'VLLM_')) or
            k in ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'NO_PROXY', 'no_proxy')}
        ray.init(num_cpus=12, num_gpus=1, include_dashboard=False, object_store_memory=2*1024**3,
            runtime_env={'env_vars': env, 'worker_process_setup_hook': 'ours.training_bootstrap.prepare_worker'})
        manager = await ObservedVisualAgentManager.create(OmegaConf.create(plan['config']))
        h = await evaluate_h(manager, datasets['H'], Path(plan['partitions']['H']['manifest']), output / 'H', plan['batch_size'])
        native.write_new(output / 'H/request-membership.json', sorted(p.name for p in (output / 'requests').glob('*.request.json')))
        calls, tokens = h['policy_calls'], h['generated_tokens']
        paired = None
        if plan['with_audit']:
            paired, c = await evaluate_pairs(manager, datasets['C'], manifest_path=Path(plan['partitions']['C']['manifest']),
                identity=EvaluationIdentity(**plan['identity']), output=output / 'C', batch_size=plan['batch_size'])
            calls += c['policy_calls']
            tokens += c['generated_tokens']
        repeated = None
        if plan['repeatability_check']:
            again_h = await evaluate_h(manager, datasets['H'], Path(plan['partitions']['H']['manifest']),
                output / 'H-repeat', plan['batch_size'])
            calls += again_h['policy_calls']
            tokens += again_h['generated_tokens']
            def differences(a, b, expected_count):
                left = {r['sample_id']: r for r in a['records']}
                right = {r['sample_id']: r for r in b['records']}
                require(set(left) == set(right) and len(left) == expected_count,
                        'Repeatability record count differs from the frozen partition')
                return {field: sum(left[k][field] != right[k][field] for k in left)
                        for field in ('committed_answer', 'raw_answer', 'native_turns')}
            repeated = {'H': differences(h, again_h, partition_image_count(plan, 'H'))}
            if plan['with_audit']:
                again_paired, again_c = await evaluate_pairs(manager, datasets['C'],
                    manifest_path=Path(plan['partitions']['C']['manifest']), identity=EvaluationIdentity(**plan['identity']),
                    output=output / 'C-repeat', batch_size=plan['batch_size'])
                calls += again_c['policy_calls']
                tokens += again_c['generated_tokens']
                repeated['C'] = differences(c, again_c, partition_image_count(plan, 'C'))
            native.write_new(output / 'repeatability.json', {'comparisons': repeated,
                'selection_uses_first_pass': True, 'limitation': 'Same-engine repetition only; no population determinism guarantee.'})
        observed = native.audit_requests(plan, output, {'policy_calls': calls, 'generated_tokens': tokens})
        value = {'status': 'COMPLETE_VISUAL_SEARCH_CANDIDATE', 'job_id': os.environ['SLURM_JOB_ID'],
            'plan_sha256': file_sha256(path), 'identity': plan['identity'], 'harness_sha256': plan['harness_sha256'],
            'candidate': asdict(CandidateMetrics(plan['candidate'], plan['harness_sha256'], h['accuracy'], h['mean_native_turns'])),
            'audit': asdict(paired) if paired is not None else None, 'observed': observed,
            'repeatability': repeated,
            'repeatability_passed': all(not n for row in repeated.values() for n in row.values()) if repeated else None,
            'limitations': plan['limitations']}
        native.write_new(output / 'result.json', value)
        print(json.dumps({'candidate': value['candidate'], 'paired_accuracy_C': paired.paired_accuracy if paired else None,
                          'policy_calls': calls, 'generated_tokens': tokens}), flush=True)
    except BaseException as error:
        native.write_new(output / 'failure.json', {'status': 'INCOMPLETE', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run'), required=True)
    for name in ('plan', 'output', 'reference-plan', 'harness'):
        parser.add_argument('--' + name, type=Path, required=name == 'plan')
    parser.add_argument('--weight-phase')
    parser.add_argument('--name')
    parser.add_argument('--with-audit', action='store_true')
    parser.add_argument('--repeatability-check', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.plan.resolve(), args.output.resolve(), args.reference_plan.resolve(), args.harness.resolve(),
                args.weight_phase, args.name, args.with_audit, args.repeatability_check)
    else:
        plan = check(args.plan.resolve())
        if args.phase == 'run':
            asyncio.run(run(plan, args.plan.resolve()))
