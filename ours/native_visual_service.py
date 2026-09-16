"""Launch real E1 paired evaluation through the shared native visual runtime.

The initial launcher is explicitly restricted to the existing engineering
fixture and trusted h0. It establishes actual loading/generation/measurement;
it neither trains a model nor executes generated candidate code.
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

from .audit_native_training_batch import require
from .evidence import EvaluationIdentity, fingerprint
from .local_completion import checkpoint_manifest, checkpoint_tensor
from .native_search import tree_hashes
from .visual_harness import BASE, load_visual_harness
from .visual_native_evaluation import pair_inputs, write_pair_parquet, evaluate_pairs, verifier_identity
from .visual_task import file_sha256

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ('ours/native_visual_service.py', 'ours/native_visual_model_worker.py',
    'ours/local_completion.py', 'ours/audit_native_training_batch.py',
    'ours/native_visual_agent_observation.py', 'ours/run_native_visual_service.sh',
    'ours/visual_native_evaluation.py', 'ours/visual_harness.py', 'ours/visual_harness_dataset.py',
    'ours/visual_harness_loop.py', 'ours/visual_evidence_dataset.py', 'ours/visual_evidence_tool_loop.py',
    'ours/visual_evidence_reward.py', 'ours/visual_task.py', 'ours/evidence.py', 'ours/training_bootstrap.py',
    'ours/visual_harnesses/base_harness.py', 'ours/visual_harness_agents.yaml', 'ours/visual_evidence_tools.yaml',
    'ours/callback_confinement.py', 'ours/visual_callback_worker.py', 'ours/isolated_visual_harness.py')


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def configuration(model, manifest, output):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from omegaconf import OmegaConf
    from verl.workers.config import RolloutConfig
    rollout = OmegaConf.structured(RolloutConfig(name='vllm'))
    settings = {'nnodes': 1, 'n_gpus_per_node': 1, 'tensor_model_parallel_size': 1,
        'data_parallel_size': 1, 'pipeline_model_parallel_size': 1,
        'dtype': 'bfloat16', 'load_format': 'auto', 'enforce_eager': True, 'gpu_memory_utilization': .6,
        'max_model_len': 8192, 'max_num_batched_tokens': 8192, 'max_num_seqs': 8,
        'enable_sleep_mode': False, 'enable_prefix_caching': True, 'enable_chunked_prefill': True,
        'prompt_length': 4096, 'response_length': 2048, 'calculate_log_probs': True,
        'temperature': 0., 'top_p': 1., 'top_k': -1, 'n': 1,
        'val_kwargs.temperature': 0., 'val_kwargs.top_p': 1., 'val_kwargs.top_k': -1, 'val_kwargs.n': 1,
        'agent.num_workers': 2, 'agent.default_agent_loop': 'visual_harness_agent',
        'agent.agent_loop_config_path': str(ROOT/'ours/visual_harness_agents.yaml'),
        'multi_turn.enable': True, 'multi_turn.max_user_turns': 3, 'multi_turn.max_assistant_turns': 3,
        'multi_turn.max_assistant_tokens': 1024, 'multi_turn.max_parallel_calls': 1,
        'multi_turn.max_tool_response_length': 512, 'multi_turn.tool_response_truncate_side': 'right',
        'multi_turn.tool_config_path': str(ROOT/'ours/visual_evidence_tools.yaml'), 'multi_turn.format': 'qwen3_coder',
        'trace.project_name': 'veto-engineering', 'trace.experiment_name': output.name,
        'engine_kwargs.vllm': {'gdn_prefill_backend': 'triton',
            'distributed_executor_backend': 'uni', 'seed': 42,
            'worker_cls': 'ours.native_visual_model_worker.NativeVisualModelWorker'}}
    for key, value in settings.items():
        OmegaConf.update(rollout, key, value, merge=False, force_add=True)
    def bind_targets(node):
        if OmegaConf.is_dict(node):
            if node.get('_target_') == '':
                klass = OmegaConf.get_type(node)
                require(klass is not dict, 'Structured native config lost its dataclass type')
                node['_target_'] = f'{klass.__module__}.{klass.__qualname__}'
            for child in node.values():
                bind_targets(child)
        elif OmegaConf.is_list(node):
            for child in node:
                bind_targets(child)
    bind_targets(rollout)
    return OmegaConf.create({'actor_rollout_ref': {
        'model': {'_target_': 'verl.workers.config.HFModelConfig', 'path': str(model), 'trust_remote_code': False},
        'rollout': OmegaConf.to_container(rollout, resolve=True)},
        'data': {'prompt_key': 'prompt', 'image_key': 'images', 'image_patch_size': 16,
            'max_prompt_length': 4096, 'max_response_length': 2048, 'return_raw_chat': True,
            'filter_overlong_prompts': True, 'filter_overlong_prompts_workers': 1,
            'cache_dir': str(output/'dataset-cache'), 'apply_chat_template_kwargs': {'enable_thinking': False},
            'custom_cls': {'path': 'pkg://ours.visual_harness_dataset', 'name': 'VisualHarnessDataset'},
            'visual_harness_path': str(BASE.resolve()), 'visual_harness_execution': 'isolated',
            'tool_config_path': str(ROOT/'ours/visual_evidence_tools.yaml')},
        'trainer': {'project_name': 'veto-engineering', 'experiment_name': output.name},
        'reward': {'custom_reward_function': {'path': 'pkg://ours.visual_evidence_reward', 'name': 'compute_score'}}})


def embedding_coordinates(model):
    tensor = checkpoint_tensor(model, 'model.language_model.embed_tokens.weight')
    return [{'token_id': row, 'column': 137 * i % tensor.shape[1],
        'expected': float(tensor[row, 137 * i % tensor.shape[1]])}
        for i, row in enumerate((0, 7, 42, 123, 502, 1024, 4096, 8192))]


def prepare(path, manifest_path, output):
    from omegaconf import OmegaConf
    manifest_path, output = manifest_path.resolve(), output.resolve()
    manifest, pairs, _ = pair_inputs(manifest_path)
    require(manifest['role'] == 'engineering' and len(pairs) == 8, 'Initial service check uses only the eight engineering pairs')
    require(not path.exists() and not output.exists(), 'Preserve previous plans and runs')
    model = ROOT/'data/models/qwen3.5-4b-common-bf16-v1'
    cfg = configuration(model, manifest_path, output)
    model_manifest = checkpoint_manifest(model)
    plan = {'kind': 'native_visual_service_engineering_plan', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'engineering', 'manifest': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
        'output': str(output), 'model': model_manifest, 'worker_coordinates': embedding_coordinates(model),
        'config': OmegaConf.to_container(cfg, resolve=True), 'batch_size': 8, 'seed': 42,
        'harness_sha256': file_sha256(BASE), 'audit_data_sha256': manifest['audit_data_sha256'],
        'verifier_sha256': verifier_identity(),
        'source_sha256': {name: file_sha256(ROOT/name) for name in SOURCES},
        'versions': {name: version(name) for name in ('torch', 'vllm', 'transformers', 'ray', 'qwen-vl-utils')},
        'bounds': {'images': 16, 'maximum_generation_calls': 48, 'maximum_generated_assistant_tokens': 16384,
            'gpus': 1, 'time_limit_seconds': 2400, 'proposer_api_calls': 0, 'optimizer_steps': 0},
        'limitations': ['Engineering figures and trusted baseline harness only; no scientific performance result.',
            'Native Manager/Worker and vLLM startup are exercised, with unchanged shared visual generation.',
            'Loaded parameter coordinates are spot checks, not a complete in-memory weight hash.',
            'No training, candidate proposal or VETO acceptance is performed by this launcher.']}
    plan['decode_sha256'] = fingerprint({'rollout': plan['config']['actor_rollout_ref']['rollout'],
        'chat_template': plan['config']['data']['apply_chat_template_kwargs'], 'model_assets': model_manifest['assets']})
    write_new(path, plan)
    return plan


def check(path, *, verify_weights=True):
    from omegaconf import OmegaConf
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'native_visual_service_engineering_plan' and plan['role'] == 'engineering', 'Wrong visual service scope')
    require(plan['bounds'] == {'images': 16, 'maximum_generation_calls': 48, 'maximum_generated_assistant_tokens': 16384,
        'gpus': 1, 'time_limit_seconds': 2400, 'proposer_api_calls': 0, 'optimizer_steps': 0}, 'Different engineering budget')
    require(set(plan['source_sha256']) == set(SOURCES), 'Incomplete source manifest')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(ROOT/name) == digest, f'Changed visual service source: {name}')
    require(plan['versions'] == {name: version(name) for name in plan['versions']}, 'Changed visual runtime')
    require(file_sha256(Path(plan['manifest'])) == plan['manifest_sha256'], 'Changed visual input manifest')
    manifest, pairs, _ = pair_inputs(Path(plan['manifest']))
    require(manifest['role'] == 'engineering' and len(pairs) == 8 and manifest['audit_data_sha256'] == plan['audit_data_sha256'], 'Changed audit data')
    require(plan['harness_sha256'] == load_visual_harness(BASE).sha256 and plan['verifier_sha256'] == verifier_identity(), 'Changed harness or verifier')
    require(plan['config']['data']['visual_harness_path'] == str(BASE.resolve()), 'Generated candidate is outside this launcher')
    expected_model = ROOT/'data/models/qwen3.5-4b-common-bf16-v1'
    require(plan['model']['path'] == str(expected_model), 'Wrong initial engineering model')
    expected_config = configuration(expected_model, Path(plan['manifest']), Path(plan['output']))
    require(plan['config'] == OmegaConf.to_container(expected_config, resolve=True), 'Changed shared visual configuration')
    decode = fingerprint({'rollout': plan['config']['actor_rollout_ref']['rollout'],
        'chat_template': plan['config']['data']['apply_chat_template_kwargs'], 'model_assets': plan['model']['assets']})
    require(decode == plan['decode_sha256'], 'Changed decoding identity')
    if verify_weights:
        require(checkpoint_manifest(Path(plan['model']['path'])) == plan['model'], 'Changed visual checkpoint bytes')
        require(embedding_coordinates(Path(plan['model']['path'])) == plan['worker_coordinates'], 'Changed worker coordinates')
    return plan


def dataset_for(plan, parquet, *, cache_override=None):
    from omegaconf import OmegaConf
    from verl.utils.tokenizer import hf_tokenizer, hf_processor
    from verl.trainer.main_ppo import create_rl_dataset
    model = plan['model']['path']
    tokenizer = hf_tokenizer(model, local_files_only=True)
    processor = hf_processor(model, local_files_only=True)
    cfg = OmegaConf.create(plan['config'])
    if cache_override is not None:
        cfg.data.cache_dir = str(cache_override)
    require(processor is not None and processor.image_processor.patch_size == cfg.data.image_patch_size, 'Different visual processor geometry')
    dataset = create_rl_dataset(str(parquet), cfg.data, tokenizer, processor, is_train=False)
    require(len(dataset) == plan['bounds']['images'], 'Filtering removed engineering pairs')
    for index in range(len(dataset)):
        dataset[index]
    return dataset


def audit_requests(plan, output, result):
    starts = sorted((output/'requests').glob('*.request.json'))
    require(len(starts) == result['policy_calls'] <= plan['bounds']['maximum_generation_calls'], 'Generation call ledger differs')
    total = 0
    images = set()
    expected = {'temperature': 0., 'top_p': 1., 'top_k': -1}
    for path in starts:
        request = json.loads(path.read_text())
        require(request['prompt_ids'] and request['images'] and not request['videos_present'], 'Actual request lost visual evidence')
        require(all(request['sampling_params'].get(k) == v for k,v in expected.items()), 'Actual sampling differs from engineering plan')
        response = json.loads((output/'requests'/f"{request['id']}.response.json").read_text())
        require(response['id'] == request['id'], 'Response identity differs')
        total += len(response['token_ids'])
        images.update(image['pixels_sha256'] for image in request['images'])
    require(total == result['generated_tokens'] <= plan['bounds']['maximum_generated_assistant_tokens'], 'Generated token ledger differs')
    require(not list((output/'requests').glob('*.failure.json')), 'Generation failed')
    require(len(list((output/'requests').glob('*.response.json'))) == len(starts), 'Unaccounted native response')
    receipts = list((output/'workers').glob('model-*.json'))
    require(len(receipts) == 1, 'Expected one actual model worker')
    worker = json.loads(receipts[0].read_text())
    require(worker['status'] == 'PASS_LOADED_COORDINATES' and worker['seed'] == plan['seed'] and
        worker['tensor_parallel_size'] == 1, 'Wrong loaded service seed or parallelism')
    require(worker['plan_sha256'] == json.loads((output/'start.json').read_text())['plan_sha256'], 'Worker loaded a different plan')
    return {'policy_calls': len(starts), 'generated_tokens': total, 'distinct_model_visible_images': len(images),
        'sampling_verified_at_native_dispatch': True, 'worker_receipt': worker,
        'request_artifact_sha256': tree_hashes(output/'requests')}


async def run(plan, path):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import ray
    import torch
    from omegaconf import OmegaConf
    from .native_visual_agent_observation import ObservedVisualAgentManager
    output = Path(plan['output'])
    require(torch.cuda.device_count() == 1 and 'H800' in torch.cuda.get_device_name(0), 'Expected one H800 allocation')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('requests', 'workers'):
        (output/name).mkdir()
    for name in SOURCES:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    (output/'plan.json').write_bytes(path.read_bytes())
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path.resolve()), VETO_NATIVE_EVALUATION_OUTPUT=str(output))
    write_new(output/'start.json', {'status': 'STARTED', 'job_id': os.environ['SLURM_JOB_ID'],
        'plan_sha256': file_sha256(path), 'bounds': plan['bounds']})
    parquet = output/'pairs.parquet'
    write_new(output/'materialization.json', write_pair_parquet(Path(plan['manifest']), parquet))
    try:
        dataset = dataset_for(plan, parquet)
        env = {key: value for key, value in os.environ.items() if key.startswith(('VETO_', 'HF_', 'TRANSFORMERS_', 'VERL_', 'VLLM_')) or key in ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'NO_PROXY', 'no_proxy')}
        ray.init(num_cpus=16, num_gpus=1, include_dashboard=False, object_store_memory=2*1024**3,
            runtime_env={'env_vars': env, 'worker_process_setup_hook': 'ours.training_bootstrap.prepare_worker'})
        manager = await ObservedVisualAgentManager.create(OmegaConf.create(plan['config']))
        identity = EvaluationIdentity(plan['model']['weights_sha256'], plan['audit_data_sha256'],
            plan['audit_data_sha256'], plan['decode_sha256'], plan['verifier_sha256'], 'engineering-initial-weights')
        receipt, result = await evaluate_pairs(manager, dataset, manifest_path=Path(plan['manifest']), identity=identity,
            output=output/'evaluation', batch_size=plan['batch_size'])
        observed = audit_requests(plan, output, result)
        write_new(output/'result.json', {'status': 'COMPLETE_REAL_NATIVE_VISUAL_EVALUATION',
            'plan_sha256': file_sha256(path), 'job_id': os.environ['SLURM_JOB_ID'], 'role': 'engineering',
            'audit': asdict(receipt), 'paired_accuracy': receipt.paired_accuracy,
            'marginal_accuracy': receipt.marginal_accuracy, 'observed': observed,
            'evaluation_sha256': file_sha256(output/'evaluation/result.json'), 'limitations': plan['limitations']})
        print(json.dumps({'status': 'COMPLETE_REAL_NATIVE_VISUAL_EVALUATION', 'role': 'engineering',
            'paired_accuracy': receipt.paired_accuracy, 'marginal_accuracy': receipt.marginal_accuracy,
            'policy_calls': observed['policy_calls']}), flush=True)
    except BaseException as error:
        write_new(output/'failure.json', {'status': 'INCOMPLETE_ENGINEERING_RUN', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'check', 'run'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'prepare':
        require(args.manifest is not None and args.output is not None, 'Manifest and output required')
        report = prepare(args.plan, args.manifest, args.output)
        print(json.dumps({'status': 'PREPARED', 'plan_sha256': file_sha256(args.plan), 'role': report['role']}), flush=True)
    else:
        plan = check(args.plan)
        if args.phase == 'check':
            print(json.dumps({'status': 'PASS_SOURCE_AND_INPUT_CHECK', 'plan_sha256': file_sha256(args.plan)}), flush=True)
        else:
            require('SLURM_JOB_ID' in os.environ, 'Actual visual service requires an allocated compute job')
            asyncio.run(run(plan, args.plan))
