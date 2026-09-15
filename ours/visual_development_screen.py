"""Small PlotQA-derived development screen with a genuine no-image control.

E1 paired measurement through the native visual loop; no E2/E3 search or E4
update is performed. This separates image necessity from crop-tool failures.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import native_visual_service as native
from .evidence import EvaluationIdentity, fingerprint
from .visual_native_evaluation import pair_inputs, write_pair_parquet, evaluate_pairs
from .visual_task import file_sha256

ROOT = native.ROOT
MODES = ('direct', 'blind', 'normalized_zoom')
EXTRA = ('ours/visual_development_screen.py', 'ours/visual_blind_control.py',
         'ours/normalized_visual_zoom.py', 'ours/normalized_visual_tools.yaml',
         'ours/run_visual_development_screen.sh', 'ours/visual_harnesses/canonical_answer_harness.py')


def configure(base, manifest_path, mode, output):
    from omegaconf import OmegaConf
    manifest, pairs, _ = pair_inputs(manifest_path)
    assert manifest['role'] == 'V' and len(pairs) == 64
    case = deepcopy(base)
    case.update(kind='native_visual_development_case', role='V', mode=mode, output=str(output),
                manifest=str(manifest_path), manifest_sha256=file_sha256(manifest_path),
                audit_data_sha256=manifest['audit_data_sha256'])
    cfg = native.configuration(Path(base['model']['path']), manifest_path, output)
    cfg.data.visual_harness_path = str(ROOT/'ours/visual_harnesses/canonical_answer_harness.py')
    case['harness_sha256'] = file_sha256(Path(cfg.data.visual_harness_path))
    tools = str(ROOT/'ours/normalized_visual_tools.yaml') if mode == 'normalized_zoom' else None
    cfg.data.tool_config_path = tools
    cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path = tools
    if mode == 'blind':
        cfg.data.custom_cls = {'path': 'pkg://ours.visual_blind_control', 'name': 'BlindVisualHarnessDataset'}
    case['config'] = OmegaConf.to_container(cfg, resolve=True)
    case['bounds'] = {**base['bounds'], 'images': 128, 'maximum_generation_calls': 384,
                      'maximum_generated_assistant_tokens': 128*1024}
    case['decode_sha256'] = fingerprint({'config': case['config'], 'mode': mode,
                                         'model_assets': base['model']['assets']})
    case['source_sha256'].update({name: file_sha256(ROOT/name) for name in EXTRA})
    case['limitations'] = ['64 predetermined development pairs, not sealed test or official PlotQA score.',
        'Blind control retains original labels for grading but removes every image before tokenization.',
        'Tools differ across diagnostic conditions; this is not a formal VETO versus WHALE comparison.',
        'All conditions share the calibrated explicit-final-line parser; strict first-screen scores remain archived.',
        'No training, proposer calls, h0 selection, or VETO efficacy claim.']
    return case


def prepare(path, output, base_path):
    base = native.check(base_path)
    source = ROOT/'data/plotqa-evidence-pairs-20260910-v1/rendered/V/manifest.json'
    full, _, _ = pair_inputs(source)
    assert full['role'] == 'V' and full['partition'] == 'V' and len(full['pairs']) == 256
    assert not path.exists() and not output.exists()
    output.mkdir()
    subset = deepcopy(full)
    subset['pairs'] = full['pairs'][:64]  # Fixed stored order, before model results.
    hashes = {h for p in subset['pairs'] for h in p['images_sha256']}
    subset['image_files'] = {h: full['image_files'][h] for h in full['image_files'] if h in hashes}
    subset['audit_data_sha256'] = fingerprint(subset['pairs'])
    subset['source_manifest_sha256'] = file_sha256(source)
    subset['selection'] = 'First 64 pairs in the frozen V manifest; no answer-dependent selection.'
    manifest_path = output/'input/manifest.json'
    manifest_path.parent.mkdir()
    for relative in subset['image_files'].values():
        target = manifest_path.parent/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent/relative, target)
    native.write_new(manifest_path, subset)
    cases = {}
    for mode in MODES:
        target = output/f'{mode}-plan.json'
        native.write_new(target, configure(base, manifest_path, mode, output/mode))
        cases[mode] = {'path': str(target), 'sha256': file_sha256(target)}
    native.write_new(path, {'kind': 'visual_development_screen', 'base_plan': str(base_path),
        'base_plan_sha256': file_sha256(base_path), 'output': str(output), 'cases': cases,
        'source_manifest': str(source), 'source_manifest_sha256': file_sha256(source),
        'manifest': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
        'source_sha256': {p: file_sha256(ROOT/p) for p in EXTRA},
        'bounds': {'gpus': 1, 'cpus': 12, 'time_limit_seconds': 3600,
                   'maximum_generation_calls': 1152, 'new_model_checkpoints': 0, 'api_calls': 0},
        'formal_matrix_started': False})


def audit_blind(plan, output, result):
    starts = list((output/'requests').glob('*.request.json'))
    assert len(starts) == result['policy_calls'] == plan['bounds']['images']
    total = 0
    for path in starts:
        request = json.loads(path.read_text())
        assert request['prompt_ids'] and not request['images'] and not request['videos_present']
        assert all(request['sampling_params'].get(k) == v for k,v in
                   {'temperature':0., 'top_p':1., 'top_k':-1}.items())
        response = json.loads(path.with_name(request['id']+'.response.json').read_text())
        assert response['id'] == request['id']
        total += len(response['token_ids'])
    assert not list((output/'requests').glob('*.failure.json'))
    assert len(list((output/'requests').glob('*.response.json'))) == len(starts)
    assert total == result['generated_tokens'] <= plan['bounds']['maximum_generated_assistant_tokens']
    workers = list((output/'workers').glob('model-*.json'))
    assert len(workers) == 1
    worker = json.loads(workers[0].read_text())
    assert worker['status'] == 'PASS_LOADED_COORDINATES' and worker['seed'] == plan['seed']
    assert worker['plan_sha256'] == json.loads((output/'start.json').read_text())['plan_sha256']
    return {'policy_calls':len(starts), 'generated_tokens':total, 'image_absence_verified':True,
            'worker_receipt':worker, 'request_artifact_sha256':native.tree_hashes(output/'requests')}


async def evaluate(case, path):
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
        (output/name).mkdir()
    for name in case['source_sha256']:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    (output/'plan.json').write_bytes(path.read_bytes())
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path), VETO_NATIVE_EVALUATION_OUTPUT=str(output))
    native.write_new(output/'start.json', {'job_id':os.environ['SLURM_JOB_ID'], 'plan_sha256':file_sha256(path)})
    parquet = output/'pairs.parquet'
    native.write_new(output/'materialization.json', write_pair_parquet(Path(case['manifest']), parquet))
    try:
        dataset = native.dataset_for(case, parquet)
        if case['mode'] == 'blind':
            image_id = json.loads((Path(case['model']['path'])/'config.json').read_text())['image_token_id']
            for i in range(len(dataset)):
                row = dataset[i]
                images, videos = await dataset.process_vision_info(row['raw_prompt'],
                    image_patch_size=16, config=OmegaConf.create(case['config']['data']))
                assert not images and not videos
                prompt = dataset.processor.apply_chat_template(row['raw_prompt'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                tokens = dataset.processor(text=[prompt], return_tensors='pt')
                assert not bool((tokens['input_ids'] == image_id).any()) and 'pixel_values' not in tokens
        env = {k:v for k,v in os.environ.items() if k.startswith(('VETO_','HF_','TRANSFORMERS_','VERL_','VLLM_')) or
               k in ('PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','NO_PROXY','no_proxy')}
        ray.init(num_cpus=12, num_gpus=1, include_dashboard=False, object_store_memory=2*1024**3,
                 runtime_env={'env_vars':env, 'worker_process_setup_hook':'ours.training_bootstrap.prepare_worker'})
        manager = await ObservedVisualAgentManager.create(OmegaConf.create(case['config']))
        identity = EvaluationIdentity(case['model']['weights_sha256'], case['audit_data_sha256'],
            case['audit_data_sha256'], case['decode_sha256'], case['verifier_sha256'], 'development-initial-weights-'+case['mode'])
        receipt, result = await evaluate_pairs(manager, dataset, manifest_path=Path(case['manifest']),
            identity=identity, output=output/'evaluation', batch_size=8)
        observed = audit_blind(case, output, result) if case['mode'] == 'blind' else native.audit_requests(case, output, result)
        native.write_new(output/'result.json', {'status':'COMPLETE_DEVELOPMENT_VISUAL_CASE', 'role':'V',
            'mode':case['mode'], 'job_id':os.environ['SLURM_JOB_ID'], 'plan_sha256':file_sha256(path),
            'audit':asdict(receipt), 'paired_accuracy':receipt.paired_accuracy,
            'marginal_accuracy':receipt.marginal_accuracy, 'observed':observed,
            'limitations':case['limitations']})
    except BaseException as error:
        native.write_new(output/'failure.json', {'status':'INCOMPLETE', 'error_type':type(error).__name__, 'error':str(error)})
        raise
    finally:
        ray.shutdown()


def run_case(path, base_path):
    case = json.loads(path.read_text())
    base = native.check(base_path)
    assert case == configure(base, Path(case['manifest']), case['mode'], Path(case['output']))
    asyncio.run(evaluate(case, path))


def run(path):
    plan = json.loads(path.read_text())
    assert plan['kind'] == 'visual_development_screen' and tuple(plan['cases']) == MODES
    assert file_sha256(Path(plan['base_plan'])) == plan['base_plan_sha256']
    assert file_sha256(Path(plan['manifest'])) == plan['manifest_sha256']
    assert file_sha256(Path(plan['source_manifest'])) == plan['source_manifest_sha256']
    for name, sha in plan['source_sha256'].items():
        assert file_sha256(ROOT/name) == sha
    results = {}
    for mode, item in plan['cases'].items():
        assert file_sha256(Path(item['path'])) == item['sha256']
        with (Path(plan['output'])/f'{mode}.log').open('x') as stream:
            subprocess.run([sys.executable,'-m','ours.visual_development_screen','--phase','case',
                '--plan',item['path'],'--base-plan',plan['base_plan']], check=True,
                stdout=stream, stderr=subprocess.STDOUT)
        p = Path(plan['output'])/mode/'result.json'
        d = json.loads(p.read_text())
        assert d['status'] == 'COMPLETE_DEVELOPMENT_VISUAL_CASE' and d['role'] == 'V'
        assert d['job_id'] == os.environ['SLURM_JOB_ID'] and d['plan_sha256'] == item['sha256']
        results[mode] = {'paired_accuracy':d['paired_accuracy'], 'marginal_accuracy':d['marginal_accuracy'],
                         'policy_calls':d['observed']['policy_calls'], 'result_sha256':file_sha256(p)}
    native.write_new(Path(plan['output'])/'result.json', {'status':'COMPLETE_VISUAL_DEVELOPMENT_SCREEN',
        'job_id':os.environ['SLURM_JOB_ID'], 'plan_sha256':file_sha256(path), 'results':results,
        'optimizer_steps':0, 'formal_matrix_started':False, 'scientific_method_verified':False})
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('prepare','run','case'), required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--base-plan', type=Path)
    a = p.parse_args()
    if a.phase == 'prepare': prepare(a.plan.resolve(),a.output.resolve(),a.base_plan.resolve())
    elif a.phase == 'case': run_case(a.plan.resolve(),a.base_plan.resolve())
    else: run(a.plan.resolve())
