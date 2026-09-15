"""Export a completed controlled E4 checkpoint without altering resume state.

For every active parameter i, certify theta_export[i] = BF16(theta_native[i]).
Report native and BF16 deltas against the common theta0 separately, including
unchanged outcomes. This is checkpoint provenance, not evidence of task gains.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .native_search import write_json
from .probe_updated_vllm import inference_contract
from .verify_alternation_transition import compare_active_parameters
from .visual_task import file_sha256

SOURCES = ('ours/controlled_checkpoint_export.py','ours/run_controlled_checkpoint_export.sh',
    'ours/canonical_native_export.py','ours/verify_alternation_transition.py',
    'ours/verify_native_transition.py','ours/probe_updated_vllm.py','ours/local_completion.py',
    'ours/audit_native_training_batch.py','ours/visual_task.py','ours/native_search.py',
    'ours/training_bootstrap.py','ours/controlled_training_result.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/base_model_merger.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/fsdp_model_merger.py')


def completed_inputs(training_path,result_path,step):
    from .controlled_pilot import check
    require(type(step) is int and step in (1,2),'Expected a saved controlled batch step')
    training = check(training_path)
    result = json.loads(result_path.read_text())
    require(result['kind']=='controlled_weight_only_training_result' and
            result['status']=='COMPLETED_WITH_PREFLIGHT_SCHEDULE_DEVIATION' and
            result['condition']==training['condition']=='weight_only' and
            result['seed']==training['seed'] and result['plan_sha256']==file_sha256(training_path),
            'Export requires the audited controlled weight-only result')
    require(result['allocation']['state']=='COMPLETED' and result['allocation']['exit_code']=='0:0' and
            not result['original_preflight_batch_ids_match'],'Missing terminal evidence or deviation disclosure')
    require([c['step'] for c in result['checkpoints']]==[1,2] and
            result['optimizer_steps']==sum(b['optimizer_steps'] for b in result['batches']),
            'Incomplete checkpoint or update accounting')
    audit = json.loads(Path(result['batch_audit_path']).read_text())
    require(file_sha256(Path(result['batch_audit_path']))==result['batch_audit_sha256'] and
            audit['status']=='PASS_NATIVE_EXECUTION_WITH_PREFLIGHT_DEVIATION' and
            audit['plan_sha256']==result['plan_sha256'],'Wrong completed batch audit')
    sources = dict(training['source_sha256'])
    sources.update(result['artifact_sha256'])
    sources.update(audit['source_sha256'])
    sources.update(audit['native_configuration_audit']['audit_source_sha256'])
    sources.update(audit['native_configuration_audit']['artifact_sha256'])
    for checkpoint in result['checkpoints']:
        for name,digest in checkpoint['artifact_sha256'].items():
            sources[str(Path(checkpoint['directory'])/name)] = digest
    for name,digest in sources.items():
        require(file_sha256(Path(name))==digest,f'Changed completed training evidence: {name}')
    require(file_sha256(Path('ours/controlled_training_result.py'))==result['result_source_sha256'],
            'Changed training result finalizer')
    initialization = json.loads(Path(training['initialization_report']).read_text())
    require(initialization['status']=='PASS' and initialization['optimizer_steps']==0 and
            initialization['audit']['exact_source_bf16_cast'] and
            initialization['exported']==training['initial_manifest'],'Wrong common untrained initialization')
    selected = result['checkpoints'][step-1]
    return training,result,initialization,selected,sources


def prepare(path,training_path,result_path,step):
    require(not path.exists(),'Preserve existing controlled export plan')
    training,result,init,selected,sources = completed_inputs(training_path,result_path,step)
    native = Path(selected['directory'])/'actor'
    stem = f"controlled-weight-only-seed{result['seed']}-{result['job_id']}-step{step}"
    target = native.parent.parent/f'hf-step-{step}-canonical'
    report = Path(f'results/{stem}-transition.json')
    require(not target.exists() and not report.exists(),'Preserve existing export artifacts')
    from .canonical_native_export import validate_inference_config
    differences = validate_inference_config(Path(training['model']),native/'huggingface')
    for name in (*SOURCES,str(training_path),str(result_path),training['initialization_report']):
        sources[name] = file_sha256(Path(name))
    plan = {'kind':'controlled_weight_only_export_plan','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_plan':str(training_path),'training_result':str(result_path),'source_job_id':result['job_id'],
        'seed':result['seed'],'step':step,'base':init['exported'],'native':str(native),
        'native_checkpoint_sha256':selected['artifact_sha256']['actor/model_world_size_1_rank_0.pt'],
        'target':str(target),'transition_report':str(report),
        'verified_aliases':init['audit']['verified_aliases'],'active_tensors':init['audit']['comparison']['tensor_count'],
        'source_sha256':sources,'versions':{n:version(n) for n in ('torch','transformers','safetensors','numpy')},
        'reviewed_native_config_differences':differences,
        'optimizer_steps_through_checkpoint':sum(b['optimizer_steps'] for b in result['batches'] if b['step']<=step),
        'training_deviation':result['deviation'],'original_preflight_batch_ids_match':False,
        'resources':{'gpus':1,'gpu_type':'rtx_4090','cpus':4,'ram_gib':64,'time_limit_seconds':1200,'max_gpu_hours':1/3},
        'new_model_calls':0,'limitations':['The complete native model, extra state and data.pt are retained for resume.',
            'CPU export under the minimum GPU allocation; allocated GPU time is charged.',
            'Parameter changes are not performance gains; unchanged BF16 output is a valid audited outcome.',
            'This weight-only checkpoint is not certified as a shared joint-arm training prefix.',
            'Actual serving identity and heldout evaluation remain separate checks.']}
    write_json(path,plan)
    return plan


def check(path):
    plan = json.loads(path.read_text())
    require(plan['kind']=='controlled_weight_only_export_plan','Wrong controlled export plan')
    require(set(SOURCES)<=set(plan['source_sha256']),'Missing export implementation binding')
    require({n:version(n) for n in plan['versions']}==plan['versions'],'Changed export runtime')
    for name,digest in plan['source_sha256'].items():
        require(file_sha256(Path(name))==digest,f'Frozen export input changed: {name}')
    require(checkpoint_manifest(Path(plan['base']['path']))==plan['base'],'Changed common initialization')
    require(file_sha256(Path(plan['native'])/'model_world_size_1_rank_0.pt')==plan['native_checkpoint_sha256'],
            'Changed selected native checkpoint')
    require(not Path(plan['target']).exists() and not Path(plan['transition_report']).exists(),'Export already attempted')
    return plan


def restore_generation_asset(base,target,archive):
    """Preserve source asset absence after the reviewed Transformers save path."""
    from transformers import AutoConfig, GenerationConfig
    require(not (base/'generation_config.json').exists(),'This common initialization has no generation asset')
    generated = target/'generation_config.json'
    if not generated.exists():return {'generated_asset_present':False}
    a = GenerationConfig.from_model_config(AutoConfig.from_pretrained(base,local_files_only=True)).to_dict()
    b = GenerationConfig.from_pretrained(target,local_files_only=True).to_dict()
    differences = {k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    reviewed = {'output_hidden_states':[False,None],'output_attentions':[False,None]}
    require(all(k in reviewed and v==reviewed[k] for k,v in differences.items()),
            'Unreviewed generated inference configuration difference')
    require(not archive.exists(),'Preserve generated configuration evidence')
    digest = file_sha256(generated)
    shutil.move(str(generated),archive)
    return {'generated_asset_present':True,'archived_path':str(archive),'sha256':digest,
            'reviewed_difference':differences,'source_asset_absence_restored':True}


def run(path,plan):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Expected the reviewed CPU Slurm export wrapper')
    job = os.environ['SLURM_JOB_ID'];root = Path(f'results/controlled-checkpoint-export-{job}')
    root.mkdir(exist_ok=False)
    start = {'job_id':job,'plan_sha256':file_sha256(path),'source_job_id':plan['source_job_id'],
        'native_checkpoint_sha256':plan['native_checkpoint_sha256'],'new_model_calls':0}
    write_json(root/'start.json',start)
    base,native,target = Path(plan['base']['path']),Path(plan['native']),Path(plan['target'])
    subprocess.run([sys.executable,'-m','ours.canonical_native_export','--base',str(base),
        '--native',str(native),'--target',str(target),'--report',str(root/'serialization.json')],check=True)
    asset = restore_generation_asset(base,target,root/'generated-generation_config.json')
    contract = inference_contract(base,target)
    comparison = compare_active_parameters(base,native,target,aliases=plan['verified_aliases'],tensor_count=plan['active_tensors'])
    require(file_sha256(native/'model_world_size_1_rank_0.pt')==plan['native_checkpoint_sha256'],
            'Native checkpoint changed during export')
    exported = checkpoint_manifest(target)
    require(checkpoint_manifest(base)==plan['base'],'Common initialization changed during export')
    report = {'kind':'controlled_weight_only_checkpoint_transition','status':'PASS',**start,
        'step':plan['step'],'base':plan['base'],'native_checkpoint':str(native/'model_world_size_1_rank_0.pt'),
        'exported':exported,'verified_aliases':plan['verified_aliases'],**comparison,
        'generation_asset_restoration':asset,'inference_contract':contract,
        'serialization_report_sha256':file_sha256(root/'serialization.json'),
        'training_deviation':plan['training_deviation'],'original_preflight_batch_ids_match':False,
        'optimizer_steps_through_checkpoint':plan['optimizer_steps_through_checkpoint'],'limitations':plan['limitations']}
    write_json(Path(plan['transition_report']),report)
    write_json(root/'measurement.json',{'status':'PASS',**start,'transition_report':plan['transition_report'],
        'transition_report_sha256':file_sha256(Path(plan['transition_report']))})
    print(json.dumps({'status':'PASS',**start,**{k:{n:v for n,v in comparison[k].items() if n!='tensors'}
                      for k in ('native_fp32','exported_bf16')}}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('prepare','check','run'),required=True)
    for name in ('plan','training-plan','result'):parser.add_argument('--'+name,type=Path,required=name=='plan')
    parser.add_argument('--step',type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    if args.phase=='prepare':
        require(args.training_plan is not None and args.result is not None,'Missing training evidence')
        prepare(args.plan,args.training_plan,args.result,args.step)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    else:
        plan = check(args.plan)
        if args.phase=='run':run(args.plan,plan)
        else:print(json.dumps({'status':'PASS','plan_sha256':file_sha256(args.plan)}),flush=True)
