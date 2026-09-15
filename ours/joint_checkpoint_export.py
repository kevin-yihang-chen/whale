"""Canonical joint checkpoint handoff after native phase one or phase two.

Require a completed independent joint phase and preserve its native resume
state. Measure every active native/exported parameter against common theta0,
including unchanged outcomes, and retain separate Slurm export completion.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .audit_native_training_batch import require
from .canonical_native_export import validate_inference_config
from .controlled_checkpoint_export import restore_generation_asset
from .joint_training_result import complete
from .local_completion import checkpoint_manifest
from .native_search import write_json
from .probe_updated_vllm import inference_contract
from .verify_alternation_transition import compare_active_parameters
from .visual_task import file_sha256

SOURCES=('ours/joint_checkpoint_export.py','ours/run_joint_checkpoint_export.sh',
    'ours/joint_training_result.py','ours/canonical_native_export.py','ours/controlled_checkpoint_export.py',
    'ours/verify_alternation_transition.py','ours/verify_native_transition.py','ours/probe_updated_vllm.py',
    'ours/local_completion.py','ours/training_bootstrap.py','ours/joint_training_phase2_result.py',
    'ours/joint_resume_receipts.py','ours/native_resume_observation.py','ours/joint_training_phase2.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/base_model_merger.py',
    'upstream/WHALE/domains/chess_puzzles/verl/model_merger/fsdp_model_merger.py')


CONDITIONS=('whale','whale_fst')
EXPORT_WRAPPER='ours/run_joint_checkpoint_export.sh'
EXPORT_DIRECTORY='joint-checkpoint-export'


def optimizer_steps(result,step):
    return result['optimizer_steps'] if step==1 else result['total_joint_optimizer_steps']


def limitations():
    return ['Independently completed joint checkpoint, not a pooled weight-only prefix.',
        'Native model/extra/data.pt remain unchanged; the original phase identity is retained.',
        'CPU export under the minimum GPU allocation; all allocated GPU time is charged.',
        'Unchanged native or BF16 parameters are valid recorded outcomes, not a reason to resample.',
        'Export identity does not prove actual serving or native GPU checkpoint restoration.',
        'Optimization and heldout task measurements remain separate stages.']


def artifact_kind(step,suffix):
    require(type(step) is int and step in (1,2),'Invalid joint export phase')
    return f'controlled_joint_phase{step}_{suffix}'


def completed_inputs(training_path,result_path):
    result=json.loads(result_path.read_text())
    step=result.get('phase')
    require(type(step) is int and step in (1,2) and result.get('kind')==artifact_kind(step,'training_result') and
            result.get('status')=='COMPLETED_NATIVE_PHASE' and result['condition'] in CONDITIONS and
            result['original_preflight_batch_ids_match'],'Export requires an independently completed joint phase')
    if step==1:finalize=complete
    else:
        from .joint_training_phase2_result import complete as finalize
    verified=finalize(training_path,Path(result['batch_audit_path']),Path(result['slurm_terminal_path']),Path(result['training_log_path']))
    require(result==verified,'Completed joint result no longer matches its full evidence')
    training=json.loads(training_path.read_text());initial=json.loads(Path(training['initialization_report']).read_text())
    require(initial['exported']==training['initial_manifest'],'Changed common initialization')
    require(len(result['checkpoints'])==1 and result['checkpoints'][0]['step']==step,'Wrong joint phase checkpoint')
    selected=result['checkpoints'][0];audit=json.loads(Path(result['batch_audit_path']).read_text())
    sources={**training['source_sha256'],**result['artifact_sha256'],**result['finalizer_source_sha256'],
             **audit['artifact_sha256'],**audit['audit_source_sha256']}
    sources.update({str(Path(selected['directory'])/name):digest for name,digest in selected['artifact_sha256'].items()})
    return training,result,initial,selected,sources


def prepare(path,training_path,result_path):
    require(not path.exists(),'Preserve existing joint export plan')
    training,result,initial,selected,sources=completed_inputs(training_path,result_path)
    step=selected['step'];native=Path(selected['directory'])/'actor';target=native.parent.parent/f'hf-step-{step}-canonical'
    stem=f"controlled-{result['condition']}-seed{result['seed']}-{result['job_id']}-step{step}"
    report=Path(f'results/{stem}-transition.json')
    require(not target.exists() and not report.exists(),'Preserve existing export artifacts')
    differences=validate_inference_config(Path(training['model']),native/'huggingface')
    for name in (*SOURCES,str(training_path),str(result_path),training['initialization_report']):sources[name]=file_sha256(Path(name))
    plan={'kind':artifact_kind(step,'export_plan'),'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_plan':str(training_path),'training_result':str(result_path),'source_job_id':result['job_id'],
        'condition':result['condition'],'seed':result['seed'],'step':step,'base':initial['exported'],
        'native':str(native),'native_checkpoint_sha256':selected['artifact_sha256']['actor/model_world_size_1_rank_0.pt'],
        'resume_checkpoint':selected,'target':str(target),'transition_report':str(report),
        'verified_aliases':initial['audit']['verified_aliases'],'active_tensors':initial['audit']['comparison']['tensor_count'],
        'source_sha256':sources,'versions':{n:version(n) for n in ('torch','transformers','safetensors','numpy')},
        'reviewed_native_config_differences':differences,'optimizer_steps_through_checkpoint':optimizer_steps(result,step),
        'resources':{'gpus':1,'gpu_type':'rtx_4090','cpus':4,'ram_gib':64,'time_limit_seconds':1200,'max_gpu_hours':1/3},
        'new_model_calls':0,'limitations':limitations()}
    write_json(path,plan);return plan


def check(path,*,fresh=True):
    plan=json.loads(path.read_text())
    require(plan['kind']==artifact_kind(plan['step'],'export_plan') and plan['condition'] in CONDITIONS and
            plan['step'] in (1,2) and plan['new_model_calls']==0,'Wrong joint export scope')
    require(set(SOURCES)<=set(plan['source_sha256']),'Missing joint export source binding')
    require({n:version(n) for n in plan['versions']}==plan['versions'],'Changed export runtime')
    for name,digest in plan['source_sha256'].items():require(file_sha256(Path(name))==digest,f'Changed export input: {name}')
    require(checkpoint_manifest(Path(plan['base']['path']))==plan['base'],'Changed common theta0')
    selected=plan['resume_checkpoint']
    require(selected['step']==plan['step'] and Path(selected['directory']).name==f"global_step_{plan['step']}" and
            Path(selected['directory']).resolve()==Path(plan['native']).parent.resolve(),
            'Different export and resume checkpoints')
    for name,digest in selected['artifact_sha256'].items():
        require(file_sha256(Path(selected['directory'])/name)==digest,f'Changed native resume artifact: {name}')
    require(plan['native_checkpoint_sha256']==selected['artifact_sha256']['actor/model_world_size_1_rank_0.pt'],
            'Wrong selected model identity')
    if fresh:require(not Path(plan['target']).exists() and not Path(plan['transition_report']).exists(),'Export already attempted')
    return plan


def run(path,plan):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Expected CPU Slurm export wrapper')
    job=os.environ['SLURM_JOB_ID'];root=Path(f'results/{EXPORT_DIRECTORY}-{job}');root.mkdir(exist_ok=False)
    start={'job_id':job,'plan_sha256':file_sha256(path),'source_job_id':plan['source_job_id'],
        'condition':plan['condition'],'seed':plan['seed'],'step':plan['step'],
        'native_checkpoint_sha256':plan['native_checkpoint_sha256'],'new_model_calls':0}
    write_json(root/'start.json',start)
    base,native,target=Path(plan['base']['path']),Path(plan['native']),Path(plan['target'])
    subprocess.run([sys.executable,'-m','ours.canonical_native_export','--base',str(base),'--native',str(native),
                    '--target',str(target),'--report',str(root/'serialization.json')],check=True)
    asset=restore_generation_asset(base,target,root/'generated-generation_config.json')
    contract=inference_contract(base,target)
    comparison=compare_active_parameters(base,native,target,aliases=plan['verified_aliases'],tensor_count=plan['active_tensors'])
    require(checkpoint_manifest(base)==plan['base'],'Common theta0 changed during export')
    for name,digest in plan['resume_checkpoint']['artifact_sha256'].items():
        require(file_sha256(Path(plan['resume_checkpoint']['directory'])/name)==digest,f'Export changed resume state: {name}')
    report={'kind':artifact_kind(plan['step'],'checkpoint_transition'),'status':'PASS',**start,
        'base':plan['base'],'exported':checkpoint_manifest(target),'resume_checkpoint':plan['resume_checkpoint'],
        'verified_aliases':plan['verified_aliases'],**comparison,'generation_asset_restoration':asset,
        'inference_contract':contract,'serialization_report_sha256':file_sha256(root/'serialization.json'),
        'optimizer_steps_through_checkpoint':plan['optimizer_steps_through_checkpoint'],
        'original_preflight_batch_ids_match':True,'limitations':plan['limitations']}
    write_json(Path(plan['transition_report']),report)
    write_json(root/'measurement.json',{'status':'PASS',**start,'transition_report':plan['transition_report'],
        'transition_report_sha256':file_sha256(Path(plan['transition_report']))})
    print(json.dumps({'status':'PASS',**start,**{k:{n:v for n,v in comparison[k].items() if n!='tensors'}
                      for k in ('native_fp32','exported_bf16')}}),flush=True)


def close(path,slurm_path):
    plan=check(path,fresh=False);report_path=Path(plan['transition_report']);report=json.loads(report_path.read_text())
    require(report['kind']==artifact_kind(plan['step'],'checkpoint_transition') and report['status']=='PASS' and
            report['plan_sha256']==file_sha256(path) and report['source_job_id']==plan['source_job_id'] and
            report['condition']==plan['condition'] and report['seed']==plan['seed'] and report['step']==plan['step'] and
            report['native_checkpoint_sha256']==plan['native_checkpoint_sha256'] and
            report['resume_checkpoint']==plan['resume_checkpoint'] and report['base']==plan['base'] and
            report['verified_aliases']==plan['verified_aliases'] and report['export_exact_native_bf16_cast'] and
            report['optimizer_steps_through_checkpoint']==plan['optimizer_steps_through_checkpoint'],
            'Invalid canonical joint transition')
    require(checkpoint_manifest(Path(plan['target']))==report['exported'],'Changed exported model or assets')
    require(all(report[key]['tensor_count']==plan['active_tensors'] for key in ('native_fp32','exported_bf16')),
            'Incomplete active-parameter comparison')
    root=Path(f"results/{EXPORT_DIRECTORY}-{report['job_id']}")
    measurement=json.loads((root/'measurement.json').read_text())
    require(measurement['status']=='PASS' and measurement['plan_sha256']==file_sha256(path) and
            measurement['transition_report_sha256']==file_sha256(report_path) and
            report['serialization_report_sha256']==file_sha256(root/'serialization.json'),'Changed export completion evidence')
    raw=slurm_path.read_text();fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
    require(fields['JobId']==report['job_id'] and fields['JobState']=='COMPLETED' and fields['ExitCode']=='0:0',
            'Export job did not complete successfully')
    allocated=dict(item.split('=',1) for item in fields['AllocTRES'].split(','))
    require(allocated.get('gres/gpu')==allocated.get('gres/gpu:rtx_4090')=='1' and
            Path(fields['Command']).resolve()==Path(EXPORT_WRAPPER).resolve(),
            'Wrong export allocation or entrypoint')
    days,_,clock=fields['RunTime'].rpartition('-');h,m,s=map(int,clock.split(':'));seconds=int(days or 0)*86400+3600*h+60*m+s
    require(0<=seconds<=plan['resources']['time_limit_seconds'],'Export exceeded its allocation budget')
    return {'kind':artifact_kind(plan['step'],'export_result'),'status':'COMPLETED_CANONICAL_EXPORT',
        'job_id':report['job_id'],'source_job_id':plan['source_job_id'],'condition':plan['condition'],'seed':plan['seed'],
        'plan_sha256':file_sha256(path),'slurm_terminal_path':str(slurm_path.resolve()),'transition_report':str(report_path.resolve()),
        'transition_report_sha256':file_sha256(report_path),'exported':report['exported'],
        'resume_checkpoint':plan['resume_checkpoint'],'step':plan['step'],
        'optimizer_steps_through_checkpoint':plan['optimizer_steps_through_checkpoint'],
        'allocation':{'seconds':seconds,'gpu_hours':seconds/3600,'alloc_tres':fields['AllocTRES'],'state':'COMPLETED','exit_code':'0:0'},
        'artifact_sha256':{str(p.resolve()):file_sha256(p) for p in (path,slurm_path,report_path,root/'start.json',root/'serialization.json',root/'measurement.json')},
        'new_model_calls_by_finalizer':0,'limitations':plan['limitations']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','check','run','close'),required=True)
    for name in ('plan','training-plan','result','slurm','output'):parser.add_argument('--'+name,type=Path,required=name=='plan')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(args.training_plan is not None and args.result is not None,'Missing completed training evidence')
        prepare(args.plan,args.training_plan,args.result)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    elif args.phase=='close':
        require(args.slurm is not None and args.output is not None and not args.output.exists(),'Missing terminal evidence or occupied output')
        result=close(args.plan,args.slurm);write_json(args.output,result);print(json.dumps(result),flush=True)
    else:
        plan=check(args.plan)
        if args.phase=='run':run(args.plan,plan)
        else:print(json.dumps({'status':'PASS','plan_sha256':file_sha256(args.plan)}),flush=True)
