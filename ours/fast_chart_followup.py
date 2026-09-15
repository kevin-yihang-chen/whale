"""Verify the complete E4 parameter change, export, then measure the shared V set."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from .fast_chart_protocol import ROOT, OUTPUT, storage_check
from .fast_chart_training import read, checkpoint, MODEL
from .visual_task import file_sha256
from .native_visual_service import write_new
from .local_completion import checkpoint_manifest

SOURCES=('ours/fast_chart_followup.py','ours/run_fast_chart_followup.sh',
    'ours/fast_chart_backbone_audit.py','ours/visual_pixel_bootstrap.py',
    'ours/canonical_native_export.py','ours/controlled_checkpoint_export.py',
    'ours/verify_alternation_transition.py','ours/probe_updated_vllm.py',
    'ours/verify_visual_resumed_parameters.py','ours/verify_native_transition.py')


def prepare(path,output,training_plan):
    from .fast_chart_evaluation import EXTRA
    from .native_visual_service import SOURCES as native_sources
    path,output,training_plan=map(lambda p:Path(p).resolve(),(path,output,training_plan))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require fresh compact followup paths')
    training=read(training_plan); result_path=Path(training['output'])/'execution-result.json'
    result=read(result_path)
    if result['status']!='COMPLETE_COMPACT_NATIVE_STAGE' or result['plan_sha256']!=file_sha256(training_plan):
        raise ValueError('Native training has not completed its execution checks')
    parent=checkpoint(Path(training['output'])/f"checkpoints/global_step_{4*training['stage']}",4*training['stage'])
    original=read(ROOT/'results/visual-pixel-followup-plan-20260910-v1.json')
    sources=set(SOURCES)|set(native_sources)|set(EXTRA)
    plan={'kind':'compact_chart_native_followup','output':str(output),'target':str(output/'hf-canonical'),
        'training_plan':str(training_plan),'training_plan_sha256':file_sha256(training_plan),
        'training_result_sha256':file_sha256(result_path),'checkpoint':parent,
        'base':checkpoint_manifest(MODEL),'aliases':original['aliases'],'active_tensors':original['active_tensors'],
        'harness':training['harness'],'harness_sha256':training['harness_sha256'],'seed':training['seed'],
        'manifest':str(OUTPUT/'dataset/V/manifest.json'),
        'manifest_sha256':file_sha256(OUTPUT/'dataset/V/manifest.json'),
        'source_sha256':{name:file_sha256(ROOT/name) for name in sorted(sources)},
        'bounds':{'gpus':1,'cpus':12,'time_limit_seconds':3600,'images':512,'new_serving_exports':1,'api_calls':0},
        'limitations':['V is used for configuration calibration, not an independent test result.',
            'No method benefit is inferred from a nonzero parameter update.']}
    storage_check(25);write_new(path,plan)
    return plan


def check(path):
    plan=read(path)
    if plan['kind']!='compact_chart_native_followup' or Path(plan['output']).exists():
        raise ValueError('Wrong or previously executed followup')
    for name,digest in plan['source_sha256'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError(f'Frozen followup source changed: {name}')
    if file_sha256(Path(plan['training_plan']))!=plan['training_plan_sha256'] or checkpoint_manifest(MODEL)!=plan['base']:
        raise ValueError('Training or base model identity changed')
    training=read(plan['training_plan'])
    if file_sha256(Path(training['output'])/'execution-result.json')!=plan['training_result_sha256']:
        raise ValueError('Training execution receipt changed')
    if checkpoint(plan['checkpoint']['directory'],plan['checkpoint']['step'])!=plan['checkpoint']:
        raise ValueError('Native checkpoint changed')
    if file_sha256(Path(plan['manifest']))!=plan['manifest_sha256'] or file_sha256(Path(plan['harness']))!=plan['harness_sha256']:
        raise ValueError('Shared validation data/harness changed')
    storage_check(25);return plan


def run(path):
    from .controlled_checkpoint_export import restore_generation_asset
    from .verify_alternation_transition import compare_active_parameters
    from .probe_updated_vllm import inference_contract,changed_embedding_coordinates
    from .fast_chart_evaluation import prepare as prepare_evaluation, check as check_evaluation, evaluate
    plan=check(path);out=Path(plan['output']);out.mkdir()
    (out/'plan.json').write_bytes(path.read_bytes())
    native=Path(plan['checkpoint']['directory'])/'actor';target=Path(plan['target'])
    try:
        from .fast_chart_backbone_audit import audit as audit_backbone
        audit_backbone(plan['training_plan'],out/'image-consumption-audit.json')
        subprocess.run([sys.executable,'-m','ours.canonical_native_export','--base',str(MODEL),
            '--native',str(native),'--target',str(target),'--report',str(out/'serialization.json')],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
        asset=restore_generation_asset(MODEL,target,out/'generated-generation_config.json')
        contract=inference_contract(MODEL,target)
        comparison=compare_active_parameters(MODEL,native,target,aliases=plan['aliases'],tensor_count=plan['active_tensors'])
        training=read(plan['training_plan'])
        if training['stage']==2:
            import torch
            from .verify_visual_resumed_parameters import compare_states
            previous=Path(training['resume_checkpoint']['directory'])/'actor/model_world_size_1_rank_0.pt'
            current=native/'model_world_size_1_rank_0.pt'
            states=[torch.load(p,map_location='cpu',mmap=True,weights_only=True) for p in (previous,current)]
            comparison['stage2_native_fp32_change']=compare_states(*states,plan['aliases'],plan['active_tensors'])
            comparison['stage2_parent_sha256']=file_sha256(previous)
        if checkpoint(plan['checkpoint']['directory'],plan['checkpoint']['step'])!=plan['checkpoint']:
            raise ValueError('Export changed native restoration artifacts')
        transition={'status':'PASS','plan_sha256':file_sha256(path),'exported':checkpoint_manifest(target),
            'image_consumption_audit_sha256':file_sha256(out/'image-consumption-audit.json'),
            'native_checkpoint':plan['checkpoint'],'inference_contract':contract,'generation_asset_restoration':asset,
            'resume_artifacts_unchanged':True,**comparison}
        write_new(out/'transition.json',transition)
        template=out/'evaluation-template-plan.json'
        case_plan=prepare_evaluation(template,manifest_path=plan['manifest'],harness=plan['harness'],model=target,
            output=out/'development',seed=plan['seed'],phase='compact-post-training-development')
        embedding=next(r for r in comparison['exported_bf16']['tensors'] if r['name']=='model.language_model.embed_tokens.weight')
        if embedding['changed_elements']>=8:
            case_plan['worker_coordinates']=[{**r,'expected':r['updated']} for r in changed_embedding_coordinates(MODEL,target)]
            case_plan['coordinates_distinguish_initial_model']=True
        else:case_plan['coordinates_distinguish_initial_model']=False
        case_plan['template_plan_sha256']=file_sha256(template)
        case=out/'evaluation-plan.json';write_new(case,case_plan)
        asyncio.run(evaluate(check_evaluation(case),case))
        score=read(out/'development/result.json')
        write_new(out/'result.json',{'status':'COMPLETE_COMPACT_NATIVE_FOLLOWUP',
            'plan_sha256':file_sha256(path),'job_id':os.environ['SLURM_JOB_ID'],
            'transition_sha256':file_sha256(out/'transition.json'),
            'evaluation_plan':str(case),'evaluation_result_sha256':file_sha256(out/'development/result.json'),
            'paired_accuracy':score['paired_accuracy'],'marginal_accuracy':score['marginal_accuracy'],
            'scientific_method_verified':False})
    except BaseException as exc:
        write_new(out/'failure.json',{'status':'INCOMPLETE','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','check','run'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--training-plan',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.plan,a.output,a.training_plan)
    elif a.action=='check':check(a.plan)
    else:run(a.plan.resolve())
