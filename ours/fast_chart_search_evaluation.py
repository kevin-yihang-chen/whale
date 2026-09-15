"""Adapt native H/C evaluation and certification to the frozen shared chart scorer."""
import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time

from . import visual_search_evaluation as original
from . import visual_search_bridge as bridge
from .chart_answer_protocol import PROTOCOL,verify
from .evidence import EvaluationIdentity
from .fast_chart_protocol import OUTPUT,storage_check
from .fast_chart_training import read
from .local_completion import checkpoint_manifest
from .visual_harness import configured_visual_harness
from .visual_native_evaluation import pair_inputs,verifier_identity
from .visual_task import file_sha256


@contextmanager
def shared_scoring():
    """Serial process adapter; only the host verifier changes from legacy A/B."""
    previous=(original.binary_answer_verifier,bridge.binary_answer_verifier)
    original.binary_answer_verifier=bridge.binary_answer_verifier=verify
    try:yield
    finally:original.binary_answer_verifier,bridge.binary_answer_verifier=previous


@contextmanager
def timed_measurements():
    h_original,c_original=original.evaluate_h,original.evaluate_pairs
    async def h(*args,**kwargs):
        start=time.monotonic();result=await h_original(*args,**kwargs)
        output=Path(args[3] if len(args)>3 else kwargs['output'])
        original.native.write_new(output/'timing.json',{'role':'H','elapsed_seconds':time.monotonic()-start})
        return result
    async def c(*args,**kwargs):
        start=time.monotonic();result=await c_original(*args,**kwargs)
        original.native.write_new(Path(kwargs['output'])/'timing.json',{'role':'C','elapsed_seconds':time.monotonic()-start})
        return result
    original.evaluate_h,original.evaluate_pairs=h,c
    try:yield
    finally:original.evaluate_h,original.evaluate_pairs=h_original,c_original


def prepare(path,output,reference,harness,*,seed,name,phase):
    path,output,reference,harness=map(lambda p:Path(p).resolve(),(path,output,reference,harness))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):raise ValueError('Require fresh search evaluation paths')
    if seed not in (42,43,44) or name not in ('h0','h1','h2','h3'):raise ValueError('Unregistered seed/candidate slot')
    ref=read(reference)
    if ref['kind']!='compact_chart_pair_evaluation' or ref['partition']!='V':raise ValueError('Require a real compact V reference plan')
    result=read(Path(ref['output'])/'result.json')
    if result['status']!='COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or result['plan_sha256']!=file_sha256(reference):
        raise ValueError('Reference weights have not completed real visual evaluation')
    plan=deepcopy(ref);parts=read(OUTPUT/'native-data/result.json')['partitions']
    plan.update(kind='compact_chart_search_evaluation',output=str(output),candidate=name,phase=phase,role='H+C',
        partitions={r:parts[r] for r in ('H','C')},with_audit=True,repeatability_check=False,batch_size=8,
        reference_plan=str(reference),reference_plan_sha256=file_sha256(reference),
        materialization=str(OUTPUT/'native-data/result.json'),materialization_sha256=file_sha256(OUTPUT/'native-data/result.json'))
    c,_,_=pair_inputs(parts['C']['manifest'])
    plan.update(manifest=parts['C']['manifest'],manifest_sha256=parts['C']['manifest_sha256'],audit_data_sha256=c['audit_data_sha256'])
    cfg=plan['config'];cfg['data']['visual_harness_path']=str(harness);cfg['data']['cache_dir']=str(output/'dataset-cache')
    cfg['actor_rollout_ref']['rollout']['engine_kwargs']['vllm']['seed']=seed
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']=output.name;cfg['trainer']['experiment_name']=output.name
    from omegaconf import OmegaConf
    configured_visual_harness(OmegaConf.create(cfg['data'])).unchanged()
    plan['harness_sha256']=file_sha256(harness);plan['seed']=seed
    plan['decode_sha256']=original.decode_identity(cfg,plan['model']['assets'])
    plan['identity']=asdict(EvaluationIdentity(plan['model']['weights_sha256'],parts['H']['manifest_sha256'],
        c['audit_data_sha256'],plan['decode_sha256'],verifier_identity(PROTOCOL),phase))
    for name_ in (*original.EXTRA,*bridge.SOURCES,'ours/fast_chart_search_evaluation.py','ours/run_fast_chart_search_evaluation.sh',str(harness)):
        plan['source_sha256'][name_]=file_sha256(original.ROOT/name_)
    plan['bounds']={'gpus':1,'cpus':12,'time_limit_seconds':3600,'images':256,'maximum_generation_calls':768,
        'maximum_generated_assistant_tokens':256*1024,'api_calls':0,'new_model_checkpoints':0}
    plan['limitations']=['H and C are optimization data; first complete measurement only.',
        'Shared host parser and verifier; no candidate scoring modifications.',
        'Batched greedy inference is not assumed bitwise deterministic.']
    storage_check(12);original.native.write_new(path,plan);return plan


def check(path):
    plan=read(path)
    if plan['kind']!='compact_chart_search_evaluation' or Path(plan['output']).exists():raise ValueError('Wrong or executed search plan')
    for name,digest in plan['source_sha256'].items():
        if file_sha256(original.ROOT/name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    for name,key in (('reference_plan','reference_plan_sha256'),('materialization','materialization_sha256')):
        if file_sha256(Path(plan[name]))!=plan[key]:raise ValueError('Evaluation input changed')
    for p in plan['partitions'].values():
        for key in ('manifest','parquet'):
            if file_sha256(Path(p[key]))!=p[key+'_sha256']:raise ValueError('Search dataset identity changed')
    if checkpoint_manifest(Path(plan['model']['path']))!=plan['model'] or original.decode_identity(plan['config'],plan['model']['assets'])!=plan['decode_sha256']:
        raise ValueError('Search model or decoding changed')
    if EvaluationIdentity(**plan['identity']).verifier_sha256!=verifier_identity(PROTOCOL):raise ValueError('Shared scorer changed')
    storage_check(12);return plan


def certify(plan_path,allocation_path):
    if read(plan_path)['kind']!='compact_chart_search_evaluation':raise ValueError('Historical search protocol is not interchangeable')
    with shared_scoring():return bridge.certify(plan_path,allocation_path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','check','run'))
    for key in ('plan','output','reference','harness'):p.add_argument('--'+key,type=Path,required=key=='plan')
    p.add_argument('--seed',type=int,choices=(42,43,44));p.add_argument('--name');p.add_argument('--phase')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.plan,a.output,a.reference,a.harness,seed=a.seed,name=a.name,phase=a.phase)
    elif a.action=='check':check(a.plan)
    else:
        plan=check(a.plan)
        with shared_scoring(),timed_measurements():asyncio.run(original.run(plan,a.plan.resolve()))
