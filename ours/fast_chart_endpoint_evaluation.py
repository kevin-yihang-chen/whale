"""Freeze a budget-defined endpoint before opening T/R or the original ChartQA subset."""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .fast_chart_protocol import ROOT,OUTPUT,CONDITIONS,storage_check
from .fast_chart_training import read
from .fast_chart_evaluation import decode_identity
from .native_visual_service import write_new
from . import native_visual_service as native
from .local_completion import checkpoint_manifest
from .visual_task import file_sha256,SYSTEM_PROMPT
from .visual_native_evaluation import pair_inputs,verifier_identity,write_pair_parquet,evaluate_pairs
from .evidence import EvaluationIdentity,fingerprint
from .chart_answer_protocol import PROTOCOL
from .chart_selection_protocol import selection_modes
from .fast_chart_admission import verified_calibration

REGISTERED_CONDITIONS = (*CONDITIONS, 'marginal_rank_floor')


def endpoint_identity(plan, *, bind=False):
    """Build or validate the same dataset/scorer identity in every endpoint path."""
    partition=plan['partition'];phase='registered-endpoint-'+partition
    if partition in ('T','R'):
        if plan['role']!=('V' if partition=='R' else 'T'):
            raise ValueError('Endpoint partition and paired data role disagree')
        verifier=verifier_identity(PROTOCOL)
        identity=asdict(EvaluationIdentity(plan['model']['weights_sha256'],plan['manifest_sha256'],
            plan['audit_data_sha256'],plan['decode_sha256'],verifier,phase))
    elif partition=='chartqa':
        from .chartqa_open_answer import PROTOCOL as external_protocol
        if plan['role']!='external':raise ValueError('ChartQA requires the external data role')
        verifier=fingerprint({'protocol':external_protocol,'sources':{name:file_sha256(ROOT/'ours'/name)
            for name in ('chartqa_open_answer.py','chart_answer_protocol.py','fast_chart_external_hook.py')}})
        identity={'weights_sha256':plan['model']['weights_sha256'],'manifest_sha256':plan['manifest_sha256'],
            'decode_sha256':plan['decode_sha256'],'verifier_sha256':verifier,'phase':phase,
            'protocol':external_protocol}
        if bind:plan.pop('audit_data_sha256',None)
        elif 'audit_data_sha256' in plan:raise ValueError('External endpoint retains an inapplicable paired audit identity')
    else:raise ValueError('Unknown endpoint partition')
    fields={'identity':identity,'phase':phase,'verifier_sha256':verifier}
    if bind:plan.update(fields)
    elif any(plan.get(key)!=value for key,value in fields.items()):
        raise ValueError('Endpoint identity differs from its weights, data, decoding or scorer')
    return identity


def register(path,reference,configuration,*,condition,seed,selection=None):
    path,reference,configuration=map(lambda p:Path(p).resolve(),(path,reference,configuration))
    ref=read(reference);cfg=verified_calibration(configuration)
    if condition not in REGISTERED_CONDITIONS or seed not in (42,43,44) or cfg['status']!='COMPLETE_SHARED_CHART_CONFIGURATION':
        raise ValueError('Require registered condition/seed and frozen shared h0/LR')
    if ref['seed']!=seed:
        raise ValueError('Endpoint reference belongs to another seed')
    result=read(Path(ref['output'])/'result.json')
    if result['status']!='COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or result['plan_sha256']!=file_sha256(reference):
        raise ValueError('Endpoint weights require completed real V evaluation')
    evidence={str(reference):file_sha256(reference),str(configuration):file_sha256(configuration),
        str(Path(ref['output'])/'result.json'):file_sha256(Path(ref['output'])/'result.json')}
    if condition!='harness_only':
        followup=read(reference.parent/'plan.json');training_path=Path(followup['training_plan']);training=read(training_path)
        if training['kind']!='compact_chart_rsft_stage' or training['stage']!=2 or training['seed']!=seed or training['rate']!=cfg['learning_rate']:
            raise ValueError('Endpoint is not the frozen four-plus-four native training budget')
        if read(reference.parent/'transition.json')['exported']!=ref['model']:
            raise ValueError('Endpoint serving weights differ from the verified native export')
        if bool(training.get('augmentation'))!=(condition=='counterfactual_augmentation'):
            raise ValueError('Claimed condition differs from the actual augmentation setting')
        evidence[str(training_path)]=file_sha256(training_path)
        # A completed zero-update branch is a reportable result, not grounds
        # for excluding a baseline or replacing the frozen checkpoint.
        evidence[str(reference.parent/'transition.json')]=file_sha256(reference.parent/'transition.json')
    else:
        from .fast_chart_training import MODEL
        if ref['model']!=checkpoint_manifest(MODEL):
            raise ValueError('Harness-only requires unchanged theta0 weights')
    if condition!='weight_only':
        if selection is None:raise ValueError('Optimized conditions require their complete selection receipt')
        selection=Path(selection).resolve();choice=read(selection)
        expected={'veto':'veto','marginal_gate':'marginal_gate','marginal_rank_floor':'marginal_rank_floor'}.get(condition,'whale')
        if choice['status']!='COMPLETE_COMPACT_SELECTION' or choice['condition']!=expected:
            raise ValueError('Wrong endpoint selection rule')
        if expected not in dict(selection_modes(choice)):
            raise ValueError('Endpoint condition is absent from its selection protocol')
        if choice['harness_sha256']!=ref['harness_sha256']:
            raise ValueError('Endpoint harness differs from its selection receipt')
        if condition!='harness_only' and Path(training['selection']).resolve()!=selection:
            raise ValueError('Endpoint used a different selection handoff during training')
        if condition=='harness_only' and choice['parent_plan_sha256'] is not None:
            raise ValueError('Harness-only cannot reuse a theta1 search')
        if condition=='harness_only':
            search_path=selection.parent/'plan.json';search=read(search_path)
            if (search['seed']!=seed or search['identity']!=choice['identity'] or
                    choice['identity']['weights_sha256']!=ref['model']['weights_sha256'] or
                    choice['archive']['h0']['candidate']['harness_sha256']!=file_sha256(Path(cfg['harness']))):
                raise ValueError('Harness-only search changed the shared starting model, harness or seed')
            evidence[str(search_path)]=file_sha256(search_path)
        evidence[str(selection)]=file_sha256(selection)
    elif file_sha256(Path(cfg['harness']))!=ref['harness_sha256']:
        raise ValueError('Weight-only endpoint changed the common h0')
    record={'kind':'compact_endpoint_registration','condition':condition,'seed':seed,'reference':str(reference),
        'selection_protocol':choice.get('selection_protocol','gate_v1') if condition!='weight_only' else None,
        'configuration':str(configuration),'evidence_sha256':evidence,'model':ref['model'],
        'harness':ref['config']['data']['visual_harness_path'],'harness_sha256':ref['harness_sha256'],
        'checkpoint_selection':'Frozen native budget, never test performance','test_accessed':False}
    write_new(path,record);return record


def prepare(path,output,registration,*,partition,common_h0=False):
    from .fast_chart_inference_policy import POLICY,load as load_inference_policy,bind
    path,output,registration=map(lambda p:Path(p).resolve(),(path,output,registration))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT):raise ValueError('Require fresh endpoint evaluation')
    endpoint=read(registration)
    if endpoint['kind']!='compact_endpoint_registration' or partition not in ('T','R','chartqa'):
        raise ValueError('Require a frozen endpoint and declared independent data role')
    for name,digest in endpoint['evidence_sha256'].items():
        if file_sha256(Path(name))!=digest:raise ValueError('Frozen endpoint evidence changed')
    policy=load_inference_policy()
    if file_sha256(Path(read(endpoint['configuration'])['harness']))!=policy['h0_sha256']:
        raise ValueError('Endpoint and throughput calibration have different common h0')
    if common_h0 and endpoint['condition'] not in ('whale','veto'):raise ValueError('Common-h0 endpoint probe is registered only for WHALE/VETO')
    plan=deepcopy(read(endpoint['reference']));cfg=plan['config']
    if common_h0:cfg['data']['visual_harness_path']=read(endpoint['configuration'])['harness']
    cfg['data']['cache_dir']=str(output/'dataset-cache');cfg['trainer']['experiment_name']=output.name
    cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']=output.name
    plan.update(kind='compact_registered_endpoint_evaluation',registration=str(registration),registration_sha256=file_sha256(registration),
        selection_protocol=endpoint.get('selection_protocol','gate_v1'),
        output=str(output),partition=partition,condition=endpoint['condition'],seed=endpoint['seed'],common_h0=common_h0,
        harness_sha256=file_sha256(Path(cfg['data']['visual_harness_path'])))
    bind(plan,policy)
    plan.update(inference_policy=str(POLICY),inference_policy_sha256=file_sha256(POLICY))
    if partition in ('T','R'):
        manifest=OUTPUT/f'dataset/{partition}/manifest.json';m,pairs,_=pair_inputs(manifest)
        plan.update(manifest=str(manifest),manifest_sha256=file_sha256(manifest),audit_data_sha256=m['audit_data_sha256'],role=m['role'])
        images=2*len(pairs)
    else:
        source=OUTPUT/'chartqa512-v1';manifest=source/'plan.json'
        if read(source/'result.json')['status']!='READY_FIXED_EXTERNAL_SUBSET':raise ValueError('External sources or overlap screen incomplete')
        plan.update(manifest=str(manifest),manifest_sha256=file_sha256(manifest),role='external',
            external_receipt_sha256=file_sha256(source/'result.json'))
        cfg['reward']['custom_reward_function']={'path':'pkg://ours.fast_chart_external_hook','name':'compute_score'}
        images=512
    plan['decode_sha256']=decode_identity(cfg,plan['model']['assets'])
    endpoint_identity(plan,bind=True)
    sources=set(plan['source_sha256']) | set(native.SOURCES)
    sources.update(('ours/fast_chart_endpoint_evaluation.py','ours/chart_selection_protocol.py','ours/fast_chart_external_hook.py','ours/chartqa_open_answer.py',
                 'ours/fast_chart_admission.py','ours/run_fast_chart_endpoint_evaluation.sh',
                 'ours/fast_chart_inference_policy.py','ours/fast_chart_throughput.py',
                 'ours/probe_updated_vllm.py','ours/audit_native_training_batch.py'))
    for name in sorted(sources):
        plan['source_sha256'][name]=file_sha256(ROOT/name)
    plan['bounds']={'gpus':1,'cpus':12,'time_limit_seconds':7200 if images==2048 else 3600,'images':images,
        'maximum_generation_calls':images*3,'maximum_generated_assistant_tokens':images*1024,'api_calls':0}
    storage_check(12);write_new(path,plan);return plan


def check(path):
    from .fast_chart_inference_policy import POLICY,load as load_inference_policy,bind
    plan=read(path)
    if plan['kind']!='compact_registered_endpoint_evaluation' or Path(plan['output']).exists():raise ValueError('Wrong or executed endpoint plan')
    if file_sha256(Path(plan['registration']))!=plan['registration_sha256']:raise ValueError('Endpoint registration changed')
    endpoint=read(plan['registration'])
    if plan['inference_policy']!=str(POLICY) or file_sha256(POLICY)!=plan['inference_policy_sha256']:
        raise ValueError('Endpoint does not use the frozen common inference policy')
    expected=deepcopy(plan);bind(expected,load_inference_policy())
    if expected!=plan:raise ValueError('Endpoint scheduling differs across methods or seeds')
    for name,digest in {**plan['source_sha256'],**endpoint['evidence_sha256']}.items():
        if file_sha256(ROOT/name)!=digest:raise ValueError('Frozen endpoint source/evidence changed')
    if file_sha256(Path(plan['manifest']))!=plan['manifest_sha256'] or checkpoint_manifest(Path(plan['model']['path']))!=plan['model']:
        raise ValueError('Endpoint data or weights changed')
    if decode_identity(plan['config'],plan['model']['assets'])!=plan['decode_sha256']:raise ValueError('Endpoint decoding changed')
    if file_sha256(Path(plan['config']['data']['visual_harness_path']))!=plan['harness_sha256']:
        raise ValueError('Endpoint harness changed')
    endpoint_identity(plan)
    storage_check(12);return plan


async def external(manager,dataset,plan,output,expected):
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn
    from .chartqa_open_answer import score_response,aggregate
    from .chartqa_open_answer import PROTOCOL as external_protocol
    from .chart_answer_protocol import SharedAnswerHarness
    if not isinstance(dataset.visual_harness,SharedAnswerHarness):
        raise ValueError('External evaluation requires the shared host parser')
    incoming=list(dataset.dataframe);ids=[r.get('visual_sample_id') for r in incoming]
    if len(ids)!=512 or len(set(ids))!=512 or set(ids)!=set(expected):
        raise ValueError('External dataset filtering or duplication changed the fixed subset')
    source=Path(plan['manifest']).parent
    for item in incoming:
        row=expected[item['visual_sample_id']]
        prompt=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'<image>\n'+row['question']}]
        if (item['prompt']!=prompt or item['images']!=[{'bytes':(source/'png'/row['source_image']).read_bytes()}]
                or item['reward_model']!={'style':'rule','ground_truth':row['answer']}
                or item['data_source']!=external_protocol or item.get('videos')):
            raise ValueError('External pixels, question or answer differ from the fixed source')
    dataset.visual_harness.unchanged()
    output.mkdir();rows=[];calls=tokens=0;artifacts={}
    batch_size=plan['batch_size']
    if batch_size not in (8,16,32) or len(dataset)%batch_size:raise ValueError('Unregistered external batch coverage')
    for offset in range(0,len(dataset),batch_size):
        items=[dataset[i] for i in range(offset,offset+batch_size)];batch=DataProto.from_single_dict(collate_fn(items));batch.meta_info.update(validate=True)
        result=await manager.generate_sequences(batch);archive=output/f'batch-{offset//batch_size:04d}.pkl'
        result.save_to_disk(archive);artifacts[archive.name]=file_sha256(archive)
        meta=result.non_tensor_batch;ids=meta['visual_sample_id'].tolist()
        if len(ids)!=batch_size or len(set(ids))!=batch_size or set(ids)!={x['visual_sample_id'] for x in items}:raise ValueError('External batch coverage differs')
        records=[]
        for i,ident in enumerate(ids):
            row=expected[ident];value=score_response(meta['visual_final_answer'][i],row['answer'])
            if value['committed_answer']!=meta['visual_committed_answer'][i] or result.batch['rm_scores'][i].sum().item()!=value['correct']:
                raise ValueError('External native reward differs from shared original-answer scoring')
            if meta['visual_harness_sha256'][i]!=plan['harness_sha256']:raise ValueError('Endpoint harness changed')
            score=result.batch['rm_scores'][i];mask=result.batch['response_mask'][i]
            attention=result.batch['attention_mask'][i,-len(mask):]
            if not bool(((mask==0)|(mask==1)).all()) or bool((mask>attention).any()):
                raise ValueError('External generated-token mask is invalid')
            if score.nonzero().flatten().tolist()!=([int(attention.sum().item())-1] if value['correct'] else []):
                raise ValueError('External reward is not placed on the final token')
            if (type(meta['visual_policy_calls'][i]) is not int or meta['visual_policy_calls'][i]<1 or
                    type(meta['visual_generated_tokens'][i]) is not int or meta['visual_generated_tokens'][i]<int(mask.sum().item())):
                raise ValueError('External call or token accounting is invalid')
            record={**value,'sample_id':ident,'split':row['split'],'source_image':row['source_image'],
                'generated_tokens':meta['visual_generated_tokens'][i],'policy_calls':meta['visual_policy_calls'][i],
                'native_turns':int(meta['__num_turns__'][i]),'tool_trace':meta['visual_harness_tool_trace'][i]}
            records.append(record);calls+=record['policy_calls'];tokens+=record['generated_tokens']
        rows.extend(records);write_new(output/f'batch-{offset//batch_size:04d}.json',records)
    dataset.visual_harness.unchanged()
    report={'status':'COMPLETE_FIXED_CHARTQA_SUBSET','records':rows,'scores':aggregate(rows,expected),
        'policy_calls':calls,'generated_tokens':tokens,'batch_artifact_sha256':artifacts}
    write_new(output/'result.json',report);return report


async def run(plan,path):
    identity=endpoint_identity(plan)
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import ray,torch
    from omegaconf import OmegaConf
    from .native_visual_agent_observation import ObservedVisualAgentManager
    if torch.cuda.device_count()!=1 or 'H800' not in torch.cuda.get_device_name(0):raise ValueError('Expected one H800')
    out=Path(plan['output']);out.mkdir();(out/'requests').mkdir();(out/'workers').mkdir();(out/'plan.json').write_bytes(path.read_bytes())
    for name in plan['source_sha256']:
        source=ROOT/name;target=out/'source'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    os.environ.update(VETO_NATIVE_EVALUATION_PLAN=str(path),VETO_NATIVE_EVALUATION_OUTPUT=str(out))
    hook='ours.training_bootstrap.prepare_worker';parquet=out/'evaluation.parquet';expected=None
    if plan['partition']=='chartqa':
        import pandas as pd
        from .chartqa_open_answer import PROTOCOL as external_protocol
        source=Path(plan['manifest']).parent;manifest=read(plan['manifest']);receipt=read(source/'result.json')
        if file_sha256(source/'result.json')!=plan['external_receipt_sha256']:raise ValueError('External source receipt changed')
        records=[];expected={r['sample_id']:r for r in manifest['rows']}
        for i,row in enumerate(manifest['rows']):
            image=source/'png'/row['source_image']
            if file_sha256(image)!=receipt['artifacts']['png/'+row['source_image']]['sha256']:raise ValueError('External image bytes changed')
            records.append({'visual_sample_id':row['sample_id'],'data_source':external_protocol,
                'prompt':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'<image>\n'+row['question']}],
                'images':[{'bytes':image.read_bytes()}],'reward_model':{'style':'rule','ground_truth':row['answer']},'extra_info':{'index':i}})
        with parquet.open('xb') as f:pd.DataFrame(records).to_parquet(f,index=False)
        hook='ours.fast_chart_external_hook.prepare_worker'
    else:write_pair_parquet(plan['manifest'],parquet)
    try:
        dataset=native.dataset_for(plan,parquet)
        env={k:v for k,v in os.environ.items() if k.startswith(('VETO_','HF_','TRANSFORMERS_','VERL_','VLLM_')) or k in
            ('PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','NO_PROXY','no_proxy')}
        ray.init(num_cpus=12,num_gpus=1,include_dashboard=False,object_store_memory=2*1024**3,
            runtime_env={'env_vars':env,'worker_process_setup_hook':hook})
        manager=await ObservedVisualAgentManager.create(OmegaConf.create(plan['config']))
        if expected is not None:result=await external(manager,dataset,plan,out/'evaluation',expected)
        else:
            _,result=await evaluate_pairs(manager,dataset,manifest_path=plan['manifest'],identity=EvaluationIdentity(**identity),
                output=out/'evaluation',batch_size=plan['batch_size'])
        observed=native.audit_requests(plan,out,result)
        write_new(out/'result.json',{'status':'COMPLETE_REGISTERED_ENDPOINT_EVALUATION','plan_sha256':file_sha256(path),
            'identity':identity,
            'job_id':os.environ['SLURM_JOB_ID'],'partition':plan['partition'],'condition':plan['condition'],'seed':plan['seed'],
            'observed':observed,'evaluation_result_sha256':file_sha256(out/'evaluation/result.json'),'scientific_method_verified':False})
    except BaseException as exc:
        write_new(out/'failure.json',{'status':'INCOMPLETE','error_type':type(exc).__name__,'error':str(exc)});raise
    finally:ray.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','prepare','check','run'))
    for key in ('plan','reference','configuration','selection','output','registration'):p.add_argument('--'+key,type=Path,required=key=='plan')
    p.add_argument('--condition',choices=REGISTERED_CONDITIONS);p.add_argument('--seed',type=int,choices=(42,43,44))
    p.add_argument('--partition',choices=('T','R','chartqa'));p.add_argument('--common-h0',action='store_true');a=p.parse_args()
    if a.action=='register':register(a.plan,a.reference,a.configuration,condition=a.condition,seed=a.seed,selection=a.selection)
    elif a.action=='prepare':prepare(a.plan,a.output,a.registration,partition=a.partition,common_h0=a.common_h0)
    elif a.action=='check':check(a.plan)
    else:asyncio.run(run(check(a.plan),a.plan.resolve()))
