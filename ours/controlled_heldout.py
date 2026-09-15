"""Frozen final evaluation of one certified trial, without proposer feedback."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess

from .audit_native_training_batch import require
from .experiment_randomization import TrialRandomization
from .heldout_cohort import CONDITIONS,SEEDS,PROTOCOL,check as check_cohort,read_protocol
from .local_completion import checkpoint_manifest
from .native_search import write_json
from .preflight import PIN
from .visual_task import file_sha256

WRAPPER='ours/run_controlled_heldout.sh'
SOURCES=('ours/controlled_heldout.py',WRAPPER,'ours/heldout_evaluation.py','ours/audit_heldout_evaluation.py',
    'ours/heldout_cohort.py','ours/native_search.py','ours/vllm_runtime.py','ours/local_completion.py',
    'ours/phase_vllm_worker.py','ours/probe_updated_vllm.py','ours/experiment_randomization.py',
    'ours/audit_mh_phase.py','ours/audit_native_training_batch.py','ours/visual_task.py',
    'ours/native_chess_semantics.md','ours/compat/autoharness_textarena/llm.py',
    'ours/compat/autoharness_textarena/config.py','ours/compat/autoharness_textarena/__init__.py',
    'upstream/WHALE/domains/chess_puzzles/autoharness_chess_puzzle/runner.py',
    'upstream/WHALE/domains/chess_puzzles/autoharness_chess_puzzle/harness.py')


def target_config(model,seed):
    return {'model':model['path'],'provider':'local-vllm','max_tokens':8129,'temperature':1.,'top_p':1.,'top_k':20,
        'batch_concurrency':4,'chat_template_kwargs':{'enable_thinking':True},'seed':seed,
        'expected_weights_sha256':model['weights_sha256'],'request_timeout_seconds':420}


def server_arguments(seed):
    return ['--dtype','bfloat16','--max-model-len','32768','--max-num-seqs','8','--max-num-batched-tokens','16384',
        '--gpu-memory-utilization','0.60','--enforce-eager','--enable-prefix-caching','--enable-chunked-prefill',
        '--gdn-prefill-backend','triton','--distributed-executor-backend','uni','--seed',str(seed),
        '--worker-cls','ours.phase_vllm_worker.CheckpointVerifiedWorker']


def select_probe_reference(model,common):
    from .probe_updated_vllm import changed_embedding_coordinates
    from .staged_search import REFERENCE
    candidates=[json.loads(REFERENCE.read_text())['exported'],common]
    for reference in candidates:
        require(checkpoint_manifest(Path(reference['path']))==reference,'Changed numerical probe reference')
        try:coordinates=changed_embedding_coordinates(Path(reference['path']),Path(model['path']))
        except ValueError as error:
            if str(error)!='Insufficient changed embedding coordinates for handoff proof':raise
        else:return reference,coordinates
    raise ValueError('No eight-coordinate numerical discrimination reference is available')


def prepare(path,cohort_path,condition,seed,root,gpus):
    from .probe_updated_vllm import inference_contract
    require(not path.exists() and not root.exists(),'Preserve existing final evaluation plan or attempt')
    protocol=read_protocol();trial=check_cohort(cohort_path,condition=condition,seed=seed)
    require(gpus in (1,2,4),'Outside predeclared physical allocation widths')
    resources=next(r for r in protocol['physical_allocations'] if r['gpus']==gpus)
    reference,coordinates=select_probe_reference(trial['model'],trial['initial_manifest'])
    contract=inference_contract(Path(trial['initial_manifest']['path']),Path(trial['model']['path']))
    plan={'kind':'controlled_heldout_plan','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'condition':condition,'seed':seed,'data_role':'test','veto_mode':'off','cohort':str(cohort_path.resolve()),
        'cohort_sha256':file_sha256(cohort_path),'protocol':str(PROTOCOL.resolve()),'protocol_sha256':file_sha256(PROTOCOL),
        'evaluation_root':str(root.resolve()),'dataset':protocol['dataset'],'dataset_sha256':protocol['dataset_sha256'],
        'examples':64,'logical_shards':4,'puzzle_ids':protocol['puzzle_ids'],'shard_ids':protocol['shard_ids'],
        'target_manifest':trial['model'],'initial_manifest':trial['initial_manifest'],
        'harness':trial['harness'],'harness_sha256':trial['harness_sha256'],'final_provenance':trial['provenance'],
        'target_config':target_config(trial['model'],seed),'server_arguments':server_arguments(seed),
        'worker_probe_reference':reference,'worker_probe_coordinates':coordinates,'inference_contract':contract,
        'startup_timeout_seconds':360,'resources':resources,'bounds':protocol['bounds_per_completed_trial'],
        'source_sha256':{n:file_sha256(Path(n)) for n in SOURCES},'upstream_pin':PIN,
        'versions':{n:version(n) for n in ('torch','vllm','transformers','chess','pandas','pyarrow','numpy')},
        'test_tasks_loaded_during_preparation':False,'new_model_calls':0,'new_api_calls':0,
        'limitations':['Final fixed model/harness measurement; no search, training or proposer feedback is run here.',
            'Four logical shards and their seeds are fixed independently of physical allocation width.',
            'Eight worker coordinates are a numerical probe, not full served-parameter inspection.',
            'Native harness retry attributes override shared environment defaults; every request counts.',
            'Inference records have prompt/output token IDs; there is no RSFT loss mask during heldout inference.',
            'Incomplete trials remain missing, and comparative/three-seed conclusions require the full declared table.']}
    write_json(path,plan);return plan


def check_plan(path,*,verify_weights=True):
    from .probe_updated_vllm import changed_embedding_coordinates,inference_contract
    plan=json.loads(path.read_text());protocol=read_protocol(Path(plan['protocol']))
    require(plan['kind']=='controlled_heldout_plan' and plan['data_role']=='test' and plan['veto_mode']=='off' and
            plan['condition'] in CONDITIONS and plan['seed'] in SEEDS and plan['examples']==64 and plan['logical_shards']==4 and
            plan['test_tasks_loaded_during_preparation'] is False and plan['new_model_calls']==plan['new_api_calls']==0,
            'Wrong final evaluation scope')
    require(file_sha256(Path(plan['cohort']))==plan['cohort_sha256'] and file_sha256(PROTOCOL)==plan['protocol_sha256'],
            'Changed sealed cohort or protocol')
    trial=check_cohort(Path(plan['cohort']),condition=plan['condition'],seed=plan['seed'],verify_dependencies=verify_weights)
    require(plan['target_manifest']==trial['model'] and plan['initial_manifest']==trial['initial_manifest'] and
            plan['harness']==trial['harness'] and plan['harness_sha256']==trial['harness_sha256'] and
            plan['final_provenance']==trial['provenance'],'Changed selected final model/harness lineage')
    require(all(plan[k]==protocol[k] for k in ('dataset','dataset_sha256','puzzle_ids','shard_ids')) and
            plan['bounds']==protocol['bounds_per_completed_trial'] and plan['resources'] in protocol['physical_allocations'],
            'Changed heldout data, sharding or allocation budget')
    require(set(plan['source_sha256'])==set(SOURCES),'Missing heldout execution source')
    for n,h in plan['source_sha256'].items():require(file_sha256(Path(n))==h,f'Changed heldout source: {n}')
    require({n:version(n) for n in plan['versions']}==plan['versions'],'Changed heldout runtime')
    require(plan['upstream_pin']==PIN and subprocess.check_output(['git','-C','upstream/WHALE','rev-parse','HEAD'],text=True).strip()==PIN and
            not subprocess.check_output(['git','-C','upstream/WHALE','status','--porcelain'],text=True).strip(),'Changed upstream')
    require(plan['target_config']==target_config(trial['model'],plan['seed']) and plan['server_arguments']==server_arguments(plan['seed']) and
            plan['startup_timeout_seconds']==360,'Changed heldout decoding or engine configuration')
    coordinates=plan['worker_probe_coordinates']
    require(len(coordinates)==8 and len({(c['token_id'],c['column']) for c in coordinates})==8 and
            all(c['base']!=c['updated'] for c in coordinates),'Missing distinct live-worker probe coordinates')
    require(file_sha256(Path(plan['harness']))==plan['harness_sha256'],'Changed selected harness')
    if verify_weights:
        reference,expected=select_probe_reference(trial['model'],trial['initial_manifest'])
        require(reference==plan['worker_probe_reference'] and expected==coordinates,'Changed numerical worker discrimination')
        require(inference_contract(Path(trial['initial_manifest']['path']),Path(trial['model']['path']))==plan['inference_contract'],
                'Changed final inference assets')
    return plan


def evaluation_kwargs(plan,private):
    return {'harness_path':plan['harness'],'dataset_path':plan['dataset'],'llm_config':plan['target_config'],
        'output_dir':str(private.resolve()/'merged'),'limit':64,'seed':plan['seed'],
        'assistant_token_budget':8129,'policy_max_tokens':8129}


def validate_request(plan,path,private,request):
    require(plan['kind']=='controlled_heldout_plan' and plan['data_role']=='test' and plan['examples']==64 and
            plan['logical_shards']==4 and private.resolve()==Path(plan['evaluation_root']).resolve(),
            'Wrong heldout request role or location')
    require(request['plan_sha256']==file_sha256(path) and request['cohort_sha256']==plan['cohort_sha256'] and
            request['condition']==plan['condition'] and request['seed']==plan['seed'] and
            request['harness_sha256']==plan['harness_sha256']==file_sha256(Path(plan['harness'])) and
            request['kwargs']==evaluation_kwargs(plan,private),'Request differs from frozen final trial')



def validate_execution_receipts(plan,path,private,result):
    start=json.loads((private/'start.json').read_text());gpus=plan['resources']['gpus']
    require(all(start[k]==v for k,v in {'job_id':result['job_id'],'plan_sha256':file_sha256(path),
        'cohort_sha256':plan['cohort_sha256'],'condition':plan['condition'],'seed':plan['seed'],
        'request_sha256':file_sha256(private/'request.json'),'physical_gpus':gpus,'new_model_calls':0}.items()) and
        file_sha256(private/'plan.json')==file_sha256(path),'Changed heldout start or copied plan')
    devices=start['visible_devices']
    require(len(devices)==len(set(devices))==gpus and all(isinstance(d,str) and d for d in devices),
            'Wrong recorded physical GPU assignment')
    expected=[{'wave':offset//gpus,'assignments':[{'replica':offset+i,'visible_device':d} for i,d in enumerate(devices)],
        'all_shards_completed':True} for offset in range(0,4,gpus)]
    require(json.loads((private/'waves.json').read_text())==expected,'Incomplete or changed logical shard waves')


def close(path,slurm_path):
    plan=check_plan(path);private=Path(plan['evaluation_root']);result=json.loads((private/'result.json').read_text())
    request=json.loads((private/'request.json').read_text());validate_request(plan,path,private,request)
    require(result['status']=='PASS' and result['plan_sha256']==file_sha256(path) and
            result['cohort_sha256']==plan['cohort_sha256'] and result['condition']==plan['condition'] and
            result['seed']==plan['seed'] and result['request_sha256']==file_sha256(private/'request.json'),
            'Incomplete or mixed final evaluation result')
    require({'audit.json','merged/val.json','merged/trajectories.jsonl','waves.json','start.json','plan.json','request.json'}<=set(result['artifact_sha256']),
            'Missing complete heldout evidence')
    for n,h in result['artifact_sha256'].items():
        p=private/n;require(p.resolve().is_relative_to(private.resolve()) and file_sha256(p)==h,f'Changed heldout artifact: {n}')
    validate_execution_receipts(plan,path,private,result)
    audit=json.loads((private/'audit.json').read_text());summary=json.loads((private/'merged/val.json').read_text())
    require(audit['kind']=='controlled_heldout_evaluation_audit' and audit['status']=='PASS' and audit['data_role']=='test' and
            audit['condition']==plan['condition'] and audit['seed']==plan['seed'] and audit['examples']==summary['num_examples']==64 and
            audit['plan_sha256']==file_sha256(path) and audit['harness_sha256']==plan['harness_sha256'] and
            audit['audit_source_sha256']==file_sha256(Path('ours/audit_heldout_evaluation.py')) and audit['new_model_calls']==0 and
            audit['solved']==result['solved']==summary['solved_examples'] and audit['calls']==result['new_model_calls'] and
            audit['generated_tokens']==result['generated_tokens'],'Final scores differ from complete native audit')
    fields=dict(re.findall(r'(\w+)=([^\s]+)',slurm_path.read_text()));gpus=plan['resources']['gpus']
    require(fields['JobId']==result['job_id'] and fields['JobState']=='COMPLETED' and fields['ExitCode']=='0:0',
            'Heldout allocation did not complete successfully')
    allocated=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
    require(allocated.get('gres/gpu')==allocated.get('gres/gpu:h800')==str(gpus) and
            Path(fields['Command']).resolve()==Path(WRAPPER).resolve(),'Wrong heldout allocation or entrypoint')
    resources=plan['resources'];memory=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT])',allocated.get('mem',''))
    require(allocated.get('cpu')==str(resources['cpus']) and memory is not None and
            Decimal(memory[1])*Decimal(1024)**('KMGT'.index(memory[2])+1)==resources['ram_gib']*1024**3,
            'Wrong heldout CPU or RAM allocation')
    days,_,clock=fields['RunTime'].rpartition('-');h,m,s=map(int,clock.split(':'));seconds=int(days or 0)*86400+3600*h+60*m+s
    require(0<=seconds<=plan['resources']['time_limit_seconds'],'Heldout allocation exceeded its frozen budget')
    return {'kind':'controlled_heldout_result','status':'COMPLETED_HELDOUT_EVALUATION','condition':plan['condition'],
        'seed':plan['seed'],'job_id':result['job_id'],'plan':str(path.resolve()),'plan_sha256':file_sha256(path),
        'cohort_sha256':plan['cohort_sha256'],'harness_sha256':plan['harness_sha256'],'target_manifest':plan['target_manifest'],
        'examples':64,'solved':audit['solved'],'success_rate':audit['solved']/64,'calls':audit['calls'],
        'generated_tokens':audit['generated_tokens'],'allocation':{'seconds':seconds,'gpu_hours':gpus*seconds/3600,'alloc_tres':fields['AllocTRES']},
        'slurm_terminal_path':str(slurm_path.resolve()),'slurm_terminal_sha256':file_sha256(slurm_path),
        'evaluation_result_sha256':file_sha256(private/'result.json'),'new_model_calls_by_finalizer':0,'limitations':plan['limitations']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','check','evaluate','worker','close'),required=True)
    for name in ('plan','cohort','root','evaluation','slurm','output'):parser.add_argument('--'+name,type=Path,required=name=='plan')
    parser.add_argument('--condition',choices=CONDITIONS);parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--gpus',type=int,choices=(1,2,4),default=4);parser.add_argument('--replica',type=int,choices=range(4))
    args=parser.parse_args()
    if args.phase=='prepare':
        require(all(v is not None for v in (args.cohort,args.condition,args.seed,args.root)),'Missing final trial preparation input')
        prepare(args.plan,args.cohort,args.condition,args.seed,args.root,args.gpus)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan),'test_tasks_loaded':False}),flush=True)
    elif args.phase=='close':
        require(args.slurm is not None and args.output is not None and not args.output.exists(),'Missing terminal evidence or occupied output')
        result=close(args.plan,args.slurm);write_json(args.output,result);print(json.dumps(result),flush=True)
    else:
        plan=check_plan(args.plan,verify_weights=args.phase!='worker')
        if args.phase=='check':print(json.dumps({'status':'PASS','test_tasks_loaded':False}),flush=True)
        else:
            from .heldout_evaluation import run_evaluation,run_worker
            require(args.evaluation is None or args.evaluation.resolve()==Path(plan['evaluation_root']),'Wrong evaluation directory')
            args.evaluation=Path(plan['evaluation_root']);os.environ.update(TrialRandomization(plan['seed']).environment())
            if args.phase=='worker':
                require(args.replica is not None,'Missing logical shard index');run_worker(args,plan)
            else:run_evaluation(args,plan)
