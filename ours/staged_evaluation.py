"""Two offline GPU shards for one candidate in the original five-round search."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import uuid

from .audit_native_training_batch import require
from .audit_staged_evaluation import audit_evaluation
from .experiment_randomization import TrialRandomization,initialize_process_randomness
from .native_search import target_service,tree_hashes,write_json
from .visual_task import file_sha256
from .vllm_runtime import RecordedVLLMClient,VLLMConfiguration


def run_worker(args, plan):
    randomization=initialize_process_randomness()
    import torch
    from autoharness_chess_puzzle import runner
    require(torch.cuda.device_count() == 1, 'Expected one visible GPU per shard')
    private = args.evaluation.resolve()
    request = json.loads((private/'request.json').read_text())
    require(request['plan_sha256'] == file_sha256(args.plan), 'Wrong request plan')
    require(file_sha256(Path(request['kwargs']['harness_path'])) == request['harness_sha256'], 'Harness changed')
    output = private/f'replica-{args.replica}'; output.mkdir(exist_ok=False)
    write_json(output/'process-randomization.json',randomization)
    clients, captured, ordered = [], [], []
    lock = threading.Lock()
    def record(event):
        with lock,(output/'request-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    class Client(RecordedVLLMClient):
        def __init__(self,config):
            super().__init__(config);clients.append(self)
        def complete_response(self,messages,*,max_tokens=None):
            ident = uuid.uuid4().hex
            record({'event':'start','id':ident,'messages':[asdict(m) for m in messages],'max_tokens':max_tokens})
            try:
                value = super().complete_response(messages,max_tokens=max_tokens)
            except BaseException as error:
                record({'event':'error','id':ident,'type':type(error).__name__,'usage':None});raise
            record({'event':'complete','id':ident,'usage':value.usage});return value
    original_read,original_summary = runner._read_examples,runner.summarize_outputs
    def shard_read(path,*,limit,seed):
        require(str(path)==plan['dataset'] and limit==32 and seed==plan['seed'],'Wrong shard dataset request')
        rows=[e for e in original_read(path,limit=limit,seed=seed) if e.example_id in plan['shard_ids'][args.replica]]
        ordered.extend(e.example_id for e in rows);return rows
    def capture(rows,metadata):
        captured.extend(rows);return original_summary(rows,metadata)
    def terminate(signum,frame):
        raise KeyboardInterrupt(f'Evaluation shard received {signum}')
    signal.signal(signal.SIGTERM,terminate)
    os.environ.update(WHALE_MH_PLAN=str(args.plan.resolve()),WHALE_MH_WORKER_RECEIPT=str(output/'worker-weights.json'))
    started=time.monotonic()
    try:
        with target_service(plan,output) as endpoint:
            proof=json.loads((output/'worker-weights.json').read_text())
            require(proof['status']=='PASS' and proof['plan_sha256']==file_sha256(args.plan),'Missing actual worker proof')
            kwargs=dict(request['kwargs'],output_dir=output/'evaluation',
                llm_config=dict(plan['target_config'],base_url=endpoint,trace_path=str(output/'generations.jsonl')))
            with patch.object(runner,'LLMClient',Client),patch.object(runner,'LLMConfig',VLLMConfiguration), \
                 patch.object(runner,'_read_examples',shard_read),patch.object(runner,'summarize_outputs',capture):
                summary=runner.evaluate_harness(**kwargs)
            require(summary['num_examples']==len(captured)==len(ordered)==16,'Incomplete native shard')
            write_json(output/'native-outputs.json',captured);write_json(output/'ordered-ids.json',ordered)
    finally:
        for client in clients:client.close()
    write_json(output/'result.json',{'status':'COMPLETED','replica':args.replica,
        'plan_sha256':file_sha256(args.plan),'harness_sha256':request['harness_sha256'],
        'seconds':time.monotonic()-started,'gpu':torch.cuda.get_device_name(),
        'artifact_sha256':tree_hashes(output)})


def run_evaluation(args, plan):
    from autoharness_chess_puzzle import runner
    private=args.evaluation.resolve()
    request=json.loads((private/'request.json').read_text())
    require(request['plan_sha256']==file_sha256(args.plan),'Wrong evaluation plan')
    kwargs=request['kwargs']
    require(kwargs['seed']==plan['seed'] and kwargs['limit']==32 and kwargs['dataset_path']==plan['dataset'],
            'Evaluation request differs from frozen task')
    require(kwargs['llm_config']==plan['target_config'] and
            kwargs['assistant_token_budget']==kwargs['policy_max_tokens']==8129,'Changed inference budget')
    require(file_sha256(Path(kwargs['harness_path']))==request['harness_sha256'],'Harness changed')
    require(not (private/'start.json').exists(),'Preserve previous evaluation attempt')
    write_json(private/'start.json',{'job_id':os.environ['SLURM_JOB_ID'],'plan_sha256':file_sha256(args.plan),
        'request_sha256':file_sha256(private/'request.json'),'new_model_calls':0})
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(devices)==len(set(devices))==2,'Expected two visible GPU devices')
    children,logs=[],[]
    try:
        for replica,device in enumerate(devices):
            log=(private/f'replica-{replica}.log').open('x');logs.append(log)
            env=dict(os.environ,**TrialRandomization(plan['seed']).environment(),CUDA_VISIBLE_DEVICES=device)
            children.append(subprocess.Popen([sys.executable,'-m','ours.staged_search','--plan',str(args.plan),
                '--phase','worker','--evaluation',str(private),'--replica',str(replica)],
                env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(child.poll() is None for child in children):
            require(not any(child.poll() not in (None,0) for child in children),'A shard failed; preserve both shards')
            time.sleep(2)
        require(all(child.returncode==0 for child in children),'A shard failed')
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
    outputs={};metadata=[]
    for replica in (0,1):
        directory=private/f'replica-{replica}'
        ids=json.loads((directory/'ordered-ids.json').read_text())
        rows=json.loads((directory/'native-outputs.json').read_text())
        require(len(ids)==len(rows)==16 and not set(ids)&set(outputs),'Duplicate or missing shard rows')
        outputs.update(zip(ids,rows,strict=True))
        metadata.append(json.loads((directory/'evaluation/val.json').read_text())['metadata'])
    require(metadata[0]==metadata[1],'Shard metadata differ')
    expected=[e.example_id for e in runner._read_examples(plan['dataset'],limit=32,seed=plan['seed'])]
    require(set(outputs)==set(expected) and len(outputs)==32,'Incomplete merged coverage')
    combined=[outputs[key] for key in expected]
    meta=dict(metadata[0],limit=32,dataset_path=plan['dataset'],harness_path=kwargs['harness_path'])
    summary=runner.summarize_outputs(combined,meta)
    merged=private/'merged';merged.mkdir(exist_ok=False)
    write_json(merged/'val.json',summary);runner.write_trajectories(combined,merged)
    with (merged/'ChessPuzzle-v0_policy_trace.jsonl').open('x') as stream:
        for row in combined:
            for event in row['harness_trace']:stream.write(json.dumps(event,ensure_ascii=False)+'\n')
    audit=audit_evaluation(plan,args.plan,private)
    write_json(private/'audit.json',audit)
    write_json(private/'result.json',{'status':'PASS','job_id':os.environ['SLURM_JOB_ID'],
        'plan_sha256':file_sha256(args.plan),'request_sha256':file_sha256(private/'request.json'),
        'artifact_sha256':tree_hashes(private),'new_model_calls':audit['calls'],
        'generated_tokens':audit['generated_tokens'],'solved':audit['solved']})
