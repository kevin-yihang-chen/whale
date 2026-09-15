"""Complete continuous weight-only seeds43/44 under the native loader contract.

Verify both full trajectory audits, actual native update summaries, saved sampler
and scheduler states, and terminal allocation. Seed42 retains its separate
preflight-deviation record. Empty batches and unchanged parameters are valid.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .audit_native_training_batch import require
from .controlled_pilot import CPU_NAME
from .controlled_pilot_continuation import CONTRACT,check
from .joint_training_result import phase_metrics as first_metrics,saved_state as first_state
from .joint_training_phase2_result import phase_metrics as second_metrics,saved_state as second_state
from .joint_resume_receipts import CHECKPOINT_FILES
from .native_search import write_json
from .staged_search import terminal_job
from .visual_task import file_sha256

SOURCE=Path(__file__)
SOURCES=('ours/controlled_continuation_result.py','ours/controlled_pilot_continuation.py',
    'ours/audit_controlled_pilot.py','ours/joint_training_result.py','ours/joint_training_phase2_result.py',
    'ours/joint_resume_receipts.py','ours/native_phase_resume.py','ours/controlled_training_result.py',
    'ours/staged_search.py','ours/audit_native_training_batch.py','ours/visual_task.py','ours/native_search.py')
AUDIT_SOURCES={'ours/audit_controlled_pilot.py','ours/audit_alternation_training.py',
    'ours/audit_native_training_batch.py','ours/native_training_trace.py','ours/controlled_pilot.py'}


def batch_metrics(log,batches):
    rows=[]
    for line in log.splitlines():
        found=re.search(r'\bstep:(\d+) - online_rsft/accepted:',line)
        if found:rows.append((int(found.group(1)),line))
    require([step for step,_ in rows]==[1,2] and [b['step'] for b in batches]==[1,2],
            'Missing, duplicate, reordered or extra native batch summaries')
    return [reader(line,batch) for reader,(_,line),batch in zip((first_metrics,second_metrics),rows,batches,strict=True)]


def complete(plan_path,audit_path,slurm_path,log_path):
    from omegaconf import OmegaConf
    plan=check(plan_path);audit=json.loads(audit_path.read_text());job=audit['job_id']
    require(plan['condition']=='weight_only' and plan['seed'] in (43,44) and plan['preflight_order_contract']==CONTRACT,
            'Expected an independent unstarted-seed continuation plan')
    require(audit['kind']=='controlled_pilot_batch_audit' and audit['status']=='PASS' and
            audit['condition']==plan['condition'] and audit['seed']==plan['seed'] and
            audit['plan_sha256']==file_sha256(plan_path) and audit['trajectories']==128 and audit['new_model_calls']==0,
            'Wrong completed continuation audit')
    require([b['step'] for b in audit['batches']]==[1,2],'Incomplete native two-batch audit')
    all_rows=[]
    for step,batch in enumerate(audit['batches'],1):
        ids=plan['native_dataset_order']['planned_batch_ids'][step-1];rows=batch['trajectory_checks']
        require(batch['puzzle_ids']==ids and len(ids)==len(set(ids))==8 and len(rows)==64 and
                {r['row'] for r in rows}==set(range(64)) and
                Counter(r['puzzle_id'] for r in rows)==Counter({key:8 for key in ids}) and
                sum(r['accepted'] for r in rows)==batch['accepted'] and
                sum(r['assistant_reencoded_tokens'] for r in rows if r['accepted'])==batch['accepted_loss_tokens'],
                'Incomplete or inconsistent native batch coverage')
        all_rows.extend(rows)
    accounting=audit['request_accounting']
    require(accounting['observed_requests_fully_accounted'] and
            accounting['recorded_starts']==accounting['completed']==sum(r['calls'] for r in all_rows) and
            all(accounting[k]==0 for k in ('errors','pending_or_interrupted','partial_tail_records','completed_without_usage')) and
            accounting['recorded_starts']<=plan['max_policy_calls'] and
            0<=accounting['known_completion_tokens']<=plan['max_assistant_output_tokens'],
            'Incomplete or over-budget request accounting')
    require(set(audit['audit_source_sha256'])==AUDIT_SOURCES,'Missing original full-audit source binding')
    for name,digest in {**audit['audit_source_sha256'],**audit['artifact_sha256']}.items():
        require(file_sha256(Path(name))==digest,f'Changed completed audit evidence: {name}')
    raw=slurm_path.read_text();allocation=terminal_job(raw,job)
    require(allocation is not None and allocation['seconds']<=plan['resources']['time_limit_seconds'],
            'Incomplete or over-budget continuous training allocation')
    fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
    require(Path(fields['StdOut']).resolve()==log_path.resolve() and
            Path(fields['Command']).resolve()==Path('ours/run_controlled_pilot_continuation.sh').resolve(),
            'Slurm log or continuation entrypoint differs')
    name=f"controlled-weight-only-seed{plan['seed']}-{job}"
    start_root=Path(f'results/controlled-pilot-{job}');start_path=start_root/'start.json'
    effective=start_root/'effective-config.yaml';start=json.loads(start_path.read_text())
    require(str(start_path) in audit['artifact_sha256'] and start['status']=='PASS' and
            start['plan_sha256']==file_sha256(plan_path) and start['run_name']==name and start['seed']==plan['seed'] and
            start['native_batch_steps']==2 and start['new_model_calls']==0 and start['preflight_order_contract']==CONTRACT and
            start['entrypoint_source_sha256']==file_sha256(Path('ours/controlled_pilot_continuation.py')),
            'Missing or changed native continuation launch receipt')
    def config(text):return OmegaConf.to_container(OmegaConf.create(text[text.index('model_engine: dp\n'):]),resolve=True)
    frozen=config(Path(plan['resolved_config']).read_text().replace(CPU_NAME,name))
    require(config(effective.read_text())==frozen,'Actual launch configuration differs from the frozen plan')
    root=Path(frozen['trainer']['default_local_dir'])
    require(root.name==name and sorted(p.name for p in root.glob('global_step_*'))==['global_step_1','global_step_2'] and
            (root/'latest_checkpointed_iteration.txt').read_text().strip()=='2','Missing or extra continuous native checkpoints')
    metrics=batch_metrics(log_path.read_text(),audit['batches'])
    first=root/'global_step_1';second=root/'global_step_2'
    states=[first_state(first,plan,metrics[0]['optimizer_steps']),
        second_state(second,{**plan,'resume_checkpoint':{'directory':str(first)}},metrics[1]['optimizer_steps'])]
    checkpoints=[{'step':step,'directory':str(directory.resolve()),
        'artifact_sha256':{n:file_sha256(directory/n) for n in CHECKPOINT_FILES}}
        for step,directory in ((1,first),(2,second))]
    artifacts=(plan_path,audit_path,slurm_path,log_path,start_path,effective,root/'latest_checkpointed_iteration.txt')
    return {'kind':'controlled_weight_only_continuation_result','status':'COMPLETED_NATIVE_CONTINUATION',
        'condition':plan['condition'],'seed':plan['seed'],'job_id':job,'plan_sha256':file_sha256(plan_path),
        'batch_audit_path':str(audit_path.resolve()),'batch_audit_sha256':file_sha256(audit_path),
        'slurm_terminal_path':str(slurm_path.resolve()),'training_log_path':str(log_path.resolve()),
        'allocation':allocation,'batches':metrics,'optimizer_steps':sum(m['optimizer_steps'] for m in metrics),
        'request_accounting':accounting,'checkpoints':checkpoints,'saved_states':states,
        'artifact_sha256':{str(p.resolve()):file_sha256(p) for p in artifacts},'result_source_sha256':file_sha256(SOURCE),
        'finalizer_source_sha256':{n:file_sha256(Path(n)) for n in SOURCES},'new_model_calls_by_finalizer':0,
        'original_preflight_batch_ids_match':True,'deviation':None,'preflight_order_contract':CONTRACT,
        'limitations':['Independent continuous weight-only training; neither a phase reset nor a shared joint prefix.',
            'Native saved model/extra/data states are retained; Adam moments are not checkpointed.',
            'Saved-state verification does not inspect in-place rollout receiver values.',
            'Empty batches and unchanged parameters do not establish or refute heldout performance.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','audit','slurm','log','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing continuation completion evidence')
    result=complete(args.plan,args.audit,args.slurm,args.log);write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','seed','job_id','optimizer_steps','allocation')}),flush=True)
