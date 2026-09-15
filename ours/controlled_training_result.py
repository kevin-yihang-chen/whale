"""Close a native two-batch trial with audited trajectories and Slurm evidence.

This records E4 execution and its costs, not parameter deltas or heldout gains.
The preflight-order deviation is retained in the top-level result status.
"""
import argparse
import json
import math
from pathlib import Path
import re

from .audit_native_training_batch import require
from .controlled_pilot import check
from .native_search import write_json
from .staged_search import terminal_job
from .visual_task import file_sha256


def metric(line,key,*,required=True,default=None):
    match=re.search(r'(?:^| - )'+re.escape(key)+r':(?:np\.(?:float64|float32|int32|int64)\()?([^\s)]+)',line)
    if not match:
        require(not required,f'Missing native metric: {key}')
        return default
    value=float(match.group(1));require(math.isfinite(value),f'Nonfinite native metric: {key}')
    return value


def batch_metrics(log,batches):
    lines={}
    for line in log.splitlines():
        found=re.search(r'\bstep:(\d+) - online_rsft/accepted:',line)
        if found:
            step=int(found.group(1));require(step not in lines,'Duplicate native batch summary')
            lines[step]=line
    require(set(lines)=={1,2},'Missing or extra native batch summaries')
    result=[]
    for batch in batches:
        step=batch['step'];line=lines[step];accepted=batch['accepted']
        require(metric(line,'online_rsft/total')==64 and metric(line,'online_rsft/accepted')==accepted,
                'Logged accepted count differs from independent audit')
        require(metric(line,'training/global_step')==step and metric(line,'online_rsft/sft_epochs')==1,
                'Native step or update epochs differ')
        updates=metric(line,'actor/sft_updates',required=accepted>0,default=0.)
        require(updates==math.ceil(accepted/8),'Unexpected original SFT minibatch count')
        fields={'step':step,'accepted':accepted,'trajectories':64,'optimizer_steps':int(updates),
            'checkpoint_save_seconds':metric(line,'timing_s/save_checkpoint'),
            'weight_sync_seconds':metric(line,'timing_s/update_weights')}
        if accepted:
            require(metric(line,'actor/sft_token_count')==batch['accepted_loss_tokens'],'Loss-mask token count differs')
            fields.update(loss_tokens=batch['accepted_loss_tokens'],loss=metric(line,'actor/sft_loss'),
                gradient_norm=metric(line,'actor/grad_norm'),actor_seconds=metric(line,'timing_s/update_actor'))
            require(fields['gradient_norm']>0,'No positive finite gradient norm')
        else:
            require(metric(line,'online_rsft/skipped_empty_sft')==1,'Missing native empty-update evidence')
            fields.update(loss_tokens=0,actor_seconds=0.)
        result.append(fields)
    return result


def complete(plan_path,audit_path,slurm_path,log_path):
    plan=check(plan_path)
    audit=json.loads(audit_path.read_text())
    require(audit['status']=='PASS_NATIVE_EXECUTION_WITH_PREFLIGHT_DEVIATION' and
            not audit['original_frozen_plan_modified'] and not audit['original_preflight_batch_ids_match'],
            'Expected the explicitly disclosed preflight-deviation audit')
    require(audit['plan_sha256']==file_sha256(plan_path),'Different audited plan')
    native=audit['native_configuration_audit']
    require(native['status']=='PASS' and native['trajectories']==128 and native['condition']=='weight_only' and
            native['seed']==plan['seed'] and native['plan_sha256']==file_sha256(plan_path),'Wrong audited trial')
    sources={**audit['source_sha256'],**native['audit_source_sha256']}
    artifacts=native['artifact_sha256']
    for name,digest in {**sources,**artifacts}.items():
        require(file_sha256(Path(name))==digest,f'Changed completed evidence: {name}')
    allocation=terminal_job(slurm_path.read_text(),native['job_id'])
    require(allocation is not None,'Training allocation has not completed')
    metrics=batch_metrics(log_path.read_text(),native['batches'])
    root=Path(f"data/native-rsft/controlled-weight-only-seed{plan['seed']}-{native['job_id']}")
    checkpoints=[]
    for step in (1,2):
        directory=root/f'global_step_{step}'
        require(json.loads((directory/'actor/fsdp_config.json').read_text())['world_size']==1,'Wrong saved actor topology')
        require(not list((directory/'actor').glob('optim_world_size_*')),'Unexpected saved Adam moments')
        files={name:file_sha256(directory/name) for name in ('data.pt','actor/model_world_size_1_rank_0.pt',
            'actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','actor/huggingface/config.json')}
        checkpoints.append({'step':step,'directory':str(directory.resolve()),'artifact_sha256':files})
    require((root/'latest_checkpointed_iteration.txt').read_text().strip()=='2','Wrong terminal checkpoint tracker')
    return {'kind':'controlled_weight_only_training_result','status':'COMPLETED_WITH_PREFLIGHT_SCHEDULE_DEVIATION',
        'job_id':native['job_id'],'seed':plan['seed'],'condition':plan['condition'],
        'plan_sha256':file_sha256(plan_path),'batch_audit_path':str(audit_path),'batch_audit_sha256':file_sha256(audit_path),
        'allocation':allocation,'batches':metrics,'optimizer_steps':sum(m['optimizer_steps'] for m in metrics),
        'request_accounting':native['request_accounting'],'checkpoints':checkpoints,
        'artifact_sha256':{str(p):file_sha256(p) for p in (plan_path,audit_path,slurm_path,log_path)},
        'result_source_sha256':file_sha256(Path(__file__)),'new_model_calls_by_finalizer':0,
        'deviation':audit['deviation'],'original_preflight_batch_ids_match':False,
        'limitations':['Native training/checkpoint completion with disclosed diagnostic-plan deviation.',
            'Full parameter-value comparison and exported/live serving identity remain separate checks.',
            'Training successes are not heldout scores, and do not establish a VETO gain or F0 completion.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','audit','slurm','log','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve previous result')
    result=complete(args.plan,args.audit,args.slurm,args.log)
    write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','job_id','batches','optimizer_steps','allocation')}),flush=True)
