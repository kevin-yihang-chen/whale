"""Certify completion of the independently sampled second joint E4 phase.

Join the full trajectory audit, native update metrics, terminal allocation and
saved model/extra/sampler state. Neither learning gains nor GPU resume execution
are inferred from this record; empty updates remain valid completed outcomes.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .audit_native_training_batch import require
from .controlled_training_result import metric
from .joint_training_phase2 import check, run_name
from .joint_resume_receipts import checkpoint_layout,verify_restore_receipts,CHECKPOINT_FILES
from .native_search import write_json
from .staged_search import terminal_job
from .visual_task import file_sha256

SOURCE = Path(__file__)
SOURCES = ('ours/joint_training_phase2_result.py','ours/controlled_training_result.py',
    'ours/joint_training_phase2.py','ours/native_phase_resume.py','ours/staged_search.py',
    'ours/audit_native_training_batch.py','ours/visual_task.py','ours/native_search.py',
    'ours/joint_resume_receipts.py','ours/native_resume_observation.py','ours/audit_joint_training_phase2.py')


def phase_metrics(log, audit):
    lines=[]
    for line in log.splitlines():
        found=re.search(r'\bstep:(\d+) - online_rsft/accepted:',line)
        if found:lines.append((int(found.group(1)),line))
    require(len(lines)==1 and lines[0][0]==2,'Missing, duplicate or extra native phase summaries')
    line=lines[0][1];accepted=audit['accepted'];tokens=audit['accepted_loss_tokens']
    require(type(accepted) is int and 0<=accepted<=64 and type(tokens) is int and tokens>=0,'Invalid audited counts')
    require(metric(line,'online_rsft/total')==64 and metric(line,'online_rsft/accepted')==accepted and
            metric(line,'training/global_step')==2 and metric(line,'online_rsft/sft_epochs')==1,'Logged phase differs from audit')
    updates=metric(line,'actor/sft_updates',required=accepted>0,default=0.)
    require(updates==math.ceil(accepted/8),'Unexpected native optimizer minibatch count')
    result={'step':2,'accepted':accepted,'trajectories':64,'optimizer_steps':int(updates),'loss_tokens':tokens,
        'checkpoint_save_seconds':metric(line,'timing_s/save_checkpoint'),
        'weight_sync_seconds':metric(line,'timing_s/update_weights')}
    require(result['checkpoint_save_seconds']>=0 and result['weight_sync_seconds']>=0,'Negative completion timing')
    if accepted:
        require(tokens>0 and metric(line,'actor/sft_token_count')==tokens,'Loss-mask token accounting differs')
        result.update(loss=metric(line,'actor/sft_loss'),gradient_norm=metric(line,'actor/grad_norm'),
                      actor_seconds=metric(line,'timing_s/update_actor'))
        require(result['gradient_norm']>=0 and result['actor_seconds']>=0,'Invalid native actor metrics')
    else:
        require(tokens==0 and metric(line,'online_rsft/skipped_empty_sft')==1,'Missing native empty-update evidence')
        result['actor_seconds']=0.
    return result


def saved_state(directory, plan, updates):
    import torch
    directory=checkpoint_layout(directory,2)
    state=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    require(state['_steps_since_snapshot']==0 and not state['_iterator_finished'],'Unexpected saved loader lifecycle')
    main=state['_snapshot']['_main_snapshot'];sampled=main['_sampler_iter_state']['sampler_iter_state']
    observed={'sampler_yielded':sampled['yielded'],
        'generator_sha256':hashlib.sha256(sampled['generator'].numpy().tobytes()).hexdigest(),
        'base_seed':main['_base_seed']}
    require(observed==plan['native_dataset_order']['checkpoint_sampler_snapshots'][1],
            'Saved sampler differs from the actual native preflight')
    extra=torch.load(directory/'actor/extra_state_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
    require(set(extra)=={'lr_scheduler','rng'} and set(extra['rng'])=={'cpu','cuda','numpy','random'},
            'Incomplete native scheduler or RNG state')
    # Native advances the scheduler once per nonempty batch, after all SFT minibatches.
    incoming=torch.load(Path(plan['resume_checkpoint']['directory'])/'actor/extra_state_world_size_1_rank_0.pt',
                        map_location='cpu',weights_only=False)['lr_scheduler']
    require(incoming['last_epoch'] in (0,1),'Unexpected incoming scheduler epoch')
    expected_epoch=incoming['last_epoch']+int(updates>0)
    require(extra['lr_scheduler']['last_epoch']==expected_epoch and
            extra['lr_scheduler']['_step_count']==incoming['_step_count']+int(updates>0) and
            extra['lr_scheduler']['base_lrs']==extra['lr_scheduler']['_last_lr']==[1e-7],
            'Saved scheduler differs from the native constant-LR phase')
    require(all(extra['rng'][key] is not None for key in ('cpu','cuda','numpy','random')),'Missing saved RNG values')
    return {'sampler':observed,'incoming_scheduler_last_epoch':incoming['last_epoch'],'scheduler_last_epoch':expected_epoch,'rng_keys':sorted(extra['rng']),
        'model_and_extra_saved':True,'adam_moments_saved':False,'gpu_restore_executed_by_finalizer':False}


def complete(plan_path,audit_path,slurm_path,log_path):
    from omegaconf import OmegaConf
    plan=check(plan_path);audit=json.loads(audit_path.read_text());job=audit['job_id']
    require(audit['kind']=='controlled_joint_phase2_batch_audit' and audit['status']=='PASS' and
            audit['phase']==2 and audit['condition']==plan['condition'] and audit['seed']==plan['seed'] and
            audit['plan_sha256']==file_sha256(plan_path) and audit['trajectories']==64 and audit['new_model_calls']==0,
            'Wrong completed joint-phase audit')
    require(audit['puzzle_ids']==plan['native_dataset_order']['planned_batch_ids'][1], 'Wrong audited data order')
    rows=audit['trajectory_checks'];accounting=audit['request_accounting']
    require(len(rows)==64 and {r['row'] for r in rows}==set(range(64)) and
            Counter(r['puzzle_id'] for r in rows)==Counter({key:8 for key in audit['puzzle_ids']}) and
            sum(r['accepted'] for r in rows)==audit['accepted'] and
            sum(r['assistant_reencoded_tokens'] for r in rows if r['accepted'])==audit['accepted_loss_tokens'],
            'Incomplete or inconsistent trajectory audit coverage')
    require(accounting['observed_requests_fully_accounted'] and
            accounting['recorded_starts']==accounting['completed']==sum(r['calls'] for r in rows) and
            all(accounting[k]==0 for k in ('errors','pending_or_interrupted','partial_tail_records','completed_without_usage')) and
            accounting['recorded_starts']<=plan['max_policy_calls'] and
            0<=accounting['known_completion_tokens']<=plan['max_assistant_output_tokens'],
            'Incomplete or over-budget request accounting')
    require(set(audit['audit_source_sha256'])=={'ours/audit_joint_training_phase2.py','ours/joint_training_phase2.py',
            'ours/audit_alternation_training.py','ours/audit_native_training_batch.py','ours/native_training_trace.py',
            'ours/joint_resume_receipts.py','ours/native_resume_observation.py','ours/native_phase_resume.py'},
            'Missing original audit source bindings')
    for name,digest in {**audit['audit_source_sha256'],**audit['artifact_sha256']}.items():
        require(file_sha256(Path(name))==digest,f'Changed completed evidence: {name}')
    raw=slurm_path.read_text();allocation=terminal_job(raw,job)
    require(allocation is not None and allocation['seconds']<=plan['resources']['time_limit_seconds'],
            'Incomplete or over-budget training allocation')
    fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
    require(Path(fields['StdOut']).resolve()==log_path.resolve() and
            Path(fields['Command']).resolve()==Path('ours/run_joint_training_phase2.sh').resolve(),
            'Slurm log or training entrypoint identity differs')
    start_root=Path(f'results/controlled-joint-phase2-{job}')
    start_path=start_root/'start.json';effective=start_root/'effective-config.yaml'
    require(str(start_path) in audit['artifact_sha256'],'Missing audited launch receipt')
    restoration=verify_restore_receipts(plan,plan_path,job,start_root)
    require(audit['restoration']==restoration and audit['harness_sha256']==plan['harness_sha256'] and
            audit['resume_checkpoint']==plan['resume_checkpoint'],'Changed audited native restoration')
    start=json.loads(start_path.read_text())
    require(start['entrypoint_source_sha256']==file_sha256(Path('ours/joint_training_phase2.py')) and
            start['plan_sha256']==file_sha256(plan_path) and start['run_name']==run_name(plan,job) and
            start['resume_checkpoint']==plan['resume_checkpoint'] and start['harness_sha256']==plan['harness_sha256'],
            'Changed training launch identity')
    def config(text):return OmegaConf.to_container(OmegaConf.create(text[text.index('model_engine: dp\n'):]),resolve=True)
    frozen=config(Path(plan['resolved_config']).read_text().replace(plan['cpu_name'],run_name(plan,job)))
    require(config(effective.read_text())==frozen,'Actual launch configuration differs from the frozen plan')
    root=Path(frozen['trainer']['default_local_dir'])
    require(root.name==run_name(plan,job) and sorted(p.name for p in root.glob('global_step_*'))==['global_step_2'],
            'Missing or extra joint checkpoints')
    require((root/'latest_checkpointed_iteration.txt').read_text().strip()=='2','Wrong final checkpoint tracker')
    metrics=phase_metrics(log_path.read_text(),audit);directory=root/'global_step_2'
    state=saved_state(directory,plan,metrics['optimizer_steps'])
    names=CHECKPOINT_FILES
    checkpoint={'step':2,'directory':str(directory.resolve()),'artifact_sha256':{n:file_sha256(directory/n) for n in names}}
    search=json.loads(Path(plan['incoming_search']['search_plan']).read_text())
    prior_steps=search['joint_handoff']['optimizer_steps']
    require(type(prior_steps) is int and 0<=prior_steps<=8,'Invalid preceding native update count')
    artifacts=(plan_path,audit_path,slurm_path,log_path,start_path,effective,root/'latest_checkpointed_iteration.txt')
    return {'kind':'controlled_joint_phase2_training_result','status':'COMPLETED_NATIVE_PHASE',
        'condition':plan['condition'],'seed':plan['seed'],'phase':2,'job_id':job,'plan_sha256':file_sha256(plan_path),
        'batch_audit_path':str(audit_path.resolve()),'batch_audit_sha256':file_sha256(audit_path),
        'slurm_terminal_path':str(slurm_path.resolve()),'training_log_path':str(log_path.resolve()),
        'allocation':allocation,'batches':[metrics],'optimizer_steps':metrics['optimizer_steps'],
        'request_accounting':audit['request_accounting'],'checkpoints':[checkpoint],'saved_state':state,
        'restoration':restoration,'incoming_search':plan['incoming_search'],'harness_sha256':plan['harness_sha256'],
        'phase1_optimizer_steps':prior_steps,'total_joint_optimizer_steps':prior_steps+metrics['optimizer_steps'],
        'artifact_sha256':{str(p.resolve()):file_sha256(p) for p in artifacts},
        'result_source_sha256':file_sha256(SOURCE),
        'finalizer_source_sha256':{n:file_sha256(Path(n)) for n in SOURCES},'new_model_calls_by_finalizer':0,
        'original_preflight_batch_ids_match':True,
        'limitations':['Independent joint second phase after its complete MH search; not a reused weight-only prefix.',
            'Source-bound GPU actor and pending data restoration receipts were checked; the finalizer does not execute GPU loading.',
            'The in-place vLLM receiver was not independently inspected. Canonical export and final inference remain separate stages.',
            'Training acceptance and optimizer steps do not imply heldout or VETO improvement.']}



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','audit','slurm','log','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing completion evidence')
    report=complete(args.plan,args.audit,args.slurm,args.log);write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','condition','seed','job_id','optimizer_steps','allocation')}),flush=True)
