"""Seal all final trial identities before any heldout task is loaded.

Completed entries are independently recertified from native search or canonical
training/export evidence. Incomplete entries carry a declared reason and their
existing evidence only; they have no score or evaluable model/harness pair.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .native_search import write_json
from .search_completion import verify_completed_search
from .visual_task import file_sha256

PROTOCOL=Path('results/controlled-heldout-protocol-20260910-v1.json')
CONDITIONS=('weight_only','harness_only','whale_fst','whale')
SEEDS=(42,43,44)
SOURCES=('ours/heldout_cohort.py','ours/search_completion.py','ours/staged_search.py',
    'ours/controlled_checkpoint_export_result.py','ours/controlled_checkpoint_export.py',
    'ours/controlled_continuation_export.py','ours/controlled_continuation_result.py',
    'ours/joint_checkpoint_export.py','ours/joint_training_phase2_result.py','ours/joint_training_phase2.py',
    'ours/local_completion.py','ours/visual_task.py')


def read_protocol(path=PROTOCOL):
    protocol=json.loads(path.read_text())
    require(protocol['kind']=='controlled_chess_heldout_protocol' and
            protocol['status']=='FROZEN_BEFORE_TEST_TASK_INSPECTION' and protocol['run_ready'] is False and
            protocol['conditions']==list(CONDITIONS) and protocol['seeds']==list(SEEDS) and
            protocol['data_role']=='test' and protocol['examples']==64 and protocol['logical_shards']==4,
            'Wrong frozen heldout protocol')
    for name in ('protocol','parent_protocol','split_manifest'):
        require(file_sha256(Path(protocol[name]))==protocol[name+'_sha256'],'Changed heldout protocol dependency')
    split=json.loads(Path(protocol['split_manifest']).read_text())['splits'];ids=protocol['puzzle_ids']
    require(ids==split['test']['puzzle_ids'] and len(set(ids))==64 and protocol['dataset']==split['test']['path'] and
            protocol['dataset_sha256']==split['test']['sha256'] and
            not set(ids)&(set(split['train']['puzzle_ids'])|set(split['mh_val']['puzzle_ids'])) and
            protocol['shard_ids']==[sorted(ids)[i::4] for i in range(4)],'Wrong heldout data roles or assignment')
    # Opaque bytes only: do not call a dataset reader during preparation.
    require(file_sha256(Path(protocol['dataset']))==protocol['dataset_sha256'],'Changed heldout data bytes')
    return protocol


def absolute_sources(*maps):
    combined={}
    for mapping in maps:
        for name,digest in mapping.items():
            key=str(Path(name).resolve())
            require(key not in combined or combined[key]==digest,'Conflicting provenance for one artifact')
            combined[key]=digest
    return combined


def certify_trial(entry):
    condition,seed=entry['condition'],entry['seed']
    require(condition in CONDITIONS and type(seed) is int and seed in SEEDS,'Outside the twelve declared trials')
    require(entry['status'] in ('COMPLETE','INCOMPLETE'),'Unknown final trial status')
    if entry['status']=='INCOMPLETE':
        require(set(entry)=={'condition','seed','status','reason','evidence_paths'} and
                isinstance(entry['reason'],str) and bool(entry['reason'].strip()) and bool(entry['evidence_paths']),
                'Incomplete trials need a declared reason and preserved evidence, without a model or score')
        sources={str(Path(p).resolve()):file_sha256(Path(p)) for p in entry['evidence_paths']}
        return {'condition':condition,'seed':seed,'status':'INCOMPLETE','declared_reason':entry['reason'],
                'dependency_sha256':sources,'evaluable':False,'new_model_calls':0}
    common_path=Path('results/canonical-initialization-20260910.json')
    common=json.loads(common_path.read_text())['exported'];references=[]
    if condition=='harness_only':
        require(set(entry)=={'condition','seed','status','search_plan'},'Wrong final fixed-weight trial inputs')
        path=Path(entry['search_plan']);plan=json.loads(path.read_text());proof=verify_completed_search(path)
        require(plan['kind']=='controlled_staged_mh_plan' and plan['condition']==proof['condition']==condition and
                plan['seed']==proof['seed']==seed and plan['target_manifest']==common,'Mixed fixed-weight final lineage')
        model=plan['target_manifest'];harness=Path(proof['harness']);harness_sha=proof['harness_sha256']
        sources=absolute_sources(plan['source_sha256'],proof['artifact_sha256'],proof['source_sha256'])
        sources.update({str((Path(plan['phase_root'])/'search'/n).resolve()):h for n,h in proof['search_sha256'].items()})
        references=[path,Path(proof['result_path'])]
        provenance={'stage':'completed_native_search','accepted':proof['accepted'],'completed_rounds':proof['completed_rounds'],
            'native_perfect_stop':proof['native_perfect_stop'],'search_result_sha256':proof['result_sha256'],
            'optimizer_steps':0,'original_preflight_batch_ids_match':True}
    else:
        require(set(entry)=={'condition','seed','status','export_plan','export_result'},'Wrong final trained trial inputs')
        export_path,result_path=Path(entry['export_plan']),Path(entry['export_result'])
        export=json.loads(export_path.read_text());result=json.loads(result_path.read_text())
        if condition=='weight_only' and seed==42:
            from .controlled_checkpoint_export_result import close
            from .controlled_checkpoint_export import completed_inputs
            inputs=completed_inputs(Path(export['training_plan']),Path(export['training_result']),2)
        elif condition=='weight_only':
            from .controlled_continuation_export import close,completed_inputs
            inputs=completed_inputs(Path(export['training_plan']),Path(export['training_result']))
        else:
            from .joint_checkpoint_export import close,completed_inputs
            inputs=completed_inputs(Path(export['training_plan']),Path(export['training_result']))
        verified=close(export_path,Path(result['slurm_terminal_path']))
        require(result==verified,'Final export result differs from complete evidence')
        training,trained,initial,selected,training_sources=inputs
        require(training['condition']==trained['condition']==result['condition']==condition and
                training['seed']==trained['seed']==result['seed']==seed and
                result['step']==selected['step']==2 and result['resume_checkpoint']==selected and
                initial['exported']==training['initial_manifest']==common and export['base']==common,
                'Mixed final training condition, seed, phase or initialization')
        if condition in ('whale','whale_fst'):
            require(trained['kind']=='controlled_joint_phase2_training_result' and trained['phase']==2 and
                    trained['harness_sha256']==training['harness_sha256'],'Joint final trial lacks its second native phase')
        harness=Path(training['harness']);harness_sha=file_sha256(harness);model=result['exported']
        if condition=='weight_only':
            from .staged_search import BASELINE
            require(harness.resolve()==BASELINE.resolve(),'Weight-only final harness is not original h0')
        sources=absolute_sources(training_sources,export['source_sha256'],result['artifact_sha256'])
        references=[export_path,result_path,Path(export['training_plan']),Path(export['training_result'])]
        provenance={'stage':'completed_native_training_and_export','native_checkpoint':selected,
            'training_job':trained['job_id'],'export_job':result['job_id'],
            'optimizer_steps':result['optimizer_steps_through_checkpoint'],
            'original_preflight_batch_ids_match':trained['original_preflight_batch_ids_match'],
            'training_deviation':trained.get('deviation')}
    require(file_sha256(harness)==harness_sha and checkpoint_manifest(Path(model['path']))==model,
            'Final model or harness bytes changed')
    references.extend((common_path,harness));sources.update({str(p.resolve()):file_sha256(p) for p in references})
    return {'condition':condition,'seed':seed,'status':'COMPLETE','evaluable':True,
        'model':model,'initial_manifest':common,'harness':str(harness.resolve()),'harness_sha256':harness_sha,
        'provenance':provenance,'dependency_sha256':sources,'new_model_calls':0}


def seal(input_path,output_path):
    require(not output_path.exists(),'Preserve an existing final trial cohort')
    protocol=read_protocol();entries=json.loads(input_path.read_text())
    expected={(c,s) for c in CONDITIONS for s in SEEDS}
    require(isinstance(entries,list) and len(entries)==12 and
            {(e['condition'],e['seed']) for e in entries}==expected,'Every declared trial must appear exactly once')
    records=[certify_trial(e) for e in sorted(entries,key=lambda e:(CONDITIONS.index(e['condition']),e['seed']))]
    report={'kind':'controlled_final_trial_cohort','status':'SEALED_FINAL_TRIAL_IDENTITIES',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':str(PROTOCOL.resolve()),
        'protocol_sha256':file_sha256(PROTOCOL),'input':str(input_path.resolve()),'input_sha256':file_sha256(input_path),
        'trials':records,'source_sha256':{str(Path(n).resolve()):file_sha256(Path(n)) for n in SOURCES},
        'test_tasks_loaded':False,'new_model_calls':0,'new_api_calls':0,
        'limitations':['Incomplete status is a declared exclusion with preserved evidence, not a task score or successful trial.',
            'All twelve statuses are frozen before test access; only certified completed entries are evaluable.',
            'Each evaluation must verify its selected dependencies and actual serving worker again.']}
    write_json(output_path,report);return report


def check(path,*,condition=None,seed=None,verify_dependencies=True):
    report=json.loads(path.read_text());read_protocol(Path(report['protocol']))
    require(report['kind']=='controlled_final_trial_cohort' and report['status']=='SEALED_FINAL_TRIAL_IDENTITIES' and
            report['protocol_sha256']==file_sha256(PROTOCOL) and report['test_tasks_loaded'] is False and
            report['new_model_calls']==report['new_api_calls']==0 and
            file_sha256(Path(report['input']))==report['input_sha256'],'Changed sealed cohort identity')
    require(set(report['source_sha256'])=={str(Path(n).resolve()) for n in SOURCES},'Missing cohort certifier source')
    for n,h in report['source_sha256'].items():require(file_sha256(Path(n))==h,'Changed cohort certification source')
    entries=report['trials'];require(len(entries)==12 and {(e['condition'],e['seed']) for e in entries}==
        {(c,s) for c in CONDITIONS for s in SEEDS},'Incomplete or duplicate sealed trial matrix')
    if condition is None:return report
    selected=next(e for e in entries if (e['condition'],e['seed'])==(condition,seed))
    require(selected['status']=='COMPLETE' and selected['evaluable'],'Unfinished trials cannot enter heldout inference')
    if verify_dependencies:
        for n,h in selected['dependency_sha256'].items():require(file_sha256(Path(n))==h,f'Changed selected final evidence: {n}')
        require(checkpoint_manifest(Path(selected['model']['path']))==selected['model'] and
                file_sha256(Path(selected['harness']))==selected['harness_sha256'],'Changed final model/harness identity')
    return selected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=seal(args.input,args.output)
    print(json.dumps({'status':report['status'],'complete':sum(t['evaluable'] for t in report['trials']),
        'incomplete':sum(not t['evaluable'] for t in report['trials']),'test_tasks_loaded':False}),flush=True)
