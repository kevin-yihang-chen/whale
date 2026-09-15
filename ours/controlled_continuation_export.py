"""Canonical export for completed continuous weight-only seeds43/44.

Reuse the joint export implementation in a process-local context with explicit
condition, artifact kind and terminal wrapper identity. Native serialization and
parameter comparisons remain shared; no joint training label is substituted.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from . import joint_checkpoint_export as backend
from .audit_native_training_batch import require
from .controlled_continuation_result import complete
from .native_search import write_json
from .visual_task import file_sha256

SOURCES=('ours/controlled_continuation_export.py','ours/run_controlled_continuation_export.sh',
    'ours/controlled_continuation_result.py','ours/controlled_pilot_continuation.py','ours/audit_controlled_pilot.py')
WRAPPER='ours/run_controlled_continuation_export.sh'


def artifact_kind(step,suffix):
    require(type(step) is int and step==2,'Continuous final export requires native step2')
    return f'controlled_weight_only_continuation_{suffix}'


def completed_inputs(training_path,result_path):
    result=json.loads(result_path.read_text())
    require(result['kind']=='controlled_weight_only_continuation_result' and
            result['status']=='COMPLETED_NATIVE_CONTINUATION' and result['condition']=='weight_only' and
            result['seed'] in (43,44) and result['original_preflight_batch_ids_match'] and result['deviation'] is None,
            'Export requires the completed native continuation, without a borrowed seed42 deviation')
    verified=complete(training_path,Path(result['batch_audit_path']),Path(result['slurm_terminal_path']),Path(result['training_log_path']))
    require(result==verified,'Continuation completion differs from its full evidence')
    training=json.loads(training_path.read_text());initial=json.loads(Path(training['initialization_report']).read_text())
    require(initial['status']=='PASS' and initial['optimizer_steps']==0 and initial['audit']['exact_source_bf16_cast'] and
            initial['exported']==training['initial_manifest'],'Wrong common untrained initialization')
    require([c['step'] for c in result['checkpoints']]==[1,2],'Incomplete native continuous checkpoint history')
    selected=result['checkpoints'][1];audit=json.loads(Path(result['batch_audit_path']).read_text())
    sources={**training['source_sha256'],**result['artifact_sha256'],**result['finalizer_source_sha256'],
             **audit['artifact_sha256'],**audit['audit_source_sha256']}
    for checkpoint in result['checkpoints']:
        sources.update({str(Path(checkpoint['directory'])/n):h for n,h in checkpoint['artifact_sha256'].items()})
    return training,result,initial,selected,sources


@contextmanager
def export_context():
    """Single export process; never enter this context from concurrent threads."""
    original_check=backend.check
    def check(path,*,fresh=True):
        plan=original_check(path,fresh=fresh)
        training=json.loads(Path(plan['training_plan']).read_text());result=json.loads(Path(plan['training_result']).read_text())
        require(plan['seed']==training['seed']==result['seed'] and plan['seed'] in (43,44) and
                plan['condition']==training['condition']==result['condition']=='weight_only' and
                result['kind']=='controlled_weight_only_continuation_result' and
                result['status']=='COMPLETED_NATIVE_CONTINUATION' and plan['source_job_id']==result['job_id'] and
                result['original_preflight_batch_ids_match'] and result['deviation'] is None and
                plan['resume_checkpoint']==result['checkpoints'][1] and plan['base']==training['initial_manifest'] and
                plan['optimizer_steps_through_checkpoint']==result['optimizer_steps'],
                'Mixed continuous training/export lineage')
        return plan
    notes=['Independent continuous weight-only final checkpoint, not a joint condition or shared prefix.',
        'Both original native checkpoints remain frozen; model/extra/data.pt are retained.',
        'CPU export uses the minimum GPU allocation and all allocated GPU time is charged.',
        'Zero native or BF16 changes are valid outcomes; no gain-based filtering or resampling.',
        'Full parameter conversion is checked; actual serving and heldout performance remain separate stages.']
    with patch.object(backend,'CONDITIONS',('weight_only',)),patch.object(backend,'EXPORT_WRAPPER',WRAPPER), \
         patch.object(backend,'EXPORT_DIRECTORY','continuation-checkpoint-export'), \
         patch.object(backend,'SOURCES',(*backend.SOURCES,*SOURCES)),patch.object(backend,'artifact_kind',artifact_kind), \
         patch.object(backend,'completed_inputs',completed_inputs),patch.object(backend,'check',check), \
         patch.object(backend,'optimizer_steps',lambda result,step:result['optimizer_steps']), \
         patch.object(backend,'limitations',lambda:list(notes)):
        yield backend


def prepare(path,training_path,result_path):
    with export_context() as exporter:return exporter.prepare(path,training_path,result_path)


def check(path,*,fresh=True):
    with export_context() as exporter:return exporter.check(path,fresh=fresh)


def run(path,plan):
    with export_context() as exporter:return exporter.run(path,plan)


def close(path,slurm_path):
    with export_context() as exporter:return exporter.close(path,slurm_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','check','run','close'),required=True)
    for name in ('plan','training-plan','result','slurm','output'):parser.add_argument('--'+name,type=Path,required=name=='plan')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(args.training_plan is not None and args.result is not None,'Missing completed continuation evidence')
        prepare(args.plan,args.training_plan,args.result)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    elif args.phase=='close':
        require(args.slurm is not None and args.output is not None and not args.output.exists(),'Missing terminal evidence or occupied output')
        result=close(args.plan,args.slurm);write_json(args.output,result);print(json.dumps(result),flush=True)
    else:
        plan=check(args.plan)
        if args.phase=='run':run(args.plan,plan)
        else:print(json.dumps({'status':'PASS','plan_sha256':file_sha256(args.plan)}),flush=True)
