"""Close the frozen seed42 weight-only export after its actual Slurm terminal.

This read-only entrypoint preserves the old plan, deviation audit and exporter.
It joins their parameter/asset reports with the native training lineage; it does
not export again, change model bytes, or infer a heldout performance gain.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re

from . import controlled_checkpoint_export as original
from .audit_native_training_batch import require
from .local_completion import checkpoint_manifest
from .native_search import write_json
from .visual_task import file_sha256

SOURCE=Path(__file__)


def close(plan_path,slurm_path):
    plan=json.loads(plan_path.read_text())
    require(plan['kind']=='controlled_weight_only_export_plan' and plan['seed']==42 and plan['step']==2 and
            plan['new_model_calls']==0 and plan['original_preflight_batch_ids_match'] is False,
            'Expected the preserved seed42 final export with its explicit schedule deviation')
    require(set(original.SOURCES)<=set(plan['source_sha256']) and
            {n:version(n) for n in plan['versions']}==plan['versions'],'Changed exporter implementation or runtime')
    training_path=Path(plan['training_plan']);result_path=Path(plan['training_result'])
    training,trained,initial,selected,training_sources=original.completed_inputs(training_path,result_path,2)
    require(plan['source_job_id']==trained['job_id'] and plan['seed']==trained['seed'] and
            plan['base']==initial['exported']==training['initial_manifest'] and
            plan['native']==str(Path(selected['directory'])/'actor') and
            plan['native_checkpoint_sha256']==selected['artifact_sha256']['actor/model_world_size_1_rank_0.pt'] and
            plan['training_deviation']==trained['deviation'] and
            plan['optimizer_steps_through_checkpoint']==trained['optimizer_steps'],
            'Export differs from completed continuous training')
    require(all(plan['source_sha256'].get(n)==h for n,h in training_sources.items()),'Incomplete training provenance in export plan')
    for name,digest in plan['source_sha256'].items():
        require(file_sha256(Path(name))==digest,f'Changed frozen export input: {name}')
    require(checkpoint_manifest(Path(plan['base']['path']))==plan['base'],'Changed common initialization')
    report_path=Path(plan['transition_report']);report=json.loads(report_path.read_text())
    require(report['kind']=='controlled_weight_only_checkpoint_transition' and report['status']=='PASS' and
            report['plan_sha256']==file_sha256(plan_path) and report['source_job_id']==plan['source_job_id'] and
            report['step']==2 and report['base']==plan['base'] and
            report['native_checkpoint']==str(Path(plan['native'])/'model_world_size_1_rank_0.pt') and
            report['native_checkpoint_sha256']==plan['native_checkpoint_sha256'] and
            report['verified_aliases']==plan['verified_aliases'] and report['export_exact_native_bf16_cast'] is True and
            report['optimizer_steps_through_checkpoint']==plan['optimizer_steps_through_checkpoint'] and
            report['training_deviation']==plan['training_deviation'] and report['original_preflight_batch_ids_match'] is False,
            'Invalid preserved canonical transition')
    require(report['exported']['path']==plan['target'] and checkpoint_manifest(Path(plan['target']))==report['exported'],
            'Changed exported model or inference assets')
    require(all(report[k]['tensor_count']==plan['active_tensors'] for k in ('native_fp32','exported_bf16')),
            'Incomplete active-parameter comparison')
    root=Path(f"results/controlled-checkpoint-export-{report['job_id']}")
    start_path=root/'start.json';measurement_path=root/'measurement.json';serialization=root/'serialization.json'
    start=json.loads(start_path.read_text());measurement=json.loads(measurement_path.read_text())
    expected={k:report[k] for k in ('job_id','plan_sha256','source_job_id','native_checkpoint_sha256','new_model_calls')}
    require(start==expected and expected['new_model_calls']==0 and
            all(measurement.get(k)==v for k,v in expected.items()) and measurement['status']=='PASS' and
            measurement['transition_report']==str(report_path) and measurement['transition_report_sha256']==file_sha256(report_path) and
            report['serialization_report_sha256']==file_sha256(serialization),'Changed export execution receipts')
    fields=dict(re.findall(r'(\w+)=([^\s]+)',slurm_path.read_text()))
    require(fields['JobId']==report['job_id'] and fields['JobState']=='COMPLETED' and fields['ExitCode']=='0:0',
            'Export job did not complete successfully')
    allocated=dict(item.split('=',1) for item in fields['AllocTRES'].split(','))
    require(allocated.get('gres/gpu')==allocated.get('gres/gpu:rtx_4090')=='1' and
            Path(fields['Command']).resolve()==Path('ours/run_controlled_checkpoint_export.sh').resolve(),
            'Wrong export allocation or entrypoint')
    days,_,clock=fields['RunTime'].rpartition('-');h,m,s=map(int,clock.split(':'));seconds=int(days or 0)*86400+3600*h+60*m+s
    require(0<=seconds<=plan['resources']['time_limit_seconds'],'Export exceeded its allocation budget')
    return {'kind':'controlled_weight_only_export_result','status':'COMPLETED_CANONICAL_EXPORT',
        'condition':'weight_only','seed':42,'step':2,'job_id':report['job_id'],'source_job_id':plan['source_job_id'],
        'plan_sha256':file_sha256(plan_path),'slurm_terminal_path':str(slurm_path.resolve()),
        'transition_report':str(report_path.resolve()),'transition_report_sha256':file_sha256(report_path),
        'exported':report['exported'],'resume_checkpoint':selected,
        'optimizer_steps_through_checkpoint':plan['optimizer_steps_through_checkpoint'],
        'allocation':{'seconds':seconds,'gpu_hours':seconds/3600,'alloc_tres':fields['AllocTRES'],'state':'COMPLETED','exit_code':'0:0'},
        'artifact_sha256':{str(p.resolve()):file_sha256(p) for p in (plan_path,slurm_path,report_path,start_path,serialization,measurement_path)},
        'result_source_sha256':file_sha256(SOURCE),'new_model_calls_by_finalizer':0,
        'training_deviation':plan['training_deviation'],'original_preflight_batch_ids_match':False,
        'limitations':['The original seed42 schedule-preflight deviation is retained; native execution and full export remain separate evidence.',
            'Native model/extra/data checkpoints are unchanged. Zero parameter changes are valid recorded outcomes.',
            'This finalizer does not observe a serving GPU or measure heldout performance.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','slurm','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing export completion evidence')
    result=close(args.plan,args.slurm);write_json(args.output,result);print(json.dumps(result),flush=True)
