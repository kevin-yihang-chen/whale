"""Read certified heldout results into a complete twelve-slot descriptive table."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import re
import statistics

from .audit_native_training_batch import require
from .controlled_heldout import close
from .heldout_cohort import CONDITIONS,SEEDS,check as check_cohort
from .native_search import write_json
from .visual_task import file_sha256

STAGES=('training','search','export','heldout')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def allocation_record(path):
    fields=dict(re.findall(r'(\w+)=([^\s]+)',path.read_text()))
    require(fields['JobState'] in TERMINAL,'Only terminal allocations enter known cost totals')
    allocated=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    days,_,clock=fields['RunTime'].rpartition('-');h,m,s=map(int,clock.split(':'))
    seconds=int(days or 0)*86400+3600*h+60*m+s
    gpus=int(allocated.get('gres/gpu',0))
    require(seconds>=0 and gpus>=0 and (seconds==0 or allocated),'Missing terminal allocation information')
    return {'job_id':fields['JobId'],'state':fields['JobState'],'exit_code':fields['ExitCode'],
        'seconds':seconds,'gpus':gpus,'gpu_hours':gpus*seconds/3600,
        'source':str(path.resolve()),'source_sha256':file_sha256(path)}


def descriptive(values):
    complete=all(value is not None for value in values)
    return {'completed_seeds':sum(v is not None for v in values),'required_seeds':3,
        'mean':statistics.mean(values) if complete else None,
        'sample_sd':statistics.stdev(values) if complete else None,'complete':complete}


def summarize(rows):
    lookup={(r['condition'],r['seed']):r for r in rows}
    require(len(rows)==12 and set(lookup)=={(c,s) for c in CONDITIONS for s in SEEDS},'Expected twelve unique trial slots')
    conditions=[]
    for condition in CONDITIONS:
        selected=[lookup[condition,s] for s in SEEDS]
        conditions.append({'condition':condition,'seeds':list(SEEDS),
            'solved':[r['solved'] for r in selected],
            'accuracy':descriptive([r['accuracy'] for r in selected]),
            'calls_per_puzzle':descriptive([r['calls']/64 if r['calls'] is not None else None for r in selected]),
            'tokens_per_puzzle':descriptive([r['generated_tokens']/64 if r['generated_tokens'] is not None else None for r in selected])})
    paired=[]
    for treatment in ('whale_fst','whale'):
        for control in ('weight_only','harness_only'):
            differences=[lookup[treatment,s]['accuracy']-lookup[control,s]['accuracy']
                if lookup[treatment,s]['accuracy'] is not None and lookup[control,s]['accuracy'] is not None else None for s in SEEDS]
            paired.append({'treatment':treatment,'control':control,'seeds':list(SEEDS),
                'accuracy_differences':differences,**descriptive(differences)})
    return conditions,paired


def build(cohort_path,index_path):
    cohort=check_cohort(cohort_path);index=json.loads(index_path.read_text())
    require(set(index)=={'evaluations','allocation_registry'},'Wrong heldout comparison inputs')
    entries=index['evaluations'];keys={(c,s) for c in CONDITIONS for s in SEEDS}
    require(len(entries)==12 and {(e['condition'],e['seed']) for e in entries}==keys,'Declare all twelve evaluation outcomes')
    trials={(t['condition'],t['seed']):t for t in cohort['trials']};rows=[];sources={str(cohort_path.resolve()):file_sha256(cohort_path),
        str(index_path.resolve()):file_sha256(index_path)};allocations={}
    def add_allocation(condition,seed,stage,path):
        require((condition,seed) in keys and stage in STAGES,'Unrecognized allocation attribution')
        record=allocation_record(path);record.update(condition=condition,seed=seed,stage=stage)
        previous=allocations.get(record['job_id'])
        require(previous is None or previous==record,'Conflicting or duplicate job attribution')
        allocations[record['job_id']]=record;sources[str(path.resolve())]=record['source_sha256']
    for entry in sorted(entries,key=lambda e:(CONDITIONS.index(e['condition']),e['seed'])):
        condition,seed=entry['condition'],entry['seed'];trial=trials[condition,seed]
        row={'condition':condition,'seed':seed,'status':entry['status'],'trial_status':trial['status'],
            'solved':None,'examples':64,'accuracy':None,'calls':None,'generated_tokens':None,
            'provenance':trial.get('provenance'),'missing_reason':None}
        if entry['status']=='COMPLETE':
            require(set(entry)=={'condition','seed','status','result'} and trial['evaluable'],'Unfinished trial cannot have a completed score')
            path=Path(entry['result']);result=json.loads(path.read_text());plan_path=Path(result['plan'])
            verified=close(plan_path,Path(result['slurm_terminal_path']))
            require(verified==result and result['condition']==condition and result['seed']==seed and
                result['cohort_sha256']==file_sha256(cohort_path) and result['target_manifest']==trial['model'] and
                result['harness_sha256']==trial['harness_sha256'],'Mixed or unverified heldout result')
            row.update(solved=result['solved'],accuracy=result['success_rate'],calls=result['calls'],
                generated_tokens=result['generated_tokens'],result=str(path.resolve()),result_sha256=file_sha256(path))
            sources.update({str(path.resolve()):file_sha256(path),str(plan_path.resolve()):file_sha256(plan_path)})
            add_allocation(condition,seed,'heldout',Path(result['slurm_terminal_path']))
        else:
            require(entry['status']=='INCOMPLETE' and set(entry)=={'condition','seed','status','reason','evidence_paths'} and
                isinstance(entry['reason'],str) and entry['reason'].strip() and entry['evidence_paths'],
                'Missing evaluation needs a reason and evidence, with no imputed score')
            row['missing_reason']=entry['reason']
            sources.update({str(Path(p).resolve()):file_sha256(Path(p)) for p in entry['evidence_paths']})
        rows.append(row)
    for entry in index['allocation_registry']:
        require(set(entry)=={'condition','seed','stage','slurm_terminal'},'Wrong allocation registry row')
        add_allocation(entry['condition'],entry['seed'],entry['stage'],Path(entry['slurm_terminal']))
    for row in rows:
        selected=[a for a in allocations.values() if (a['condition'],a['seed'])==(row['condition'],row['seed'])]
        row['known_gpu_hours_by_stage']={stage:sum(a['gpu_hours'] for a in selected if a['stage']==stage)
            if any(a['stage']==stage for a in selected) else None for stage in STAGES}
        row['failed_terminal_allocations']=sum(a['state']!='COMPLETED' or a['exit_code']!='0:0' for a in selected)
    conditions,paired=summarize(rows)
    return {'kind':'controlled_heldout_comparison','status':'COMPLETE_TWELVE_SCORES' if all(r['status']=='COMPLETE' for r in rows) else 'INCOMPLETE_COMPARISON',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'cohort_sha256':file_sha256(cohort_path),
        'trials':rows,'conditions':conditions,'paired_seed_differences':paired,
        'known_terminal_allocations':list(allocations.values()),'source_sha256':sources,
        'report_source_sha256':file_sha256(Path(__file__)),'new_model_calls':0,'new_api_calls':0,
        'test_task_reader_used':False,
        'limitations':['Means and sample SD are defined only for all three seeds; missing scores are never imputed.',
            'Paired differences are descriptive, with no significance or mechanism-attribution claim.',
            'Cost attribution follows the declared registry; terminal costs include failures but do not prove every attempt was listed.',
            'Cost cells are known allocation subtotals, not full project totals; absent evidence is null rather than zero.',
            'API accounting and shared setup allocations are reported separately from these per-trial GPU subtotals.',
            'The full report rechecks saved finalizer evidence; it neither reads task rows nor generates replies.']}


def markdown(report):
    lines=['# Controlled Chess heldout comparison','',report['status'],'',
        '| Condition | Seed42 | Seed43 | Seed44 | Mean ± sample SD (%) |',
        '|---|---:|---:|---:|---:|']
    for row in report['conditions']:
        values=[f'{n}/64' if n is not None else 'missing' for n in row['solved']];stats=row['accuracy']
        value=f"{100*stats['mean']:.2f} ± {100*stats['sample_sd']:.2f}" if stats['complete'] else 'incomplete'
        lines.append('| '+' | '.join([row['condition'],*values,value])+' |')
    lines.extend(['','Paired per-seed differences and known allocation subtotals are in the JSON/CSV.',
        'Missing scores have no aggregate; cost coverage is limited to the listed terminal records.',''])
    for row in report['trials']:
        if row['missing_reason']:lines.append(f"- {row['condition']} / seed{row['seed']}: {row['missing_reason']}")
    return '\n'.join(lines)+'\n'


def export(report,output):
    require(not output.exists(),'Preserve an existing heldout comparison');output.mkdir(parents=True)
    write_json(output/'comparison.json',report);(output/'comparison.md').write_text(markdown(report))
    columns=['condition','seed','status','trial_status','solved','examples','accuracy','calls','generated_tokens',
        'missing_reason','failed_terminal_allocations',*[stage+'_known_gpu_hours' for stage in STAGES]]
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in report['trials']:
        record={k:row[k] for k in columns if k in row}
        record.update({stage+'_known_gpu_hours':value for stage,value in row['known_gpu_hours_by_stage'].items()});writer.writerow(record)
    (output/'trials.csv').write_text(stream.getvalue())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','index','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=build(args.cohort,args.index);export(report,args.output)
    print(json.dumps({'status':report['status'],'completed':sum(t['status']=='COMPLETE' for t in report['trials']),
        'new_model_calls':0,'test_task_reader_used':False}),flush=True)
