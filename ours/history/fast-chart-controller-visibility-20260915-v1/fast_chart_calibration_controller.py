"""Sequential h0 -> three native LR stages -> V calibration, with no test access."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .fast_chart_protocol import ROOT,OUTPUT,CompactExecutionBudget,storage_check
from .fast_chart_training import RATES,read
from .native_visual_service import write_new
from .visual_task import file_sha256


def choose_rate(rows):
    if set(rows)!={str(rate) for rate in RATES}:
        raise ValueError('All three prespecified learning rates must finish')
    for rate,row in rows.items():
        if row['status']!='COMPLETE_COMPACT_NATIVE_FOLLOWUP' or not 0<=row['marginal_accuracy']<=1:
            raise ValueError('Incomplete learning-rate measurement')
    return min(RATES,key=lambda rate:(-rows[str(rate)]['marginal_accuracy'],rate))


def terminal(plan_path):
    directory=OUTPUT/'allocations'/Path(plan_path).stem
    if not (directory/'submission.json').exists():return None
    if not (directory/'allocation-result.json').exists():return 'PENDING'
    record=read(directory/'allocation-result.json')
    if record['state']!='COMPLETED':
        raise RuntimeError(f"Allocation {record['job_id']} ended {record['state']}; no automatic retry")
    return 'COMPLETED'


def command(directory,name,args):
    path=directory/(name+'.log')
    with path.open('x') as stream:
        subprocess.run([sys.executable,'-m',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def verify_recovery(new_plan,manifest):
    record=read(manifest)
    if record['kind']!='compact_native_transport_recovery' or record['retry_phase']!='retry':
        raise ValueError('Wrong recovery authorization scope')
    for name,digest in record['evidence_sha256'].items():
        if file_sha256(Path(name))!=digest:raise ValueError('Preserved failed-run evidence changed')
    old_path=Path(record['failed_training_plan']);old,new=read(old_path),read(new_plan)
    for key in ('seed','rate','stage','harness_sha256','h0_calibration','parent_plan','selection','resume_checkpoint',
                'training_data','cpu_preflight','augmentation','model','worker_coordinates','bounds'):
        if old[key]!=new[key]:raise ValueError(f'Retry changed the frozen experimental setting: {key}')
    def normalized(plan,path):
        return json.dumps(plan['config'],sort_keys=True).replace(str(path),'<PLAN>').replace(plan['output'],'<OUTPUT>')
    if normalized(old,old_path)!=normalized(new,Path(new_plan)):
        raise ValueError('Retry changed configuration beyond output and plan paths')
    return record


def advance(directory,h0_plan,recovery=None):
    h0=read(h0_plan); calibration=Path(h0['output'])/'result.json'
    if terminal(h0_plan)!='COMPLETED':return {'status':'WAITING_INITIAL_HARNESS_CALIBRATION'}
    if read(calibration)['status']!='COMPLETE_SHARED_H0_CALIBRATION':raise ValueError('Incomplete h0 calibration')
    results={}
    for rate in RATES:
        name=f'lr-{rate:g}-seed42'
        train_plan=directory/(directory.name+'-'+name+'-training-plan.json')
        if not train_plan.exists():
            (directory/'controller-status.json').write_text(json.dumps({'status':'PREPARING_LR_TRAINING','rate':rate})+'\n')
            command(directory,name+'-prepare',['ours.fast_chart_training','prepare','--plan',str(train_plan),
                '--output',str(directory/name/'training'),'--calibration',str(calibration),'--seed','42','--rate',str(rate)])
        is_retry=recovery is not None and rate==RATES[0]
        if is_retry:verify_recovery(train_plan,recovery)
        state=terminal(train_plan)
        if state is None:
            command(directory,name+'-submit',['ours.fast_chart_submit','--kind','training','--phase',
                'retry' if is_retry else 'setup','--plan',str(train_plan)])
            return {'status':'SUBMITTED_LR_TRAINING','rate':rate,'plan':str(train_plan)}
        if state=='PENDING':return {'status':'WAITING_LR_TRAINING','rate':rate}
        followup=directory/(directory.name+'-'+name+'-followup-plan.json')
        if not followup.exists():
            command(directory,name+'-followup-prepare',['ours.fast_chart_followup','prepare','--plan',str(followup),
                '--output',str(directory/name/'followup'),'--training-plan',str(train_plan)])
        state=terminal(followup)
        if state is None:
            command(directory,name+'-followup-submit',['ours.fast_chart_submit','--kind','followup','--phase','setup','--plan',str(followup)])
            return {'status':'SUBMITTED_LR_DEVELOPMENT','rate':rate}
        if state=='PENDING':return {'status':'WAITING_LR_DEVELOPMENT','rate':rate}
        record_path=directory/name/'followup/result.json'
        result=read(record_path)
        if result['plan_sha256']!=file_sha256(followup):raise ValueError('LR result identity differs')
        results[str(rate)]={**result,'result_path':str(record_path),'result_sha256':file_sha256(record_path),
            'training_plan':str(train_plan),'training_plan_sha256':file_sha256(train_plan)}
    rate=choose_rate(results)
    final={'status':'COMPLETE_SHARED_CHART_CONFIGURATION','h0_calibration':str(calibration),
        'h0_calibration_sha256':file_sha256(calibration),'harness':read(calibration)['harness'],
        'learning_rate':rate,'results':results,'selection':'Ordinary V accuracy; ties choose smaller LR.',
        'reusable_seed42_common_prefix':results[str(rate)]['training_plan'],
        'scientific_method_verified':False,'formal_expansion_launched':False,
        'budget':CompactExecutionBudget().compact_summary(),'storage':storage_check(),
        'next_gate':'Forecast all18 conditions from measured allocation costs before formal expansion.'}
    write_new(directory/'result.json',final)
    return final


def run(directory,h0_plan,recovery=None):
    directory=Path(directory).resolve();h0_plan=Path(h0_plan).resolve()
    if not directory.is_relative_to(OUTPUT):raise ValueError('Controller must stay under compact output')
    recovery=Path(recovery).resolve() if recovery else None
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_new(directory/'controller-start.json',{'pid':os.getpid(),'h0_plan':str(h0_plan),
            'h0_plan_sha256':file_sha256(h0_plan),'source_sha256':file_sha256(Path(__file__)),
            'recovery':str(recovery) if recovery else None,'recovery_sha256':file_sha256(recovery) if recovery else None,
            'scope':'Three common4-batch LR calibrations and V followups; no proposer, test, cleanup or formal expansion.'})
        try:
            while not (directory/'result.json').exists():
                report=advance(directory,h0_plan,recovery)
                print(json.dumps(report if 'results' not in report else {'status':report['status'],'learning_rate':report['learning_rate']}),flush=True)
                (directory/'controller-status.json').write_text(json.dumps(report,indent=2)+'\n')
                if report['status']=='COMPLETE_SHARED_CHART_CONFIGURATION':break
                time.sleep(30)
        except BaseException as exc:
            failure={'status':'STOPPED_REQUIRES_DIAGNOSIS',
                'error_type':type(exc).__name__,'error':str(exc),'no_automatic_retry':True}
            write_new(directory/'controller-failure.json',failure)
            (directory/'controller-status.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--h0-plan',type=Path,required=True);p.add_argument('--recovery',type=Path)
    a=p.parse_args();run(a.output,a.h0_plan,a.recovery)
