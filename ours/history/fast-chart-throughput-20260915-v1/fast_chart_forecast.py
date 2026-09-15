"""Project the full compact scope from measured work without authorizing expansion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .fast_chart_protocol import OUTPUT, CompactExecutionBudget, storage_check
from .fast_chart_admission import verified_calibration, read
from .native_visual_service import write_new
from .visual_task import file_sha256


def allocation(plan):
    digest = file_sha256(Path(plan))
    matches = [p.parent for p in (OUTPUT/'allocations').glob('*/submission.json')
        if read(p).get('plan_sha256') == digest]
    if len(matches) != 1:
        raise ValueError('Require exactly one allocation for this plan identity')
    root = matches[0]
    receipt, terminal = read(root/'submission.json'), read(root/'allocation-result.json')
    if receipt['plan_sha256'] != digest or terminal['state'] != 'COMPLETED':
        raise ValueError('Forecast requires matching completed allocation evidence')
    return float(terminal['gpu_hours'])


def allocated_gib(root):
    inodes = {}
    for path in Path(root).rglob('*'):
        if path.is_symlink():
            raise ValueError('Forecast cannot account for checkpoint symlinks')
        if path.is_file():
            stat = path.stat()
            inodes[stat.st_dev,stat.st_ino] = stat.st_blocks*512
    return sum(inodes.values())/1024**3


def evaluation_projection(loading_seconds, seconds_per_image, overhead_seconds=0):
    def hours(images):
        return (loading_seconds+seconds_per_image*images+overhead_seconds)/3600
    return {'one_V512_gpu_hours':hours(512), 'one_H128_C128_gpu_hours':hours(256),
        'one_T2048_gpu_hours':hours(2048), 'one_ChartQA512_proxy_gpu_hours':hours(512),
        'final_T_18_conditions_plus_6_common_h0_gpu_hours':24*hours(2048),
        'external_18_conditions_gpu_hours':18*hours(512),
        'independent_R_six_1024_pair_passes_gpu_hours':6*hours(2048),
        'final_evaluation_phase_proxy_gpu_hours':24*hours(2048)+18*hours(512),
        'assumptions':['Standalone loading per endpoint; no unmeasured batching speedup credited.',
            'ChartQA and future harness token lengths are unknown; chart V throughput is only a proxy.',
            'R provision covers WHALE/VETO across3 seeds; an optional larger scope needs a new forecast.']}


def forecast(configuration=None):
    h0_path=OUTPUT/'h0-calibration-v2/result.json';h0=read(h0_path)
    if h0['status']!='COMPLETE_SHARED_H0_CALIBRATION':raise ValueError('Complete h0 calibration first')
    cases=[read(r['result']) for r in h0['results'].values()]
    timings=[r['timing_seconds'] for r in cases]
    # Suite allocation contains a little orchestration overhead beyond its3 cases.
    suite=read(OUTPUT/'allocations/fast-chart-h0-calibration-plan-20260915-v2/allocation-result.json')
    if suite['state']!='COMPLETED':raise ValueError('Initial calibration has not reached its terminal state')
    overhead=max(0,(suite['seconds']-sum(t['total'] for t in timings))/3)
    evaluation=evaluation_projection(max(t['loading'] for t in timings),
        max(t['evaluation']/512 for t in timings),overhead)
    budget=CompactExecutionBudget().compact_summary();storage=storage_check()
    report={'status':'PENDING_MEASURED_TRAINING_AND_STORAGE','created_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_sha256':{str(h0_path):file_sha256(h0_path)},'gpu_budget':budget,'storage':storage,
        'evaluation_projection':evaluation,'formal_expansion_admitted':False,
        'deletion_authorized':False,'scientific_method_verified':False}
    if configuration is None:return report
    cfg=verified_calibration(configuration)
    trials=[]
    for item in cfg['results'].values():
        train_path=Path(item['training_plan']);train=read(train_path)
        result_path=Path(item['result_path']);followup_path=result_path.parent/'plan.json';followup=read(followup_path)
        trials.append({'training_gpu_hours':allocation(train_path),'followup_gpu_hours':allocation(followup_path),
            'native_gib':allocated_gib(Path(train['output'])/'checkpoints/global_step_4'),
            'export_gib':allocated_gib(followup['target'])})
    train=max(t['training_gpu_hours'] for t in trials);followup=max(t['followup_gpu_hours'] for t in trials)
    native=max(t['native_gib'] for t in trials);export=max(t['export_gib'] for t in trials)
    # Seed42's winning calibration prefix is reused. Other two seeds and all15
    # real continuations remain. No shared prefix is counted as an independent run.
    phases={'core':11*(train+followup)+24*evaluation['one_H128_C128_gpu_hours']+
        3*evaluation['one_V512_gpu_hours']+evaluation['independent_R_six_1024_pair_passes_gpu_hours'],
        'ablation':6*(train+followup),'evaluation':evaluation['final_evaluation_phase_proxy_gpu_hours']}
    retained=storage['new_root_allocated_gib']+17*(native+export)
    # This is an alternative retention estimate only. The temporary native final
    # exists during verification; removing it requires separate path-level consent.
    rolling=storage['new_root_allocated_gib']+2*(native+export)+15*export+native
    total=sum(float(v) for v in budget['charged_gpu_hours'].values())+sum(phases.values())+5
    limits={'core':40,'ablation':15,'evaluation':20}
    gaps={name:max(0,float(budget['charged_gpu_hours'][name])+amount-limits[name]) for name,amount in phases.items()}
    report.update(status='BUDGET_OR_STORAGE_REVISION_REQUIRED' if any(gaps.values()) or total>100 or retained>400
        else 'PROJECTION_WITHIN_CAPS_REQUIRES_REAL_SEARCH_VALIDATION',
        configuration_sha256=file_sha256(Path(configuration)),measured_lr_trials=trials,
        remaining_phase_proxy_gpu_hours=phases,phase_gap_gpu_hours=gaps,total_proxy_with_5h_retry_reserve=total,
        projected_gib_keep_all_native_and_exports=retained,
        alternative_rolling_peak_gib_requires_separate_cleanup_authorization=rolling,
        limitations=['Training uses the largest observed LR allocation; future candidate trajectories may take longer.',
            'API requests remain under the separate global45CNY journal; this forecast does not spend or reserve API funds.',
            'Storage estimates exclude growth in future trajectory evidence; measure it before admitting all18 conditions.',
            'No phase is silently reallocated, seed removed, checkpoint deleted, or unknown result imputed.'])
    return report


def after_calibration(configuration, output):
    """Bounded CPU continuation; it never submits jobs or revises resource caps."""
    configuration, output = Path(configuration).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT):
        raise ValueError('Require a fresh compact calibration-review output')
    started = time.monotonic()
    while not configuration.exists():
        failure = configuration.parent/'controller-failure.json'
        if failure.exists():
            report={'status':'CALIBRATION_FAILED_NO_EXPANSION','failure':str(failure),
                'failure_sha256':file_sha256(failure),'formal_expansion_admitted':False}
            write_new(output,report);return report
        if time.monotonic()-started > 24*3600:
            report={'status':'CALIBRATION_REVIEW_WAIT_EXPIRED','formal_expansion_admitted':False}
            write_new(output,report);return report
        time.sleep(30)
    try:
        report=forecast(configuration)
    except (ValueError,KeyError,FileNotFoundError) as exc:
        report={'status':'CALIBRATION_EVIDENCE_REQUIRES_DIAGNOSIS','configuration':str(configuration),
            'error_type':type(exc).__name__,'error':str(exc),'formal_expansion_admitted':False}
    write_new(output,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--after-calibration',action='store_true')
    args=parser.parse_args()
    if args.after_calibration:
        if args.configuration is None:parser.error('--after-calibration requires --configuration')
        result=after_calibration(args.configuration,args.output)
    else:
        result=forecast(args.configuration);write_new(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','evaluation_projection','formal_expansion_admitted') if k in result},indent=2))
