"""Exploratory E2-v2/E3-v2 archive replay and fixed R512 selection probe.

This compares harnesses at theta1. It neither trains a model nor supplies a
formal VETO-v2 result: the archived candidates predate H-feedback delivery.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from .acceptance import CandidateMetrics, EvidenceConstrainedAcceptance
from .chart_answer_protocol import parse_answer, verify
from .chart_selection_protocol import selection_modes
from .evidence import AuditReceipt, EvaluationIdentity, fingerprint
from .fast_chart_protocol import ROOT, OUTPUT, storage_check
from .native_visual_service import write_new
from .visual_task import file_sha256

PROTOCOL = ROOT/'ours/chart_selection_v2_20260916.md'


def read(path):
    return json.loads(Path(path).read_text())


def replay(selection):
    """Regrade all historical H/C answers before evaluating any new R answers."""
    selection=Path(selection).resolve(); saved=read(selection)
    if saved['status']!='COMPLETE_COMPACT_SELECTION' or saved.get('selection_protocol','gate_v1')!='gate_v1':
        raise ValueError('Require a preserved v1 exploratory archive')
    identity=EvaluationIdentity(**saved['identity']); candidates=[]; audits={}; rows=[]
    evidence={str(selection):file_sha256(selection)}
    def keep(path):
        path=Path(path).resolve(); evidence[str(path)]=file_sha256(path); return read(path)
    search=keep(selection.parent/'plan.json')
    if search['identity']!=saved['identity'] or search['seed']!=42 or set(saved['archive'])!={'h0','h1','h2','h3'}:
        raise ValueError('Require the complete seed42 first-search archive')
    for name,item in saved['archive'].items():
        c=CandidateMetrics(**item['candidate']); a=item['audit']
        audit=AuditReceipt(EvaluationIdentity(**a['identity']),a['harness_sha256'],a['role'],a['pair_ids'],a['correctness'])
        if fingerprint(a)!=saved['receipt']['audit_sha256'][name] or c.name!=name:
            raise ValueError('Archived audit receipt changed')
        attachment=search['baseline'] if name=='h0' else keep(selection.parent/f'evaluation-{name}.json')
        plan=keep(attachment['plan'])
        if evidence[str(Path(attachment['plan']).resolve())]!=attachment['plan_sha256']:
            raise ValueError('Historical candidate plan changed')
        harness=selection.parent/f'search/harnesses/{name}/harness.py'
        if file_sha256(harness)!=c.harness_sha256: raise ValueError('Historical candidate code changed')
        evidence[str(harness)]=c.harness_sha256
        for role in ('H','C'):
            manifest=keep(plan['partitions'][role]['manifest'])
            if manifest['partition']!=role or evidence[str(Path(plan['partitions'][role]['manifest']).resolve())]!=plan['partitions'][role]['manifest_sha256']:
                raise ValueError('Historical data identity changed')
            result=keep(Path(plan['output'])/role/'result.json')
            truth=({fingerprint({'role':'H','sample_id':r['sample_id']}):r['answer'] for r in manifest['examples']} if role=='H' else
                   {fingerprint({'pair_id':p['pair_id'],'side':i}):p['answers'][i] for p in manifest['pairs'] for i in (0,1)})
            records={r['sample_id']:r for r in result['records']}
            if set(records)!=set(truth) or len(records)!=len(result['records']) or len(records)!=128:
                raise ValueError('Incomplete historical measurement')
            correct={}
            for ident,r in records.items():
                answer=parse_answer(r['raw_answer']); correct[ident]=int(verify(answer,truth[ident]))
                if answer!=r['committed_answer'] or correct[ident]!=r['correct']: raise ValueError('Historical grading differs')
            if role=='H':
                if sum(correct.values())/128!=c.accuracy or sum(r['native_turns'] for r in records.values())/128!=c.mean_turns:
                    raise ValueError('Historical H metrics differ')
            else:
                if result['audit']!=a or fingerprint(manifest['pairs'])!=identity.audit_data_sha256:
                    raise ValueError('Historical pair receipt differs')
                pairs={p['pair_id'] for p in manifest['pairs']}
                if pairs!=set(audit.pair_ids): raise ValueError('Historical pair coverage differs')
                for ident,values in zip(audit.pair_ids,audit.correctness):
                    if tuple(correct[fingerprint({'pair_id':ident,'side':i})] for i in (0,1))!=values:
                        raise ValueError('Historical pair outcomes differ')
        candidates.append(c); audits[name]=audit
        rows.append({'candidate':name,'H_correct':round(c.accuracy*128),
            'C_single_correct':sum(sum(v) for v in audit.correctness),
            'C_both_correct':sum(x*y for x,y in audit.correctness)})
    choices={}
    for version in ('gate_v1','evidence_v2'):
        choices[version]={condition:asdict(EvidenceConstrainedAcceptance(mode).select(candidates,incoming='h0',
            identity=identity,audits=audits)) for condition,mode in selection_modes({'selection_protocol':version})}
    return {'status':'EXPLORATORY_ARCHIVE_REPLAY','rows':rows,'decisions':choices,'seed':42,
        'identity':asdict(identity),'evidence_sha256':evidence,'new_model_calls':0,
        'limitations':'Old generic proposals, not a feedback-conditioned v2 search; optimization scores are not independent benefits.'}


def r_subset(manifest):
    if manifest.get('role')!='V' or manifest.get('partition')!='R' or len(manifest['pairs'])!=1024:
        raise ValueError('Require the reserved R1024 source pool')
    pairs=sorted(manifest['pairs'],key=lambda p:(p['source_id'],p['pair_id']))
    if len({p['source_id'] for p in pairs})!=1024: raise ValueError('R probe requires distinct source charts')
    return pairs[:512]


def prepare(path, output, selection):
    from . import fast_chart_evaluation as evaluation
    from .fast_chart_inference_policy import POLICY, load, bind
    from .local_completion import checkpoint_manifest
    path,output,selection=map(lambda p:Path(p).resolve(),(path,output,selection))
    if path.exists() or output.exists() or not output.is_relative_to(OUTPUT): raise ValueError('Require fresh diagnostic paths')
    report=replay(selection)
    version=report['decisions']['evidence_v2']
    names=tuple(dict.fromkeys(version[c]['selected'] for c in ('veto','whale','marginal_rank_floor')))
    if names!=('h1','h3'): raise ValueError('This registered diagnostic compares the reviewed h1/h3 archive only')
    search=read(selection.parent/'plan.json'); baseline=read(search['baseline']['plan'])
    reference=Path(baseline['reference_plan']); ref=read(reference)
    if ref['model']['weights_sha256']!=report['identity']['weights_sha256'] or checkpoint_manifest(Path(ref['model']['path']))!=ref['model']:
        raise ValueError('Diagnostic weights differ from the archived first-stage weights')
    policy=load(); source=OUTPUT/'dataset/R/manifest.json'; manifest=read(source)
    subset=r_subset(manifest); image_ids={sha for p in subset for sha in p['images_sha256']}
    selected_sources={p['source_id'] for p in subset}
    derived={'role':'V','partition':'R','name':'PlotQA-EvidencePairs-R512-selection-probe-v2',
        'pairs':subset,'examples':[],'answer_protocol':manifest['answer_protocol'],
        'image_files':{sha:f'images/{sha}.png' for sha in sorted(image_ids)},
        'task_by_source':{s:t for s,t in manifest['task_by_source'].items() if s in selected_sources},
        'audit_data_sha256':fingerprint(subset),'parent_manifest_sha256':file_sha256(source),
        'subset_rule':'First 512 distinct source charts in ascending source-id order; independent of responses.'}
    storage_check(12); output.mkdir(); (output/'images').mkdir()
    for sha in sorted(image_ids):
        image=(source.parent/manifest['image_files'][sha]).resolve()
        if not image.is_relative_to(source.parent) or file_sha256(image)!=sha:
            raise ValueError('R source pixels changed')
        os.link(image,output/derived['image_files'][sha])
    write_new(output/'archive-replay.json',report); write_new(output/'R512-manifest.json',derived)
    evidence=dict(report['evidence_sha256'])
    for p in (reference,source,POLICY,PROTOCOL,output/'R512-manifest.json',output/'archive-replay.json'):
        evidence[str(p)]=file_sha256(p)
    cases={}
    for name in names:
        case=deepcopy(ref); dest=output/name; cfg=case['config']; harness=selection.parent/f'search/harnesses/{name}/harness.py'
        cfg['data']['visual_harness_path']=str(harness); cfg['data']['cache_dir']=str(dest/'dataset-cache')
        cfg['trainer']['experiment_name']=dest.name; cfg['actor_rollout_ref']['rollout']['trace']['experiment_name']=dest.name
        case.update(kind='compact_selection_probe_case',output=str(dest),partition='R',role='V',
            phase='exploratory-v2-seed42-selection-R512',manifest=str(output/'R512-manifest.json'),
            manifest_sha256=file_sha256(output/'R512-manifest.json'),audit_data_sha256=derived['audit_data_sha256'],
            harness_sha256=file_sha256(harness),seed=42,probe_protocol_sha256=file_sha256(PROTOCOL),
            limitations=['Exploratory independent R512 selection diagnostic, not a final trained-model endpoint.'])
        bind(case,policy); case['decode_sha256']=evaluation.decode_identity(cfg,case['model']['assets'])
        sources=set(ref['source_sha256'])|set(evaluation.native.SOURCES)|set(evaluation.EXTRA)
        sources.update(('ours/chart_selection_diagnostic.py','ours/chart_selection_v2_20260916.md',
            'ours/acceptance.py','ours/chart_selection_protocol.py','ours/run_chart_selection_diagnostic.sh',
            'ours/fast_chart_inference_policy.py','ours/fast_chart_statistics.py',str(harness.relative_to(ROOT))))
        case['source_sha256']={n:file_sha256(ROOT/n) for n in sorted(sources)}
        case['bounds']={'gpus':1,'cpus':12,'time_limit_seconds':3600,'images':1024,
            'maximum_generation_calls':3072,'maximum_generated_assistant_tokens':1048576,'new_model_checkpoints':0,'api_calls':0}
        cp=output/f'{name}-plan.json'; write_new(cp,case); cases[name]={'plan':str(cp),'sha256':file_sha256(cp)}
    suite={'kind':'compact_selection_probe_suite','output':str(output),'cases':cases,'evidence_sha256':evidence,
        'selection_protocol':'evidence_v2','prospective_new_search':False,'phase':'core',
        'comparison':'h1 minus h3 on the same R512 sources and theta1 weights; no tuning after this pass.',
        'continuation_gate':'Positive paired difference and source-cluster 95% interval lower bound > 0, with ordinary loss <= 0.01; otherwise stop expansion.',
        'bounds':{'gpus':1,'cpus':12,'time_limit_seconds':7200,'api_calls':0,'new_model_checkpoints':0,'images':2048}}
    write_new(path,suite); return suite


def check(path):
    from .fast_chart_evaluation import decode_identity
    from .local_completion import checkpoint_manifest
    from .visual_native_evaluation import verifier_identity
    from .chart_answer_protocol import PROTOCOL as ANSWERS
    suite=read(path)
    if suite['kind']!='compact_selection_probe_suite' or tuple(suite['cases'])!=('h1','h3'):
        raise ValueError('Wrong registered diagnostic')
    if suite['bounds']!={'gpus':1,'cpus':12,'time_limit_seconds':7200,'api_calls':0,'new_model_checkpoints':0,'images':2048}:
        raise ValueError('Changed diagnostic budget')
    output=Path(suite['output'])
    if not output.is_relative_to(OUTPUT) or (output/'result.json').exists(): raise ValueError('Diagnostic already completed or unaccounted')
    for name,digest in suite['evidence_sha256'].items():
        if file_sha256(Path(name))!=digest: raise ValueError('Diagnostic input evidence changed')
    reference=None
    for name,entry in suite['cases'].items():
        case=read(entry['plan'])
        if file_sha256(Path(entry['plan']))!=entry['sha256'] or Path(case['output']).exists(): raise ValueError('Changed or already attempted diagnostic case')
        if case['kind']!='compact_selection_probe_case' or case['partition']!='R' or Path(case['output'])!=output/name:
            raise ValueError('Invalid diagnostic case')
        for src,digest in case['source_sha256'].items():
            if file_sha256(ROOT/src)!=digest: raise ValueError('Frozen diagnostic source changed: '+src)
        if file_sha256(Path(case['manifest']))!=case['manifest_sha256'] or file_sha256(Path(case['config']['data']['visual_harness_path']))!=case['harness_sha256']:
            raise ValueError('Diagnostic data or harness changed')
        if checkpoint_manifest(Path(case['model']['path']))!=case['model'] or verifier_identity(ANSWERS)!=case['verifier_sha256']:
            raise ValueError('Diagnostic weights or scoring changed')
        if decode_identity(case['config'],case['model']['assets'])!=case['decode_sha256']:
            raise ValueError('Diagnostic decoding changed')
        binding=(case['model'],case['decode_sha256'],case['manifest_sha256'],case['seed'],case['batch_size'])
        if reference is not None and binding!=reference: raise ValueError('Diagnostic cases differ beyond their harness')
        reference=binding
    storage_check(12); return suite


def contrast(first,second):
    """Paired source bootstrap conditional on this seed and candidate archive."""
    import numpy as np
    if not first or set(first)!=set(second): raise ValueError('Mismatched diagnostic pairs')
    ids=sorted(first); groups={}
    for key in ids:
        if first[key]['source_id']!=second[key]['source_id']: raise ValueError('Source changed')
        groups.setdefault(first[key]['source_id'],[]).append(key)
    counts=np.array([len(keys) for keys in groups.values()])
    if len(groups)<2: raise ValueError('Need independent source groups')
    sums=np.array([sum(first[k]['paired']-second[k]['paired'] for k in keys) for keys in groups.values()])
    rng=np.random.default_rng(20260916); boot=[]
    for start in range(0,20000,256):
        draw=rng.integers(0,len(groups),size=(min(256,20000-start),len(groups)))
        boot.extend((sums[draw].sum(axis=1)/counts[draw].sum(axis=1)).tolist())
    point=float(sums.sum()/counts.sum()); ordinary=sum(first[k]['marginal']-second[k]['marginal'] for k in ids)/len(ids)
    interval=np.quantile(boot,[.025,.975]).tolist()
    return {'paired_difference':point,'ordinary_difference':ordinary,'source_cluster_95_percentile_interval':interval,
        'pairs':len(ids),'sources':len(groups),'bootstrap_seed':20260916,'replicates':20000,
        'status':'SUPPORTS_BOUNDED_CONTINUATION' if point>0 and interval[0]>0 and ordinary>=-.01 else 'INSUFFICIENT_SELECTION_EVIDENCE',
        'limitations':'One seed and previously observed candidates; not a three-seed method result or evidence of post-training benefit.'}


def run_suite(path):
    suite=check(path); output=Path(suite['output']); results={}; records={}
    from .fast_chart_statistics import paired_records
    write_new(output/'start.json',{'job_id':os.environ['SLURM_JOB_ID'],'plan_sha256':file_sha256(path)})
    try:
        for name,entry in suite['cases'].items():
            with (output/f'{name}.log').open('x') as log:
                subprocess.run([sys.executable,'-m','ours.chart_selection_diagnostic','run-case','--plan',entry['plan']],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
            case=read(entry['plan']); rp=Path(case['output'])/'result.json'; result=read(rp)
            if result['status']!='COMPLETE_COMPACT_CHART_PAIR_EVALUATION' or result['plan_sha256']!=entry['sha256']:
                raise ValueError('Incomplete diagnostic case')
            records[name],proof=paired_records(case['manifest'],Path(case['output'])/'evaluation/result.json')
            results[name]={'paired_accuracy':result['paired_accuracy'],'marginal_accuracy':result['marginal_accuracy'],
                'result':str(rp),'result_sha256':file_sha256(rp),'sample_evidence':proof}
        report={'status':'COMPLETE_EXPLORATORY_SELECTION_PROBE','job_id':os.environ['SLURM_JOB_ID'],
            'plan_sha256':file_sha256(path),'results':results,'h1_minus_h3':contrast(records['h1'],records['h3']),
            'new_training_performed':False,'method_claim_verified':False}
        write_new(output/'result.json',report); print(json.dumps(report),flush=True)
    except BaseException as exc:
        write_new(output/'failure.json',{'status':'INCOMPLETE','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=('replay','prepare','check','run-suite','run-case'))
    p.add_argument('--plan',type=Path); p.add_argument('--output',type=Path); p.add_argument('--selection',type=Path)
    a=p.parse_args()
    if a.action=='replay':
        report=replay(a.selection); write_new(a.output,report); print(json.dumps(report['decisions'],indent=2))
    elif a.action=='prepare': prepare(a.plan,a.output,a.selection)
    elif a.action=='check': check(a.plan); print('PASS')
    elif a.action=='run-suite': run_suite(a.plan)
    else:
        from .fast_chart_evaluation import evaluate
        case=read(a.plan)
        if case['kind']!='compact_selection_probe_case': raise ValueError('Wrong diagnostic case')
        asyncio.run(evaluate(case,a.plan))
