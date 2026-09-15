"""Trace shared computation to real ledger entries without making it free.

This is reporting infrastructure. Attributed costs describe the executed shared
study, not a claim that a standalone method could reuse other methods for free.
Unknown API charges retain their reserved upper bound and are never imputed as0.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path

from .fast_chart_protocol import ROOT, LEDGER
from .glm_gateway import BudgetJournal, cost
from .native_visual_service import write_new
from .research_budget import ExecutionBudget, terminal_allocation
from .visual_task import file_sha256

API_LEDGER=ROOT/'data/glm-budget/ledger.jsonl'


def amount(value):
    return {'exact_fraction':str(value),'display_value':float(value)}


def totals(resources):
    result={}
    for unit in ('gpu_hours','cny'):
        selected=[r for r in resources if r['unit']==unit]
        settled=sum((Fraction(r['settled']) for r in selected),Fraction())
        held=sum((Fraction(r['held']) for r in selected),Fraction())
        result[unit]={'settled':amount(settled),'unsettled_reserved':amount(held),
            'settled_plus_reserved_upper_bound':amount(settled+held)}
    return result


def snapshot(gpu_ledger=LEDGER,api_ledger=API_LEDGER):
    gpu_ledger,api_ledger=Path(gpu_ledger),Path(api_ledger)
    resources=[];evidence={}
    with ExecutionBudget(gpu_ledger).locked() as (_,rows):
        pending,_,jobs=ExecutionBudget.balances(rows)
        evidence[str(gpu_ledger.resolve())]=file_sha256(gpu_ledger)
        reservations={r['id']:r for r in rows if r['event']=='reserve'}
        for row in rows:
            if row['event']!='settle':continue
            terminal_path=Path(row['terminal_path'])
            if file_sha256(terminal_path)!=row['terminal_sha256']:
                raise ValueError('GPU terminal evidence changed')
            terminal=terminal_allocation(terminal_path.read_text(),row['job_id'])
            if terminal is None or abs(terminal['gpu_hours']-Decimal(row['gpu_hours']))>Decimal('1e-8'):
                raise ValueError('GPU cost differs from its real terminal allocation')
            evidence[str(terminal_path)]=row['terminal_sha256']
            resources.append({'id':'gpu:'+row['id'],'unit':'gpu_hours','settled':row['gpu_hours'],'held':'0',
                'status':terminal['state'],'job_id':row['job_id'],'purpose':reservations[row['id']]['purpose']})
        for ident,row in pending.items():
            resources.append({'id':'gpu:'+ident,'unit':'gpu_hours','settled':'0','held':row['gpu_hours'],
                'status':'UNSETTLED_RESERVATION','job_id':jobs.get(ident),'purpose':row['purpose']})
    with BudgetJournal(api_ledger).locked() as (_,rows):
        _,pending=BudgetJournal.balances(rows)
        evidence[str(api_ledger.resolve())]=file_sha256(api_ledger)
        for index,row in enumerate(rows):
            if row['event']=='settle':
                resources.append({'id':'api:'+row['id'],'unit':'cny','settled':row['cny'],'held':'0',
                    'status':'SETTLED_CONSERVATIVE_USAGE'})
            elif row['event']=='initial_probe':
                resources.append({'id':f'api:initial-probe-{index}','unit':'cny',
                    'settled':str(cost(row['input_tokens'],row['output_tokens'])),'held':'0',
                    'status':'SETTLED_CONSERVATIVE_USAGE'})
        for ident,value in pending.items():
            resources.append({'id':'api:'+ident,'unit':'cny','settled':'0','held':str(value),
                'status':'UNRESOLVED_PROVIDER_CALL'})
    if len({r['id'] for r in resources})!=len(resources):
        raise ValueError('Duplicate resource in accounting snapshot')
    return {'status':'PROJECT_TOTALS_ONLY_METHOD_ATTRIBUTION_PENDING',
        'created_utc':datetime.now(timezone.utc).isoformat(),'resources':resources,'totals':totals(resources),
        'evidence_sha256':evidence,'actual_api_provider_bill_cny':None,
        'contains_unsettled_resources':any(Fraction(r['held'])>0 for r in resources),
        'scope':'Vision GPU ledger including prior work; global project API ledger including prior work.',
        'limits':['Reserved amounts are bounds, not actual charges or invoice values.',
            'A resource may represent shared computation or a failure; neither may be silently omitted.',
            'Method costs require an explicit complete beneficiary manifest. No score is inferred from cost.']}


def attribute(report,manifest):
    if manifest['kind']!='compact_shared_cost_attribution':
        raise ValueError('Wrong cost attribution manifest')
    owners=manifest['owners']
    if not owners or len(set(owners))!=len(owners):
        raise ValueError('Require unique named conditions or separately declared project overhead')
    resources={r['id']:r for r in report['resources']}
    assignments={r['resource_id']:r for r in manifest['assignments']}
    if len(assignments)!=len(manifest['assignments']) or set(assignments)!=set(resources):
        raise ValueError('Every resource, including failures and reservations, must be attributed exactly once')
    shares={owner:[] for owner in owners}
    for ident,rule in assignments.items():
        beneficiaries=rule['beneficiaries']
        if (not beneficiaries or len(set(beneficiaries))!=len(beneficiaries) or
                not set(beneficiaries)<=set(owners) or not rule.get('reason','').strip()):
            raise ValueError('Require unique valid beneficiaries and an explicit shared-cost reason')
        source=resources[ident]
        for owner in beneficiaries:
            shares[owner].append({**source,'settled':str(Fraction(source['settled'])/len(beneficiaries)),
                'held':str(Fraction(source['held'])/len(beneficiaries)),
                'share':str(Fraction(1,len(beneficiaries))),'reason':rule['reason']})
    attributed={owner:{'resources':rows,'totals':totals(rows)} for owner,rows in shares.items()}
    if totals([r for rows in shares.values() for r in rows])!=totals(report['resources']):
        raise ValueError('Shared-cost attribution does not conserve the project total')
    return {**report,'status':'COMPLETE_COST_ATTRIBUTION_WITH_UNSETTLED_BOUNDS' if report['contains_unsettled_resources']
        else 'COMPLETE_ACCOUNTED_COST_ATTRIBUTION','attributed':attributed,
        'interpretation':'Equal shares across declared consumers of each executed resource; not standalone algorithm cost.',
        'allocation_reason':manifest['scope']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--manifest',type=Path)
    args=parser.parse_args();report=snapshot()
    if args.manifest:
        report=attribute(report,json.loads(args.manifest.read_text()))
        report['attribution_manifest_sha256']=file_sha256(args.manifest)
    write_new(args.output,report)
    print(json.dumps({'status':report['status'],'totals':report['totals'],
        'contains_unsettled_resources':report['contains_unsettled_resources']},indent=2))
