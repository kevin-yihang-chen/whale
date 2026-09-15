"""Publication figures from complete, traceable sample records and allocation receipts."""
import argparse
import json
from pathlib import Path

from .fast_chart_statistics import paired_records
from .research_budget import terminal_allocation
from .visual_task import file_sha256
from .native_visual_service import write_new


def measured_case(entry):
    plan_path=Path(entry['plan']);plan=json.loads(plan_path.read_text());root=Path(plan['output'])
    result=json.loads((root/'result.json').read_text())
    if result['plan_sha256']!=file_sha256(plan_path) or not result['status'].startswith('COMPLETE_'):
        raise ValueError('Only complete identity-matched measurements may enter figures')
    if plan['partition'] not in ('V','R','T'):raise ValueError('Do not plot optimization C as independent performance')
    rows,identity=paired_records(plan['manifest'],root/'evaluation/result.json')
    return {'paired':sum(r['paired'] for r in rows.values())/len(rows),
        'ordinary':sum(r['marginal'] for r in rows.values())/len(rows),
        'partition':plan['partition'],'identity':identity,'plan_sha256':file_sha256(plan_path)}


def allocated_cost(entries):
    total=0.;seen=set();evidence=[]
    for entry in entries:
        path=Path(entry['receipt']);record=json.loads(path.read_text());share=entry['share']
        if not 0<share<=1 or record['job_id'] in seen:raise ValueError('Invalid or repeated allocation share')
        seen.add(record['job_id']);terminal=path.parent/'slurm-terminal.txt'
        if file_sha256(terminal)!=record['terminal_sha256']:raise ValueError('Cost evidence changed')
        value=terminal_allocation(terminal.read_text(),record['job_id'])
        if value is None or str(value['gpu_hours'])!=record['gpu_hours']:raise ValueError('Cost is not the actual Slurm allocation')
        total+=float(value['gpu_hours'])*share
        evidence.append({'job_id':record['job_id'],'share':share,'gpu_hours':str(value['gpu_hours']),'receipt_sha256':file_sha256(path)})
    return total,evidence


def render(specification,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec=json.loads(Path(specification).read_text());output=Path(output);output.mkdir(exist_ok=False)
    evidence=[]
    fig,axes=plt.subplots(1,2,figsize=(8,3),sharex=True)
    for series in spec['stage_series']:
        values=[measured_case(entry) for entry in series['cases']]
        if len(values)<3 or len({(v['partition'],v['identity']['manifest_sha256']) for v in values})!=1:
            raise ValueError('Stage curves require matched independent data and at least three real measurements')
        for ax,metric,title in zip(axes,('ordinary','paired'),('Single-image accuracy','Both-images accuracy')):
            ax.plot(range(len(values)),[v[metric]*100 for v in values],marker='o',label=series['name'])
            ax.set_title(title);ax.set_xticks(range(len(values)),[e['stage'] for e in series['cases']],rotation=15);ax.set_ylabel('%')
        evidence.append({'series':series['name'],'measurements':values})
    axes[0].legend();fig.tight_layout();fig.savefig(output/'stage-results.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.5,3))
    for point in spec['cost_points']:
        case=measured_case(point);cost,receipts=allocated_cost(point['allocations'])
        if case['partition']!='T':raise ValueError('Final cost comparison requires registered independent test results')
        ax.scatter([cost],[case['paired']*100]);ax.annotate(point['name'],(cost,case['paired']*100),xytext=(4,4),textcoords='offset points')
        evidence.append({'point':point['name'],'measurement':case,'gpu_hours':cost,'allocations':receipts})
    ax.set_xlabel('Allocated GPU-hours, declared shared-cost attribution');ax.set_ylabel('Both-images accuracy (%)')
    fig.tight_layout();fig.savefig(output/'performance-cost.pdf');plt.close(fig)
    write_new(output/'evidence.json',{'specification_sha256':file_sha256(Path(specification)),'evidence':evidence,
        'limitation':'GPU cost panel does not combine API currency with GPU-hours. Report API costs separately in the cost table.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--specification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.specification,a.output)
