"""Build an evidence-indexed manuscript; never substitute forecasts for results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from .fast_chart_protocol import ROOT,OUTPUT,schedule,CompactExecutionBudget,storage_check
from .visual_task import file_sha256
from .native_visual_service import write_new


def method_figure(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(8,3));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',linewidth=.9,facecolor=color,edgecolor='#344054'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.2)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.2,'color':'#344054'})
    box(.1,1.65,1.5,.9,'Common model\n+ incoming h', '#eef2f6')
    box(2,1.65,1.5,.9,'Native RSFT\nStage1 (E4)', '#eef2f6')
    box(3.9,1.65,1.5,.9,'Candidate\nsearch on H', '#eef2f6')
    box(5.8,1.65,1.6,.9,'Evidence gate\n+ ranking (E2–E3)', '#d5e9e1')
    box(7.9,1.65,1.8,.9,'Native RSFT\nStage2 (E4)', '#eef2f6')
    box(5.8,.05,1.6,.9,'Paired C audit\nBoth correct (E1)', '#d5e9e1')
    for a,b in (((1.6,2.1),(2,2.1)),((3.5,2.1),(3.9,2.1)),((5.4,2.1),(5.8,2.1)),((7.4,2.1),(7.9,2.1)),((6.6,.95),(6.6,1.65))):arrow(a,b)
    ax.text(5,3.3,'VETO: test visual evidence before choosing the next training harness',ha='center',fontsize=11)
    ax.text(5,2.85,'Shared host parser / verifier · fixed weights and incoming reference during selection',ha='center',fontsize=8.5)
    ax.text(2.5,.35,'C is optimization data.\nIndependent T remains outside selection.',ha='center',fontsize=8.5)
    fig.savefig(path,bbox_inches='tight');plt.close(fig)


def build(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT):raise ValueError('Require a fresh manuscript build')
    storage_check(1);output.mkdir()
    os.environ.setdefault('MPLCONFIGDIR',str(OUTPUT/'runtime-cache/matplotlib'))
    for name in ('main.tex','supplement.tex'):(output/name).write_bytes((ROOT/'ours/paper'/name).read_bytes())
    sources=OUTPUT/'public-sources'
    (output/'cvpr.sty').write_bytes((sources/'author-kit/cvpr.sty').read_bytes())
    method_figure(output/'method.pdf')
    index={'scientific_status':'UNVALIDATED','formal_conditions':schedule()['runs'],
        'protocol':str(ROOT/'ours/fast_chart_protocol_20260915.md'),'budget':CompactExecutionBudget().compact_summary(),
        'calibration_results':{},'test_results':[],'source_sha256':{name:file_sha256(ROOT/'ours/paper'/name) for name in ('main.tex','supplement.tex')}}
    calibration=OUTPUT/'h0-calibration-v2'
    for name in ('direct','structured','brief_reasoning'):
        result=calibration/name/'result.json'
        if result.exists():
            value=json.loads(result.read_text())
            if value['status']!='COMPLETE_COMPACT_CHART_PAIR_EVALUATION':raise ValueError('Incomplete score cannot enter paper index')
            index['calibration_results'][name]={'path':str(result),'sha256':file_sha256(result),
                'marginal_accuracy':value['marginal_accuracy'],'paired_accuracy':value['paired_accuracy'],
                'claim':'Development configuration calibration only; no VETO effect.'}
    complete=calibration/'result.json'
    if complete.exists():
        shared=json.loads(complete.read_text())
        if shared['status']!='COMPLETE_SHARED_H0_CALIBRATION' or len(index['calibration_results'])!=3:
            raise ValueError('Only a completed common calibration may enter the supplement')
        lines=[r'\begin{table}[h]\centering\small',r'\begin{tabular}{lrr}\toprule',
            r'Prompt & Single image & Both images\\\midrule']
        labels={'direct':'Direct','structured':'Structured','brief_reasoning':'Brief reasoning'}
        for name,row in index['calibration_results'].items():
            if row['sha256']!=shared['results'][name]['result_sha256']:
                raise ValueError('Calibration table identity differs from the selected common configuration')
            lines.append(f"{labels[name]} & {100*row['marginal_accuracy']:.2f} & {100*row['paired_accuracy']:.2f}"+r'\\')
        lines.extend([r'\bottomrule\end{tabular}',r'\caption{Complete initial-prompt calibration on256 development pairs. Scores are percentages; these are not VETO gains or test results.}\end{table}',
            'The structured prompt was selected by ordinary development accuracy. Learning-rate calibration remains a separate required step.'])
        (output/'calibration.tex').write_text('\n'.join(lines)+'\n')
        index['shared_h0_calibration']={'path':str(complete),'sha256':file_sha256(complete),'selected':shared['selected']}
    write_new(output/'claim-evidence-index.json',index)
    env={**os.environ,'XDG_CACHE_HOME':str(sources/'tex-cache')}
    for name in ('main','supplement'):
        with (output/f'{name}.build.log').open('x') as log:
            subprocess.run([str(sources/'tectonic'),'--keep-logs',f'{name}.tex'],cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_new(output/'build-result.json',{'status':'COMPILED_MANUSCRIPT_SCAFFOLD','pdfs':{name:file_sha256(output/name) for name in ('main.pdf','supplement.pdf')},
        'author_kit':'cvpr-org/author-kit@291758547e923160eb4d37079b7b9f0dfce82355',
        'template_limitation':'Official repository currently labels this kit2026. Provisional drafting layout; not a verified CVPR2027 submission package.',
        'figure_status':{'method':'CREATED','stage_results':'PENDING_REAL_BRANCH_RESULTS','performance_cost':'PENDING_REAL_BRANCH_RESULTS'},
        'scientific_method_verified':False,'claim_evidence_index_sha256':file_sha256(output/'claim-evidence-index.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
