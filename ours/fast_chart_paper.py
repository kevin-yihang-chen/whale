"""Build an evidence-indexed manuscript; never substitute forecasts for results."""
import argparse
import json
import os
from pathlib import Path
import re
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


def learning_rate_table(configuration):
    """Use all three completed calibration trials, with their evidence hashes."""
    from .fast_chart_admission import verified_calibration
    from .fast_chart_calibration_controller import choose_rate
    cfg = verified_calibration(configuration)
    if choose_rate(cfg['results']) != cfg['learning_rate']:
        raise ValueError('Paper calibration does not follow the frozen ordinary-accuracy rule')
    lines = [r'\begin{table}[h]\centering\small',r'\begin{tabular}{lrr}\toprule',
        r'Learning rate & Single image & Both images\\\midrule']
    rows = {}
    for rate, exponent in ((1e-7, -7), (1e-6, -6), (1e-5, -5)):
        item = cfg['results'][str(rate)]
        path, training_path = Path(item['result_path']), Path(item['training_plan'])
        if file_sha256(path) != item['result_sha256'] or file_sha256(training_path) != item['training_plan_sha256']:
            raise ValueError('A learning-rate table source changed')
        value = json.loads(path.read_text())
        if (value['status'] != 'COMPLETE_COMPACT_NATIVE_FOLLOWUP' or
                any(value[k] != item[k] for k in ('marginal_accuracy', 'paired_accuracy'))):
            raise ValueError('Learning-rate scores differ from the completed trial')
        followup = path.parent/'plan.json'
        transition = path.parent/'transition.json'
        if (value['plan_sha256'] != file_sha256(followup) or
                value['transition_sha256'] != file_sha256(transition)):
            raise ValueError('Learning-rate training handoff changed')
        label = f'$10^{{{exponent}}}$'
        if rate == cfg['learning_rate']: label += r' (selected)'
        lines.append(f"{label} & {100*value['marginal_accuracy']:.2f} & {100*value['paired_accuracy']:.2f}"+r'\\')
        rows[str(rate)] = {'path':str(path),'sha256':file_sha256(path),
            'training_plan':str(training_path),'training_plan_sha256':file_sha256(training_path),
            'followup_plan_sha256':file_sha256(followup),'transition_sha256':file_sha256(transition),
            'marginal_accuracy':value['marginal_accuracy'],'paired_accuracy':value['paired_accuracy']}
    lines.extend([r'\bottomrule\end{tabular}',
        r'\caption{Complete learning-rate calibration on the same256 development pairs after four native RSFT batches. Scores are percentages. These are configuration measurements, not VETO gains or independent test results.}\end{table}',
        r'All formal conditions use the selected rate. Ordinary development accuracy determines selection; ties use the smaller rate. Seed42 shares the selected first-stage checkpoint; this calibration is not an additional independent trial.'])
    return '\n'.join(lines)+'\n', {'configuration':str(configuration),'configuration_sha256':file_sha256(Path(configuration)),
        'selected':cfg['learning_rate'],'trials':rows,'claim':'Common training configuration; no VETO effect.'}


def search_table(directory):
    """Import only a complete, reconstructed selection archive and its hashes."""
    directory = Path(directory).resolve()
    report_path, manifest_path = directory/'result.json', directory/'manifest.json'
    report, manifest = [json.loads(p.read_text()) for p in (report_path, manifest_path)]
    protocol = report.get('selection_protocol', 'gate_v1')
    if protocol == 'safety_v3':
        from .fast_chart_search_report_v3 import latex
        decision_names = {'whale', 'veto', 'point_gate'}
        report_source = ROOT/'ours/fast_chart_search_report_v3.py'
    else:
        from .fast_chart_search_report import latex
        decision_names = {'whale', 'veto', 'marginal_gate'}
        report_source = ROOT/'ours/fast_chart_search_report.py'
    table_path = directory/'candidate-selection.tex'
    names = [r['candidate'] for r in report['rows']]
    failed = set(report['failed_slots'])
    if (len(names) != len(set(names)) or 'h0' not in names or set(names) & failed or
            set(names) | failed != {'h0','h1','h2','h3'} or
            set(report['decisions']) != decision_names or
            any(v['accepted_harness'] not in names for v in report['decisions'].values())):
        raise ValueError('Search table does not cover all allocated slots and valid decisions')
    if (report['status'] != 'COMPLETE_RECONSTRUCTED_SEARCH_REPORT' or
            manifest['report_sha256'] != file_sha256(report_path) or
            manifest['table_sha256'] != file_sha256(table_path) or
            manifest['source_sha256'] != file_sha256(report_source) or
            table_path.read_text() != latex(report)):
        raise ValueError('Search table differs from its complete reconstructed evidence')
    for path, digest in report['evidence_sha256'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('A search table evidence source changed')
    return table_path.read_text(), {'report': str(report_path), 'report_sha256': file_sha256(report_path),
        'manifest_sha256': file_sha256(manifest_path), 'table_sha256': file_sha256(table_path),
        'equivalent_decisions': report['equivalent_decisions'],
        'claim': 'Optimization-set selection audit only; no independent method benefit.'}


def illustrative_case(directory):
    """Load an explicitly post-hoc case without changing its source image bytes."""
    from .chart_answer_protocol import decode_truth, parse_answer, verify
    directory = Path(directory).resolve()
    path = directory/'case.json'
    case = json.loads(path.read_text())
    if case['status'] != 'COMPLETE_POSTHOC_ILLUSTRATIVE_CASE' or case['role'] != 'C':
        raise ValueError('Require a disclosed optimization-set illustration')
    for source, digest in case['evidence_sha256'].items():
        if file_sha256(Path(source)) != digest:
            raise ValueError('Illustrative case evidence changed')
    for name, digest in case['images'].items():
        if name not in ('case-0.png','case-1.png') or file_sha256(directory/name) != digest:
            raise ValueError('Illustrative image differs from the measured chart')
    if set(case['images']) != {'case-0.png','case-1.png'} or set(case['candidate_records']) != {'h0','h1','h2'}:
        raise ValueError('Incomplete illustrated pair or candidate coverage')
    answers = case['pair']['answers']
    for records in case['candidate_records'].values():
        if len(records) != 2: raise ValueError('Missing illustrated answer')
        for record, truth in zip(records, answers, strict=True):
            if record['tool_trace']:
                raise ValueError('This illustration text requires the recorded absence of tool calls')
            if record['committed_answer'] != parse_answer(record['raw_answer']) or record['correct'] != int(verify(record['committed_answer'],truth)):
                raise ValueError('Illustrated answer differs from the shared scorer')
    replacements = {'\\':r'\textbackslash{}','{':r'\{','}':r'\}','%':r'\%','&':r'\&',
        '#':r'\#','_':r'\_','$':r'\$','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    def escape(text): return re.sub(r'[\\{}%&#_$^~]', lambda match:replacements[match.group()],text)
    lines = [r'\begin{figure*}[t]\centering',
        r'\includegraphics[width=.48\textwidth]{case-0.png}\hfill\includegraphics[width=.48\textwidth]{case-1.png}',
        r'\caption{Post-hoc optimization-set illustration, images A (left) and B (right). '+escape(case['pair']['question'])+
        ' No inference about which image is the original source is made from display order.}',r'\end{figure*}',
        r'\begin{table}[h]\centering\small\begin{tabular}{lrr}\toprule',r'Candidate & Image A & Image B\\\midrule']
    for name, records in case['candidate_records'].items():
        lines.append(name+' & '+' & '.join(escape(r['committed_answer'])+(' (correct)' if r['correct'] else ' (wrong)') for r in records)+r'\\')
    truth = [decode_truth(answer)['value'] for answer in answers]
    lines.extend([r'\bottomrule\end{tabular}\caption{Same-question paired answers for the displayed example.}\end{table}',
        'Exact target values are '+escape(truth[0])+' and '+escape(truth[1])+r', with the shared 5\% relative tolerance.',
        escape(case['observed_interpretation']),
        'This example was selected after observing h0/h1/h2 and is not representative-frequency evidence. '+escape(case['selection_rule']),
        'All six displayed trajectories contain no tool calls.'])
    return '\n'.join(lines)+'\n', {'path':str(path),'sha256':file_sha256(path),
        'images':case['images'],'claim':'Post-hoc reasoning-failure illustration, not causal visual non-use or a VETO benefit.'}


def cycle_table(directory):
    """Only a complete, evidence-bound E4 trajectory may populate this table."""
    from .fast_chart_cycle_report import latex
    directory = Path(directory).resolve()
    report_path, table_path = directory/'result.json', directory/'cycle-development.tex'
    report, manifest = json.loads(report_path.read_text()), json.loads((directory/'manifest.json').read_text())
    if (report['status'] != 'COMPLETE_RECONSTRUCTED_FIRST_CYCLE'
            or manifest['report_sha256'] != file_sha256(report_path)
            or manifest['table_sha256'] != file_sha256(table_path)
            or manifest['source_sha256'] != file_sha256(ROOT/'ours/fast_chart_cycle_report.py')
            or table_path.read_text() != latex(report)):
        raise ValueError('Cycle table differs from its completed report and source')
    for path, digest in report['evidence_sha256'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('Cycle table evidence changed')
    return table_path.read_text(), {'report':str(report_path), 'report_sha256':file_sha256(report_path),
        'table_sha256':file_sha256(table_path), 'equivalent_decisions':report['equivalent_decisions'],
        'claim':'Completed E4 cycle and descriptive V trajectory; no independent VETO effect.'}


def build(output, configuration=None, search_report=None, case=None, cycle_report=None):
    lr_table = learning_rate_table(configuration) if configuration is not None else None
    selection_table = search_table(search_report) if search_report is not None else None
    case_section = illustrative_case(case) if case is not None else None
    cycle_section = cycle_table(cycle_report) if cycle_report is not None else None
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
            'The structured prompt was selected by ordinary development accuracy. Learning-rate calibration is reported separately below.'])
        (output/'calibration.tex').write_text('\n'.join(lines)+'\n')
        index['shared_h0_calibration']={'path':str(complete),'sha256':file_sha256(complete),'selected':shared['selected']}
    if lr_table is not None:
        (output/'learning-rate-calibration.tex').write_text(lr_table[0])
        index['shared_learning_rate_calibration'] = lr_table[1]
    if selection_table is not None:
        (output/'candidate-selection.tex').write_text(selection_table[0])
        index['candidate_selection'] = selection_table[1]
    if case_section is not None:
        (output/'illustrative-case.tex').write_text(case_section[0])
        for name in case_section[1]['images']:
            shutil.copyfile(Path(case)/name, output/name)
        index['illustrative_case'] = case_section[1]
    if cycle_section is not None:
        (output/'cycle-development.tex').write_text(cycle_section[0])
        index['completed_first_cycle'] = cycle_section[1]
        cycle = json.loads(Path(cycle_section[1]['report']).read_text())
        initial, prefix, final = [cycle['development'][k] for k in ('initial','prefix','continued')]
        overview = (r'\paragraph{Completed single-cycle check.} '
            f"The seed 42 continuation accepted {cycle['training']['successful_trajectories']}/256 trajectories and completed "
            f"{cycle['training']['optimizer_updates']} native updates. Development paired correctness changed from "
            f"{prefix['pairs_correct']}/256 to {final['pairs_correct']}/256; the initial model scored {initial['pairs_correct']}/256. "
            f"Single-image correctness changed from {prefix['images_correct']}/512 to {final['images_correct']}/512. "
            'All three selection rules chose h3, and only the prospectively specified continuation was executed. '
            'Both the weights and evaluation harness changed. This trajectory is not an independent VETO effect; '
            'the remaining seeds, control branches and test endpoints are pending.\n')
        (output/'cycle-overview.tex').write_text(overview)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--configuration',type=Path);p.add_argument('--search-report',type=Path)
    p.add_argument('--case',type=Path);p.add_argument('--cycle-report',type=Path)
    a=p.parse_args();build(a.output,a.configuration,a.search_report,a.case,a.cycle_report)
