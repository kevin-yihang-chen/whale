"""Verify completed E3 selection before exposing its selected harness to E4.

Replay native validation, per-round summaries, Pareto selection and stopping
against every audited evaluation. Frontier writes are redirected to a temporary
folder; the native sorted ingestion used by the frozen controller is retained.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from .audit_native_training_batch import require
from .controlled_conditions import native_search_context
from .native_search import tree_hashes,write_json
from .staged_search import StagedSearch,verify_evaluation
from .visual_task import file_sha256

SOURCE=Path(__file__).resolve()


def audit_archive(plan_path,plan):
    """Audit an archive after the caller has verified its experiment plan."""
    require(plan==json.loads(plan_path.read_text()),'Supplied search plan differs from its file')
    require(plan['condition'] in ('harness_only','whale','whale_fst') and
            plan['iterations']==5 and plan['proposals_per_iter']==3,'Wrong search completion scope')
    root=Path(plan['phase_root']).resolve();search=root/'search';logs=search/'logs'
    state=json.loads((root/'state.json').read_text())
    require(state['status']=='COMPLETED','Search is not completed')
    result=json.loads((root/'result.json').read_text());plan_sha=file_sha256(plan_path)
    require(result['status']=='COMPLETED_NATIVE_SEARCH' and result['plan_sha256']==state['plan_sha256']==plan_sha and
            file_sha256(root/'plan.json')==plan_sha,'Search completion identity differs')
    before=tree_hashes(search)
    require(result['search_sha256']==before,'Completed search archive changed')
    require(result['evaluations']==json.loads((root/'evaluations.json').read_text()),'Evaluation history changed')
    entries=result['evaluations'];names=[v['harness'] for v in entries]
    require(len(set(names))==len(names) and names[0]=='h0' and len(names)<=16,'Invalid evaluation sequence')
    require({p.name for p in (root/'evaluations').iterdir()}==set(names),'Unaccounted evaluation attempt')
    controller=StagedSearch(plan,plan_path);controller.evaluations=entries;controller.verify_public()
    totals={'evaluations':len(entries),'examples':0,'policy_calls':0,'generated_tokens':0,'gpu_hours':0.}
    evidence={str(p):file_sha256(p) for p in (plan_path,root/'plan.json',root/'state.json',root/'result.json',
        root/'evaluations.json',root/'native-config.json')}
    for entry in entries:
        private=root/'evaluations'/entry['harness']
        require(Path(entry['directory']).resolve()==private,'Evaluation escaped its own trial')
        summary,terminal=verify_evaluation(private,plan,plan_path)
        require(entry['solved']==summary['solved_examples'] and all(entry[k]==v for k,v in terminal.items()),
                'Recorded evaluation score or allocation differs')
        audit=json.loads((private/'audit.json').read_text())
        totals['examples']+=audit['examples'];totals['policy_calls']+=audit['calls']
        totals['generated_tokens']+=audit['generated_tokens'];totals['gpu_hours']+=terminal['gpu_hours']
        for name in ('audit.json','result.json','request.json','slurm-terminal.txt'):
            evidence[str(private/name)]=file_sha256(private/name)
    comparisons=sorted(logs.glob('iteration_*/comparison.json'))
    require(1<=len(comparisons)<=5 and [p.parent.name for p in comparisons]==
            [f'iteration_{n:03d}' for n in range(1,len(comparisons)+1)],'Missing or extra completed rounds')
    require({p.name for p in root.glob('proposal-request-*.json')}==
            {f'proposal-request-{n}.json' for n in range(1,len(comparisons)+1)},'Unaccounted proposal request')
    require(totals['examples']==32*len(entries) and totals['policy_calls']<=totals['examples']*18 and
            totals['generated_tokens']<=totals['examples']*8129,'Completed search exceeded its generation budget')
    expected_evaluations=['h0'];expected_candidates=['h0'];evolution=[];history=[]
    native_config=json.loads((root/'native-config.json').read_text())
    expected_config={'task_profiles':{'mh_val':{'dataset_path':plan['dataset'],'limit':32}},
        'models':[plan['target_config']],'seeds':[plan['seed']],
        'eval':{'assistant_token_budget':8129,'policy_max_tokens':8129}}
    require(native_config==expected_config,'Native search configuration changed')
    with native_search_context(plan['condition'],root,plan['incoming_harness']) as (native,benchmark,provenance), \
            tempfile.TemporaryDirectory(prefix='whale-completed-selection-') as tmp:
        require(result['provenance']==provenance,'Different native selection context')
        all_results=benchmark.load_results(logs)
        require(len(all_results)==len(entries) and {key[2] for key in all_results}==set(names),
                'Unexpected native score coverage')
        for number,path in enumerate(comparisons,1):
            comparison=json.loads(path.read_text());candidates=comparison['candidates']
            allocated=[f'h{n}' for n in range(3*(number-1)+1,3*number+1)]
            requested=json.loads((root/f'proposal-request-{number}.json').read_text())
            require(requested['iteration']==number and requested['next_names']==allocated and
                    Path(requested['run_dir']).resolve()==search and
                    requested['task_prompt']==native.render_task_prompt(number,allocated),'Changed native proposal request')
            require(comparison['iteration']==number and candidates==json.loads((path.parent/'pending_eval.json').read_text())['candidates'] and
                    1<=len(candidates)<=3 and len({v['name'] for v in candidates})==len(candidates) and
                    all(v['name'] in allocated for v in candidates),'Wrong allocated candidate metadata')
            require([v['name'] for v in candidates]==[n for n in allocated if n in {v['name'] for v in candidates}],
                    'Changed native candidate ordering')
            valid=[];rejections=[]
            for candidate in candidates:
                name=candidate['name'];expected_candidates.append(name)
                try:native.validate_candidate(search,name)
                except Exception as error:
                    require(plan['condition']=='whale_fst','Invalid full-harness candidate was selected')
                    rejections.append({'iteration':number,'harness':name,'status':'rejected_prompt_only',
                        'reason':str(error)[:300],'hypothesis':candidate.get('hypothesis',''),'axis':candidate.get('axis','?')})
                else:valid.append(name)
            require(bool(valid),'Completed round has no valid candidates')
            expected_evaluations.extend(valid)
            prefix={key:value for key,value in all_results.items() if key[2] in expected_evaluations}
            with patch.object(benchmark,'load_results',return_value=prefix),redirect_stdout(io.StringIO()):
                frontier=benchmark.print_frontier(Path(tmp),native_config)
            require(frontier==comparison['frontier'],'Recorded native frontier differs from audited scores')
            rows=native.candidate_summary_rows(number,candidates,logs,frontier)
            require(rows==comparison['summary'],'Recorded candidate summary differs')
            stop=native.find_early_stop_candidate(rows,valid,1.)
            accepted=stop['harness'] if stop else native.pick_accepted(frontier,search)
            require(stop==comparison['early_stop'] and accepted==comparison['accepted_harness'],'Native selection or stopping differs')
            require(stop is None or number==len(comparisons),'Search continued after native perfect-score stop')
            evolution.extend(rows);evolution.extend(rejections)
            history.append({'iteration':number,'accepted':accepted,'evaluated':valid,'rejected':[r['harness'] for r in rejections],
                'early_stop':stop,'comparison_sha256':file_sha256(path)})
        require(len(comparisons)==5 or stop is not None,'Search ended before the frozen stopping condition')
    require(names==expected_evaluations,'Missing, repeated or reordered candidate evaluation')
    require({p.name for p in (search/'harnesses').iterdir() if p.is_dir()}==set(expected_candidates),
            'Unaccounted candidate slot')
    observed_evolution=[json.loads(line) for line in (logs/'evolution_summary.jsonl').read_text().splitlines()]
    require(observed_evolution==evolution,'Native evolution history is missing or duplicated')
    require(json.loads((logs/'frontier_val.json').read_text())==frontier,'Final frontier differs')
    require(result['accepted']==state['accepted']==accepted==(logs/'accepted_harness.txt').read_text().strip(),
            'Selected harness identity differs')
    harness=(search/f'harnesses/{accepted}/harness.py').resolve()
    require(result['accepted_harness_sha256']==file_sha256(harness),'Selected harness bytes changed')
    require(tree_hashes(search)==before,'Completion audit changed the search archive')
    amendments=[]
    for path in sorted(root.glob('recovery-*/amendment.json')):
        amendment=json.loads(path.read_text())
        require(amendment['original_plan_sha256']==plan_sha,'Recovery used a different plan')
        for source,digest in amendment['source_sha256'].items():
            require(file_sha256(Path(source))==digest,'Recovery implementation changed')
        amendments.append({'path':str(path),'sha256':file_sha256(path)})
    return {'kind':'completed_staged_search_audit','status':'PASS_COMPLETED_NATIVE_SEARCH',
        'condition':plan['condition'],'seed':plan['seed'],'plan_sha256':plan_sha,
        'result_path':str(root/'result.json'),'result_sha256':file_sha256(root/'result.json'),
        'harness':str(harness),'harness_sha256':file_sha256(harness),'accepted':accepted,
        'completed_rounds':len(history),'native_perfect_stop':stop is not None,'history':history,'totals':totals,
        'search_sha256':before,'artifact_sha256':evidence,'recovery_amendments':amendments,
        'source_sha256':{str(SOURCE):file_sha256(SOURCE)},'new_model_calls':0,
        'limitations':['Selection on MH optimization data only; no heldout or visual performance is inferred.',
            'The frozen shared sorted score ingestion is retained, including its tie order.',
            'Full evaluation audits are verified by their hashes; no generation is repeated.',
            'A joint checkpoint handoff must separately certify this same condition and seed.']}


def verify_completed_search(plan_path):
    raw=json.loads(plan_path.read_text())
    if raw['kind']=='controlled_joint_staged_mh_plan':
        from .joint_staged_search import check_plan
    else:
        from .staged_search import check_plan
    return audit_archive(plan_path,check_plan(plan_path))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing completion proof')
    report=verify_completed_search(args.plan);write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','condition','seed','accepted','completed_rounds','totals')}),flush=True)
