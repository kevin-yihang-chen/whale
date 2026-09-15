"""One native Meta-Harness round, shared candidate archive and E2-E3 branch handoffs."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from .acceptance import EvidenceConstrainedAcceptance
from .adapter import WHALEAcceptanceAdapter
from .chart_answer_protocol import decode_truth
from .evidence import EvaluationIdentity
from .fast_chart_protocol import ROOT,OUTPUT
from .fast_chart_search_evaluation import certify
from .glm_gateway import BudgetJournal,MODEL
from .isolated_visual_harness import load_isolated_visual_harness
from .native_search import tree_hashes
from .native_visual_service import write_new
from .scoped_proposer import isolated_native_proposal
from .visual_search_bridge import boundary,once,PhaseBoundary
from .visual_task import file_sha256

CONTRACT=ROOT/'ours/fast_chart_search_contract.md'
SLOTS=('h1','h2','h3')


def read(path):return json.loads(Path(path).read_text())


def init(root,baseline_plan,allocation,*,parent_plan=None):
    root=Path(root).resolve()
    if root.exists() or not root.is_relative_to(OUTPUT):raise ValueError('Require fresh compact search root')
    base=certify(baseline_plan,allocation)
    if base['candidate'].name!='h0':raise ValueError('Baseline must be incoming h0')
    parent_sha=None
    if parent_plan:
        parent_plan=Path(parent_plan).resolve();parent=read(parent_plan)
        if parent['kind']!='compact_chart_rsft_stage' or parent['stage']!=1 or parent['seed']!=base['plan']['seed']:
            raise ValueError('Wrong common first-stage parent')
        ref=Path(base['plan']['reference_plan'])
        transition=read(ref.parent/'transition.json')
        if transition['exported']!=base['plan']['model'] or transition['native_checkpoint']['directory']!=str(Path(parent['output'])/'checkpoints/global_step_4'):
            raise ValueError('Search weights are not this first-stage export')
        parent_sha=file_sha256(parent_plan)
    root.mkdir();run=root/'search';(run/'harnesses/h0').mkdir(parents=True);(run/'logs/claude_sessions').mkdir(parents=True)
    shutil.copyfile(base['plan']['config']['data']['visual_harness_path'],run/'harnesses/h0/harness.py')
    shutil.copyfile(CONTRACT,root/'contract.md')
    write_new(root/'native-config.json',{'task_profiles':{'H':{'limit':128}},'models':[{'model':'qwen35-4b'}],'seeds':[base['plan']['seed']]})
    write_new(root/'plan.json',{'kind':'compact_native_harness_search','incoming':'h0','slots':SLOTS,
        'identity':base['result']['identity'],'seed':base['plan']['seed'],'iterations':1,'proposals_per_iter':3,
        'parent_plan':str(parent_plan) if parent_plan else None,'parent_plan_sha256':parent_sha,
        'max_api_requests':12,'max_cli_turns':12,'data_role':'mh_val','actual_data_role':'H',
        'baseline':{'plan':str(baseline_plan),'allocation':str(allocation),
            'plan_sha256':file_sha256(baseline_plan),'allocation_sha256':file_sha256(allocation)},
        'source_sha256':{str(p):file_sha256(p) for p in (Path(__file__),CONTRACT)},
        'selection_pass':'First complete H/C pass; failures consume allocated slots.',
        'method_claim':'Single limited-budget alternation; no long-horizon guarantee.'})


def load(root):
    plan=read(root/'plan.json')
    if plan['kind']!='compact_native_harness_search' or tuple(plan['slots'])!=SLOTS:raise ValueError('Wrong search plan')
    for name,digest in plan['source_sha256'].items():
        if file_sha256(Path(name))!=digest:raise ValueError('Frozen search code/contract changed')
    for key in ('plan','allocation'):
        if file_sha256(Path(plan['baseline'][key]))!=plan['baseline'][key+'_sha256']:raise ValueError('Baseline evidence changed')
    return plan


def attach(root,candidate_plan,allocation):
    plan=load(root);candidate_plan=Path(candidate_plan).resolve();allocation=Path(allocation).resolve()
    candidate=certify(candidate_plan,allocation);name=candidate['candidate'].name
    if name not in SLOTS or candidate['result']['identity']!=plan['identity']:
        raise ValueError('Candidate belongs to another slot or fixed-weight phase')
    if candidate['candidate'].harness_sha256!=file_sha256(root/f'search/harnesses/{name}/harness.py'):
        raise ValueError('Evaluated candidate differs from proposed code')
    once(root/f'evaluation-{name}.json',{'plan':str(candidate_plan),'allocation':str(allocation),
        'plan_sha256':file_sha256(candidate_plan),'allocation_sha256':file_sha256(allocation)})


def failed_evaluation(root,candidate_plan,allocation):
    plan=load(root);candidate=read(candidate_plan);terminal=read(allocation);name=candidate['candidate']
    if name not in SLOTS or candidate['identity']!=plan['identity']:
        raise ValueError('A failed incoming baseline must stop the phase; only allocated candidate failures can consume slots')
    if terminal['state']=='COMPLETED' or not (Path(candidate['output'])/'failure.json').exists():
        raise ValueError('Require explicit failed execution and preserved failure evidence')
    once(root/f'failure-{name}.json',{'status':'FAILED_CANDIDATE_CONSUMED_SLOT','candidate':name,
        'plan':str(candidate_plan),'plan_sha256':file_sha256(Path(candidate_plan)),
        'allocation':str(allocation),'allocation_sha256':file_sha256(Path(allocation)),
        'failure_sha256':file_sha256(Path(candidate['output'])/'failure.json'),'scored_as_zero':False})


def public_feedback(run,item):
    name=item['candidate'].name;target=run/f'logs/H/{name}/qwen35-4b';target.mkdir(parents=True,exist_ok=True)
    h=item['h'];records={r['sample_id']:r for r in h['records']}
    once(target/'val.json',{'num_examples':128,'success_rate':h['accuracy'],'avg_reward':h['accuracy'],
        'mean_turn_count':h['mean_native_turns'],'role':'H'})
    once(target/'feedback.json',{'role':'H','examples':[{'example':i,
        'question':row['prompt'][-1]['content'].removeprefix('<image>\n'),
        'model_answer':records[row['visual_sample_id']]['raw_answer'],
        'correct':records[row['visual_sample_id']]['correct'],
        'expected_answer':decode_truth(row['reward_model']['ground_truth'])['value']}
        for i,row in enumerate(item['h_rows'])]})


def advance(root):
    from meta_harness import meta_harness_chess_puzzle as native,chess_puzzle_benchmark as benchmark
    plan=load(root);run=root/'search';base=certify(plan['baseline']['plan'],plan['baseline']['allocation'])
    archive=[base];identity=EvaluationIdentity(**plan['identity'])
    for name in SLOTS:
        receipt=root/f'evaluation-{name}.json'
        if receipt.exists():
            r=read(receipt)
            if file_sha256(Path(r['plan']))!=r['plan_sha256'] or file_sha256(Path(r['allocation']))!=r['allocation_sha256']:
                raise ValueError('Attached evaluation changed')
            archive.append(certify(r['plan'],r['allocation']))
    pick_original,results_original=native.pick_accepted,benchmark.load_results
    initial=True
    def visible_results(path):
        # Re-entering the native loop after an offline boundary must not turn a
        # previously evaluated candidate into the phase incoming reference.
        results=dict(sorted(results_original(path).items()))
        return {k:v for k,v in results.items() if not initial or k[-1]=='h0'}
    def pick(frontier,run_dir):
        nonlocal initial
        if initial:
            initial=False
            adapter=WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance('paired'))
            with patch.object(native,'pick_accepted',pick_original):
                once(root/'initial-acceptance.json',boundary(adapter,native,run,[base],identity,stage='initial',frontier=frontier))
        return pick_original(frontier,run_dir)
    def sweep(config,harnesses,logs_dir,**kwargs):
        result=[]
        for name,path in harnesses:
            if (root/f'failure-{name}.json').exists():result.append((name,False));continue
            item=next((a for a in archive if a['candidate'].name==name),None)
            if item is None:
                once(root/f'evaluation-request-{name}.json',{'candidate':name,'harness':str(path),
                    'harness_sha256':file_sha256(path),'identity':plan['identity'],
                    'reference':base['plan']['reference_plan'],'seed':plan['seed']})
                raise PhaseBoundary(f'WAITING_EVALUATION_{name}')
            public_feedback(run,item);result.append((name,True))
        return result
    def validate(run_dir,name):
        if name not in SLOTS:raise ValueError('Unallocated candidate')
        try:load_isolated_visual_harness(run_dir/f'harnesses/{name}/harness.py').unchanged()
        except Exception as exc:
            once(root/f'failure-{name}.json',{'status':'FAILED_CANDIDATE_CONSUMED_SLOT','candidate':name,
                'error_type':type(exc).__name__,'error':str(exc)})
    def propose_boundary(**kwargs):
        once(root/'proposal-request.json',{k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()})
        if not (root/'proposal-ready.json').exists():raise PhaseBoundary('WAITING_NETWORKED_PROPOSER')
        ready=read(root/'proposal-ready.json');session=Path(ready['session'])
        if ready['request_sha256']!=file_sha256(root/'proposal-request.json') or tree_hashes(session)!=ready['session_sha256']:
            raise ValueError('Proposer receipt changed')
        for name,digest in ready.get('imported_sha256',{}).items():
            if file_sha256(run/name)!=digest:raise ValueError('Proposed candidate or metadata changed after import')
        meta=read(session/'meta.json')
        return native.claude_wrapper.parse_stream_events((session/'events.jsonl').read_text(),kwargs['task_prompt'],
            MODEL,meta['duration_seconds'],meta['exit_code'],cwd=meta['cwd'])
    args=argparse.Namespace(run_name='search',config=str(root/'native-config.json'),iterations=1,proposals_per_iter=3,
        proposer_model=MODEL,proposer_effort='low',propose_timeout=300,early_stop_success_rate=1.,fresh=False,force=False,
        start_iteration=1,early_stop_min_iters=0,early_stop_patience=2,eval_only=False,use_api_key=True,prompt_only=False)
    task='Read h0 and its H feedback. Produce the three allocated standalone candidates h1,h2,h3 and pending_eval.json. '+CONTRACT.read_text()
    with ExitStack() as stack:
        for obj,key,value in ((native,'RUNS_DIR',root),(native,'BASELINE_HARNESS',run/'harnesses/h0/harness.py'),
            (native,'PROMPT_ONLY',False),(native,'SKILL_DIR',root/'contract.md'),
            (native,'render_task_prompt',lambda iteration,names:task),(native,'next_harness_names',lambda directory,count:list(SLOTS)),
            (native,'run_sweep',sweep),(native,'pick_accepted',pick),(native,'validate_candidate',validate),
            (native,'propose_claude_with_retries',propose_boundary),
            (benchmark,'load_results',visible_results)):
            stack.enter_context(patch.object(obj,key,value))
        stack.enter_context(patch.dict('os.environ',{'BASELINE_HARNESS_OVERRIDE':''}))
        try:native.run_evolve(args)
        except PhaseBoundary as wait:
            print(json.dumps({'status':str(wait),'root':str(root)}),flush=True);return
    comparison=read(run/'logs/iteration_001/comparison.json');stage='early_stop' if comparison['early_stop'] else 'ordinary'
    selections={}
    for condition,mode in (('whale','off'),('veto','paired'),('marginal_gate','marginal_gate')):
        adapter=WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance(mode))
        value=boundary(adapter,native,run,archive,identity,stage=stage,frontier=comparison['frontier'],
            rows=comparison['summary'],valid_names=list(SLOTS))
        selected=value['accepted_harness'];harness=run/f'harnesses/{selected}/harness.py'
        value.update(status='COMPLETE_COMPACT_SELECTION',condition=condition,parent_plan_sha256=plan['parent_plan_sha256'],
            selected_harness=str(harness),harness_sha256=file_sha256(harness),identity=plan['identity'],
            archive={x['candidate'].name:{'candidate':asdict(x['candidate']),'audit':asdict(x['audit'])} for x in archive})
        once(root/f'selection-{condition}.json',value)
        restored=boundary(adapter,native,run,archive,identity,stage='resume',frontier=comparison['frontier'],
            rows=comparison['summary'],valid_names=list(SLOTS),resume=value['receipt'])
        once(root/f'restored-selection-{condition}.json',restored);selections[condition]=selected
    for name in SLOTS:
        if name not in {x['candidate'].name for x in archive} and not (root/f'failure-{name}.json').exists():
            once(root/f'failure-{name}.json',{'status':'FAILED_CANDIDATE_CONSUMED_SLOT','candidate':name,'error':'Allocated slot missing from complete valid archive'})
    once(root/'result.json',{'status':'COMPLETE_COMPACT_SEARCH_AND_SELECTION','selections':selections,
        'equivalent_decisions':len(set(selections.values()))==1,'candidate_attempts':3,'scientific_method_verified':False})


def propose(root):
    from meta_harness import meta_harness_chess_puzzle as native
    plan=load(root);request=read(root/'proposal-request.json')
    if request['next_names']!=list(SLOTS) or request['iteration']!=1:raise ValueError('Unexpected proposal budget')
    if (root/'paid-proposal').exists():raise ValueError('No automatic paid retries')
    certify(plan['baseline']['plan'],plan['baseline']['allocation'])
    request['run_dir']=Path(request['run_dir']);journal=BudgetJournal(ROOT/'data/glm-budget/ledger.jsonl')
    with patch.object(native,'SKILL_DIR',root/'contract.md'),patch.object(native,'PROMPT_ONLY',False),\
         patch.object(native,'PROPOSER_SYSTEM_PROMPT',CONTRACT.read_text()):
        result=isolated_native_proposal(native,plan,root/'paid-proposal',journal,**request)
    if result.exit_code!=0:raise RuntimeError('Paid proposal failed; retain fees/artifacts and do not retry automatically')
    sessions=list((root/'search/logs/claude_sessions').iterdir())
    if len(sessions)!=1:raise ValueError('Unexpected proposer sessions')
    write_new(root/'proposal-ready.json',{'request_sha256':file_sha256(root/'proposal-request.json'),
        'session':str(sessions[0]),'session_sha256':tree_hashes(sessions[0]),'budget':journal.summary(),
        'imported_sha256':{name:file_sha256(root/'search'/name) for name in
            set(read(root/'paid-proposal/proposal-integrity.json')['imported'])|{'pending_eval.json'}},
        'actual_data_role':'H','optimization_audit_transferred':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('init','advance','propose','attach','failed-evaluation'))
    p.add_argument('--root',type=Path,required=True)
    for key in ('baseline-plan','allocation','parent-plan','candidate-plan'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='init':init(root,a.baseline_plan.resolve(),a.allocation.resolve(),parent_plan=a.parent_plan)
    elif a.action=='advance':advance(root)
    elif a.action=='propose':propose(root)
    elif a.action=='attach':attach(root,a.candidate_plan,a.allocation)
    else:failed_evaluation(root,a.candidate_plan,a.allocation)
