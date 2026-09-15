"""Resume E3 at the interrupted third proposal with lossless metadata mapping.

The candidate field is accepted only when all slot aliases agree.
Candidate bytes and all previously audited evaluations are reused. The original
selection loop continues at iteration three; no paid proposal is regenerated.
An independently frozen amendment records this execution compatibility change.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from . import scoped_proposer
from .audit_native_training_batch import require
from .controlled_conditions import native_search_context
from .glm_gateway import BudgetJournal
from .native_search import tree_hashes, write_json
from .staged_search import StagedSearch, check_plan, native_arguments
from .staged_search_recovery import normalize_report
from .visual_task import file_sha256

FAILURE = 'Candidate aliases conflict or leave allocation'
SOURCE = Path(__file__)
ORIGINAL_NORMALIZE = scoped_proposer.normalize_candidates
PREVIOUS_SOURCES = [Path('ours/staged_search_recovery.py'), Path('ours/staged_metadata_recovery.py')]


def normalize_metadata(payload, names, parents):
    """Wrap a bare list and resolve agreeing name/slot/harness/id/candidate aliases only."""
    if isinstance(payload, list):
        payload = {'candidates': payload}
    require(isinstance(payload, dict) and isinstance(payload.get('candidates'), list), FAILURE)
    entries = []
    for entry in payload['candidates']:
        require(isinstance(entry, dict), 'Invalid candidate metadata')
        aliases = [entry[key] for key in ('name','slot','harness','id','candidate') if key in entry]
        require(bool(aliases) and all(isinstance(v,str) and v in names for v in aliases) and
                len(set(aliases)) == 1, 'Candidate aliases conflict or leave allocation')
        entries.append(dict(entry, name=aliases[0]))
    return ORIGINAL_NORMALIZE(dict(payload, candidates=entries), names, parents)


def check_metadata_proposal(view, before, names, iteration, archive):
    with patch.object(scoped_proposer, 'normalize_candidates', normalize_metadata):
        return normalize_report(view, before, names, iteration, archive)


def compatible_proposal(native, plan, output, journal, **kwargs):
    def check(view, before, names, iteration):
        return check_metadata_proposal(view, before, names, iteration, output/'report-normalization')
    with patch.object(scoped_proposer, 'normalize_candidates', normalize_metadata), \
         patch.object(scoped_proposer, 'check_proposal', check):
        return scoped_proposer.isolated_native_proposal(native, plan, output, journal, **kwargs)


def interruption_snapshot(plan, plan_path):
    root = Path(plan['phase_root'])
    state = json.loads((root/'state.json').read_text())
    require(state['status']=='FAILED' and state['error_type']=='ValueError' and state['error']==FAILURE,
            'Recovery requires the recorded third-proposal metadata interruption')
    try:
        os.kill(state['controller_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Previous controller is still alive')
    require(not (root/'result.json').exists() and not (root/'recovery-3').exists(), 'Already resumed or completed')
    for index,(source,kind) in enumerate(zip(PREVIOUS_SOURCES,
            ('staged_search_report_path_recovery','staged_search_second_metadata_recovery')),1):
        prior = json.loads((root/f'recovery-{index}/amendment.json').read_text())
        require(prior['kind']==kind and prior['original_plan_sha256']==file_sha256(plan_path) and
                prior['source_sha256']=={str(source.resolve()):file_sha256(source)},
                'Changed previous recovery provenance')
    evaluations = json.loads((root/'evaluations.json').read_text())
    require([v['harness'] for v in evaluations]==['h0','h1','h2','h3','h4','h5','h6'] and
            sorted(p.name for p in (root/'evaluations').iterdir())==['h0','h1','h2','h3','h4','h5','h6'],
            'Expected exactly seven audited evaluations')
    controller = StagedSearch(plan, plan_path)
    controller.evaluations = evaluations
    controller.verify_public()
    for number in (1,2):
        comparison = json.loads((root/f'search/logs/iteration_{number:03d}/comparison.json').read_text())
        require(comparison['iteration']==number and comparison['early_stop'] is None,
                'Earlier selection is incomplete or should have stopped')
    require((root/'search/logs/accepted_harness.txt').read_text().strip()==comparison['accepted_harness'],
            'Second-round accepted harness changed')
    require(sorted(p.name for p in (root/'search/harnesses').iterdir())==['h0','h1','h2','h3','h4','h5','h6'],
            'Third-round candidate already imported')
    request = json.loads((root/'proposal-request-3.json').read_text())
    require(request['iteration']==3 and request['next_names']==['h7','h8','h9'] and
            Path(request['run_dir'])==root/'search', 'Wrong third proposal allocation')
    receipt = json.loads((root/'proposal-3/proposer-result.json').read_text())
    require(receipt['exit_code']==0 and receipt['iteration']==3 and
            receipt['allocated_slots']==request['next_names'] and receipt['budget']['unresolved_calls']==0,
            'Incomplete third paid proposal')
    search = tree_hashes(root/'search')
    before = {k:v for k,v in search.items() if k!='pending_eval.json'}
    view = root/'proposal-3/proposer-workspace'
    with tempfile.TemporaryDirectory(prefix='whale-metadata-review-') as tmp:
        copy = Path(tmp)/'view'
        shutil.copytree(view,copy,ignore=shutil.ignore_patterns('.cli-state'))
        allowed = check_metadata_proposal(copy,before,request['next_names'],3,Path(tmp)/'normalization')
        payload = normalize_metadata(json.loads((copy/'pending_eval.json').read_text()),request['next_names'],set(['h0','h1','h2','h3','h4','h5','h6']))
    require(all(f'harnesses/{n}/harness.py' in allowed for n in request['next_names']) and
            [entry['name'] for entry in payload['candidates']]==request['next_names'],
            'Expected three complete candidates in the allocated order')
    return {'state':state,'search_sha256':search,
        'proposal_sha256':tree_hashes(view,ignore_cli_state=True),'normalized_metadata':payload,
        'receipt_sha256':file_sha256(root/'proposal-3/proposer-result.json'),
        'request_sha256':file_sha256(root/'proposal-request-3.json'),
        'evaluations_sha256':file_sha256(root/'evaluations.json'),
        'native_config_sha256':file_sha256(root/'native-config.json'),
        'previous_recovery_sha256':{str(n):tree_hashes(root/f'recovery-{n}') for n in (1,2)},
        'raw_sse_sha256':tree_hashes(root/'proposal-3/raw-sse')}


def prepare(plan_path, amendment_path):
    require(not amendment_path.exists(), 'Preserve existing recovery amendment')
    plan = check_plan(plan_path)
    snapshot = interruption_snapshot(plan,plan_path)
    amendment = {'kind':'staged_search_third_metadata_recovery',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'original_plan':str(plan_path.resolve()),'original_plan_sha256':file_sha256(plan_path),
        'source_sha256':{str(SOURCE.resolve()):file_sha256(SOURCE)},'snapshot':snapshot,
        'change':'Wrap a bare candidate list and map agreeing name/slot/harness/id/candidate aliases; retain all original fields.',
        'recovery':'Original run_evolve at start_iteration=3, iterations=3; replay completed proposal 3 once.',
        'new_calls_for_replayed_proposal':0,'new_model_calls_for_reused_evaluations':0,
        'original_plan_modified':False,
        'limitations':['Explicit execution amendment after proposal generation and before third-round evaluation.',
            'Candidate code and native ranking remain unchanged; metadata and report prose are not verified scientific claims.',
            'Native perfect-score stopping is retained; patience stopping was disabled in the frozen trial.',
            'Not general crash recovery; another interruption requires a separate audited continuation.']}
    write_json(amendment_path,amendment)
    return amendment


def check_amendment(path):
    amendment = json.loads(path.read_text())
    require(amendment['kind']=='staged_search_third_metadata_recovery','Wrong recovery amendment')
    original = Path(amendment['original_plan'])
    require(file_sha256(original)==amendment['original_plan_sha256'],'Changed original plan')
    require(amendment['source_sha256']=={str(SOURCE.resolve()):file_sha256(SOURCE)},'Changed recovery source')
    plan = check_plan(original)
    require(interruption_snapshot(plan,original)==amendment['snapshot'],'Changed interrupted trial evidence')
    return amendment,plan,original


def replay_proposal(root,recovery,snapshot,**kwargs):
    serial = {k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()}
    require(serial==json.loads((root/'proposal-request-3.json').read_text()),'Third proposal request changed')
    original = root/'search'
    require(tree_hashes(original)==snapshot['search_sha256'],'Native continuation changed proposal inputs')
    source = root/'proposal-3/proposer-workspace'
    require(tree_hashes(source,ignore_cli_state=True)==snapshot['proposal_sha256'],'Paid proposal bytes changed')
    view = recovery/'validated-proposal'
    shutil.copytree(source,view,ignore=shutil.ignore_patterns('.cli-state'))
    before = {k:v for k,v in snapshot['search_sha256'].items() if k!='pending_eval.json'}
    allowed = check_metadata_proposal(view,before,kwargs['next_names'],3,recovery/'report-normalization')
    payload = normalize_metadata(json.loads((view/'pending_eval.json').read_text()),kwargs['next_names'],set(['h0','h1','h2','h3','h4','h5','h6']))
    require(payload==snapshot['normalized_metadata'],'Changed reviewed metadata')
    for name in allowed:
        destination = original/name
        require(name=='pending_eval.json' or not destination.exists(),'Refuse overwrite of an existing candidate or report')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if name=='pending_eval.json':
            write_json(destination,payload)
        else:
            destination.write_bytes((view/name).read_bytes())
    shutil.copytree(view/'logs/claude_sessions',original/'logs/claude_sessions',dirs_exist_ok=True)
    write_json(recovery/'proposal-replay.json',{'status':'PASS','iteration':3,'imported':allowed,
        'imported_sha256':{n:file_sha256(original/n) for n in allowed},
        'original_proposal_sha256':snapshot['proposal_sha256'],'normalized_metadata':payload,'new_api_calls':0})
    return SimpleNamespace(show=lambda:print('Replayed the exact completed third proposal; zero new API calls.',flush=True))


class MetadataRecoveredSearch(StagedSearch):
    def run_recovered(self,amendment,amendment_path):
        recovery = self.root/'recovery-3'
        recovery.mkdir(exist_ok=False)
        (recovery/'amendment.json').write_bytes(amendment_path.read_bytes())
        (recovery/'failed-state.json').write_bytes((self.root/'state.json').read_bytes())
        self.evaluations = json.loads((self.root/'evaluations.json').read_text())
        self.verify_public()
        journal = BudgetJournal(Path('data/glm-budget/ledger.jsonl'))
        self.state('RECOVERING_THIRD_PROPOSAL',amendment_sha256=file_sha256(amendment_path),budget=journal.summary())
        args = native_arguments(self.plan,self.root)
        args.start_iteration,args.iterations = 3,3
        require(args.early_stop_min_iters==0,'Cannot reset active patience state')
        require(file_sha256(self.root/'native-config.json')==amendment['snapshot']['native_config_sha256'],
                'Changed native configuration')
        with native_search_context(self.plan['condition'],self.root,self.plan['incoming_harness']) as (native,benchmark,provenance):
            original_sweep = native.run_sweep
            def sweep(config,harnesses,logs_dir,**kwargs):
                self.verify_public()
                rows = original_sweep(config,harnesses,logs_dir,**kwargs)
                require(len(rows)==len(harnesses) and all(ok for _,ok in rows),'Native evaluation failed')
                require(all(n in {v['harness'] for v in self.evaluations} for n,_ in harnesses),'Unaudited native score')
                self.verify_public()
                return rows
            def propose(**kwargs):
                self.verify_public()
                number = kwargs['iteration']
                if number==3:
                    return replay_proposal(self.root,recovery,amendment['snapshot'],**kwargs)
                require(4<=number<=5,'Outside original remaining rounds')
                request = self.root/f'proposal-request-{number}.json'
                require(not request.exists(),'Refuse repeated paid proposal')
                write_json(request,{k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()})
                self.state('PROPOSING',iteration=number,slots=kwargs['next_names'],budget=journal.summary())
                return compatible_proposal(native,self.plan,self.root/f'proposal-{number}',journal,**kwargs)
            try:
                with patch.object(benchmark,'evaluate_harness',self.evaluate),patch.object(native,'run_sweep',sweep), \
                     patch.object(native,'propose_claude_with_retries',propose):
                    native.run_evolve(args)
                self.verify_public()
                accepted = native.get_accepted_harness(self.root/'search')
                write_json(self.root/'result.json',{'status':'COMPLETED_NATIVE_SEARCH','plan_sha256':file_sha256(self.plan_path),
                    'recovery_amendment_sha256':file_sha256(amendment_path),
                    'previous_recovery_amendment_sha256':{str(n):file_sha256(self.root/f'recovery-{n}/amendment.json') for n in (1,2)},
                    'accepted':accepted,'accepted_harness_sha256':file_sha256(self.root/f'search/harnesses/{accepted}/harness.py'),
                    'search_sha256':tree_hashes(self.root/'search'),'evaluations':self.evaluations,
                    'budget':journal.summary(),'provenance':provenance,'limitations':self.plan['limitations']+amendment['limitations']})
                self.state('COMPLETED',accepted=accepted,recovery_amendment_sha256=file_sha256(amendment_path))
            except BaseException as error:
                self.state('FAILED',error_type=type(error).__name__,error=str(error),budget=journal.summary())
                raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('prepare','check','coordinate'),required=True)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--submit-evaluations',action='store_true')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(args.plan is not None,'Missing original plan')
        prepare(args.plan,args.amendment)
        print(json.dumps({'status':'PREPARED','amendment_sha256':file_sha256(args.amendment)}),flush=True)
        return
    amendment,plan,original=check_amendment(args.amendment)
    if args.phase=='check':
        print(json.dumps({'status':'PASS','amendment_sha256':file_sha256(args.amendment)}),flush=True)
    else:
        require('SLURM_JOB_ID' not in os.environ,'Coordinator belongs on the networked login node')
        MetadataRecoveredSearch(plan,original,submit=args.submit_evaluations).run_recovered(amendment,args.amendment)


if __name__=='__main__':
    main()
