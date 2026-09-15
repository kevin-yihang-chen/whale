"""Explicit recovery of an interrupted first proposal, without new sampling.

This preserves E3's original five-round selection loop. A separately frozen
amendment records the report-path compatibility correction; the original plan,
failed proposal, audited baseline and evaluation implementation stay unchanged.
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
from .visual_task import file_sha256

FAILURE = 'Unexpected proposer artifact: logs/iteration_1/report.md'
SOURCE = Path(__file__)
ORIGINAL_CHECK = scoped_proposer.check_proposal


def normalize_report(view, before, names, iteration, archive):
    """Accept only the current unpadded report, retaining its original bytes."""
    hashes = tree_hashes(view, ignore_cli_state=True)
    alias = f'logs/iteration_{iteration}/report.md'
    canonical = f'logs/iteration_{iteration:03d}/report.md'
    if alias not in hashes or alias == canonical:
        return ORIGINAL_CHECK(view, before, names, iteration)
    require(alias not in before and canonical not in hashes, 'Ambiguous or protected iteration report')
    raw = (view/alias).read_bytes()
    require(0 < len(raw) <= 65536, 'Empty or oversized iteration report')
    raw.decode('utf-8')
    # Check a disposable normalized copy before changing any original artifact.
    with tempfile.TemporaryDirectory(prefix='whale-report-check-') as tmp:
        copy = Path(tmp)/'view'
        shutil.copytree(view, copy, ignore=shutil.ignore_patterns('.cli-state'))
        (copy/canonical).parent.mkdir(parents=True, exist_ok=True)
        (copy/alias).rename(copy/canonical)
        allowed = ORIGINAL_CHECK(copy, before, names, iteration)
    archive.mkdir(parents=True, exist_ok=False)
    (archive/'report-original.md').write_bytes(raw)
    write_json(archive/'normalization.json', {'status':'PASS', 'iteration':iteration,
        'source':alias, 'destination':canonical, 'report_sha256':hashes[alias],
        'before_normalization_sha256':hashes,
        'scope':'Filename normalization only; report prose is unverified proposer commentary.'})
    (view/canonical).parent.mkdir(parents=True, exist_ok=True)
    (view/alias).rename(view/canonical)
    require(ORIGINAL_CHECK(view,before,names,iteration) == allowed, 'Normalization changed the validated import set')
    return allowed


def compatible_proposal(native, plan, output, journal, **kwargs):
    def check(view,before,names,iteration):
        return normalize_report(view,before,names,iteration,output/'report-normalization')
    with patch.object(scoped_proposer,'check_proposal',check):
        return scoped_proposer.isolated_native_proposal(native,plan,output,journal,**kwargs)


def interruption_snapshot(plan, plan_path):
    root = Path(plan['phase_root'])
    state = json.loads((root/'state.json').read_text())
    require(state['status']=='FAILED' and state['error_type']=='ValueError' and state['error']==FAILURE,
            'Recovery is restricted to the recorded first-proposal report-path interruption')
    try:
        os.kill(state['controller_pid'],0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Previous controller is still alive')
    require(not (root/'result.json').exists() and not (root/'recovery-1').exists(), 'Trial already resumed or completed')
    evaluations = json.loads((root/'evaluations.json').read_text())
    require([v['harness'] for v in evaluations]==['h0'] and
            sorted(p.name for p in (root/'evaluations').iterdir())==['h0'], 'Recovery requires exactly one audited baseline')
    controller = StagedSearch(plan,plan_path)
    controller.evaluations = evaluations
    controller.verify_public()
    search = tree_hashes(root/'search')
    require(not (root/'search/pending_eval.json').exists() and
            sorted(p.name for p in (root/'search/harnesses').iterdir())==['h0'], 'Candidates already imported')
    request = json.loads((root/'proposal-request-1.json').read_text())
    require(request['iteration']==1 and request['next_names']==['h1','h2','h3'] and
            Path(request['run_dir'])==root/'search', 'Wrong interrupted proposal allocation')
    receipt = json.loads((root/'proposal-1/proposer-result.json').read_text())
    require(receipt['exit_code']==0 and receipt['iteration']==1 and
            receipt['allocated_slots']==request['next_names'] and
            receipt['budget']['unresolved_calls']==0, 'Incomplete original paid proposal')
    view = root/'proposal-1/proposer-workspace'
    with tempfile.TemporaryDirectory(prefix='whale-recovery-review-') as tmp:
        review = Path(tmp)/'review'
        shutil.copytree(view,review,ignore=shutil.ignore_patterns('.cli-state'))
        allowed = normalize_report(review,search,request['next_names'],1,Path(tmp)/'normalization')
    require(all(f'harnesses/{n}/harness.py' in allowed for n in request['next_names']) and
            'pending_eval.json' in allowed, 'Incomplete first-round candidates')
    return {'state':state, 'search_sha256':search,
        'proposal_sha256':tree_hashes(view,ignore_cli_state=True),
        'receipt_sha256':file_sha256(root/'proposal-1/proposer-result.json'),
        'request_sha256':file_sha256(root/'proposal-request-1.json'),
        'evaluations_sha256':file_sha256(root/'evaluations.json'),
        'native_config_sha256':file_sha256(root/'native-config.json'),
        'raw_sse_sha256':tree_hashes(root/'proposal-1/raw-sse')}


def prepare(plan_path, amendment_path):
    require(not amendment_path.exists(),'Preserve existing recovery amendment')
    plan = check_plan(plan_path)
    snapshot = interruption_snapshot(plan,plan_path)
    amendment = {'kind':'staged_search_report_path_recovery',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'original_plan':str(plan_path.resolve()),'original_plan_sha256':file_sha256(plan_path),
        'source_sha256':{str(SOURCE.resolve()):file_sha256(SOURCE)},'snapshot':snapshot,
        'change':'Accept current iteration report with or without zero padding; no other new paths.',
        'recovery':'Replay the exact completed first proposal and audited h0 inside the original loop.',
        'new_calls_for_replayed_proposal':0,'new_model_calls_for_reused_h0':0,
        'original_plan_modified':False,
        'limitations':['Explicit execution amendment after proposal generation, before candidate evaluation.',
            'Not automatic crash recovery. A second interruption requires a new reviewed procedure.',
            'Proposer report prose is not a verified result or causal analysis.']}
    write_json(amendment_path,amendment)
    return amendment


def check_amendment(path):
    amendment = json.loads(path.read_text())
    require(amendment['kind']=='staged_search_report_path_recovery','Wrong recovery amendment')
    original = Path(amendment['original_plan'])
    require(file_sha256(original)==amendment['original_plan_sha256'],'Changed original plan')
    require(amendment['source_sha256']=={str(SOURCE.resolve()):file_sha256(SOURCE)},'Changed recovery implementation')
    plan = check_plan(original)
    require(interruption_snapshot(plan,original)==amendment['snapshot'],'Changed interrupted trial evidence')
    return amendment, plan, original


def replay_proposal(root, recovery, snapshot, **kwargs):
    serial = {k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()}
    require(serial==json.loads((root/'proposal-request-1.json').read_text()), 'Original first proposal request changed')
    original = root/'search'
    require(tree_hashes(original)==snapshot['search_sha256'],'Native restart changed the original proposal inputs')
    source = root/'proposal-1/proposer-workspace'
    require(tree_hashes(source,ignore_cli_state=True)==snapshot['proposal_sha256'],'Paid proposal bytes changed')
    view = recovery/'validated-proposal'
    shutil.copytree(source,view,ignore=shutil.ignore_patterns('.cli-state'))
    allowed = normalize_report(view,snapshot['search_sha256'],kwargs['next_names'],1,recovery/'report-normalization')
    payload = scoped_proposer.normalize_candidates(json.loads((view/'pending_eval.json').read_text()),kwargs['next_names'],{'h0'})
    for name in allowed:
        destination = original/name
        require(not destination.exists(),'Refuse to overwrite a recovered candidate or report')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if name=='pending_eval.json':
            write_json(destination,payload)
        else:
            destination.write_bytes((view/name).read_bytes())
    shutil.copytree(view/'logs/claude_sessions',original/'logs/claude_sessions',dirs_exist_ok=True)
    write_json(recovery/'proposal-replay.json',{'status':'PASS','iteration':1,'imported':allowed,
        'imported_sha256':{n:file_sha256(original/n) for n in allowed},
        'original_proposal_sha256':snapshot['proposal_sha256'],'new_api_calls':0,
        'metadata_normalization':'Unambiguous slot to name; original metadata retained.'})
    return SimpleNamespace(show=lambda:print('Replayed the exact completed first proposal; zero new API calls.',flush=True))


class RecoveredSearch(StagedSearch):
    def run_recovered(self, amendment, amendment_path):
        recovery = self.root/'recovery-1'
        recovery.mkdir(exist_ok=False)
        (recovery/'amendment.json').write_bytes(amendment_path.read_bytes())
        (recovery/'failed-state.json').write_bytes((self.root/'state.json').read_bytes())
        self.evaluations = json.loads((self.root/'evaluations.json').read_text())
        self.verify_public()
        journal = BudgetJournal(Path('data/glm-budget/ledger.jsonl'))
        self.state('RECOVERING_FIRST_PROPOSAL',amendment_sha256=file_sha256(amendment_path),budget=journal.summary())
        args = native_arguments(self.plan,self.root)
        require(file_sha256(self.root/'native-config.json')==amendment['snapshot']['native_config_sha256'],
                'Changed native configuration during recovery')
        with native_search_context(self.plan['condition'],self.root,self.plan['incoming_harness']) as (native,benchmark,provenance):
            original_sweep = native.run_sweep
            def sweep(config,harnesses,logs_dir,**kwargs):
                self.verify_public()
                rows = original_sweep(config,harnesses,logs_dir,**kwargs)
                require(len(rows)==len(harnesses) and all(ok for _,ok in rows),'Native evaluation failed; inspect preserved evidence')
                require(all(n in {v['harness'] for v in self.evaluations} for n,_ in harnesses),'Unaudited native score')
                self.verify_public()
                return rows
            def propose(**kwargs):
                self.verify_public()
                number = kwargs['iteration']
                if number==1:
                    return replay_proposal(self.root,recovery,amendment['snapshot'],**kwargs)
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
                    'recovery_amendment_sha256':file_sha256(amendment_path),'accepted':accepted,
                    'accepted_harness_sha256':file_sha256(self.root/f'search/harnesses/{accepted}/harness.py'),
                    'search_sha256':tree_hashes(self.root/'search'),'evaluations':self.evaluations,
                    'budget':journal.summary(),'provenance':provenance,
                    'limitations':self.plan['limitations']+amendment['limitations']})
                self.state('COMPLETED',accepted=accepted,recovery_amendment_sha256=file_sha256(amendment_path))
            except BaseException as error:
                self.state('FAILED',error_type=type(error).__name__,error=str(error),budget=journal.summary())
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('prepare','check','coordinate'),required=True)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--submit-evaluations',action='store_true')
    args = parser.parse_args()
    if args.phase=='prepare':
        require(args.plan is not None,'Missing original plan')
        prepare(args.plan,args.amendment)
        print(json.dumps({'status':'PREPARED','amendment_sha256':file_sha256(args.amendment)}),flush=True)
        return
    amendment,plan,original = check_amendment(args.amendment)
    if args.phase=='check':
        print(json.dumps({'status':'PASS','amendment_sha256':file_sha256(args.amendment)}),flush=True)
    else:
        require('SLURM_JOB_ID' not in os.environ,'Coordinator belongs on the networked login node')
        RecoveredSearch(plan,original,submit=args.submit_evaluations).run_recovered(amendment,args.amendment)


if __name__=='__main__':
    main()
