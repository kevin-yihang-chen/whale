"""Independent remaining harness-only seeds with predeclared proposer compatibility."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from . import staged_search
from .audit_native_training_batch import require
from .experiment_randomization import TrialRandomization
from .native_search import write_json
from .staged_candidate_recovery import compatible_proposal
from .visual_task import file_sha256

ENTRYPOINT='ours.independent_harness_search'
CONTRACT='bare_list_or_candidates_and_agreeing_name_slot_harness_id_candidate_v2'
SOURCES=('ours/independent_harness_search.py','ours/staged_candidate_recovery.py',
         'ours/staged_metadata_recovery.py','ours/staged_search_recovery.py')


def prepare(path,seed,root):
    require(seed in (43,44),'This entrypoint starts only the two unrun harness-only seeds')
    require(not path.exists() and not root.exists(),'Preserve an existing plan or independent trial')
    with tempfile.TemporaryDirectory(prefix='whale-independent-harness-settings-') as tmp:
        plan=staged_search.prepare(Path(tmp)/'settings.json',seed,root)
    plan.update(coordinator_entrypoint=ENTRYPOINT,metadata_contract=CONTRACT)
    plan['source_sha256'].update({n:file_sha256(Path(n)) for n in SOURCES})
    plan['limitations'].append('Predeclared report-path and agreeing metadata-alias normalization preserves candidate code and original selection.')
    write_json(path,plan);check_plan(path,verify_weights=False);return plan


def check_plan(path,*,verify_weights=True):
    plan=staged_search.check_plan(path,verify_weights=verify_weights)
    require(plan['seed'] in (43,44) and plan.get('coordinator_entrypoint')==ENTRYPOINT and
        plan.get('metadata_contract')==CONTRACT and set(SOURCES)<=set(plan['source_sha256']),
        'Missing independent-trial proposer compatibility contract')
    return plan


class IndependentHarnessSearch(staged_search.StagedSearch):
    def run(self):
        with patch.object(staged_search,'isolated_native_proposal',compatible_proposal):
            return super().run()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','check','coordinate'),required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--root',type=Path)
    parser.add_argument('--seed',type=int,choices=(43,44));parser.add_argument('--submit-evaluations',action='store_true')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(args.seed is not None and args.root is not None,'Missing independent trial identity')
        prepare(args.plan,args.seed,args.root)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    else:
        plan=check_plan(args.plan)
        if args.phase=='check':print(json.dumps({'status':'PASS','plan_sha256':file_sha256(args.plan)}),flush=True)
        else:
            require('SLURM_JOB_ID' not in os.environ,'Proposer coordinator belongs on the networked login node')
            os.environ.update(TrialRandomization(plan['seed']).environment())
            IndependentHarnessSearch(plan,args.plan,submit=args.submit_evaluations).run()
