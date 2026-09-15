"""One-time allocation handoff; independent trial definitions are unchanged.

Finalize the already-running harness-only search, then use the released account
slot for the separately completed weight-only checkpoint export. This is resource
scheduling, with no transfer of the selected harness into weight-only training.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

from .audit_native_training_batch import require
from .controlled_checkpoint_export import check as check_export
from .native_search import write_json
from .search_completion import verify_completed_search
from .visual_task import file_sha256

SEARCH=Path('results/controlled-harness-only-seed42-plan-20260910-v1.json')
EXPORT=Path('results/controlled-weight-only-seed42-export-plan-20260910-v1.json')
COMPLETION=Path('results/controlled-harness-only-seed42-completion-20260910.json')
ROOT=Path('data/seed42-search-export-handoff-20260910')
EXPECTED_SEARCH_SHA='ade2e2a9ed315ab5abddb0b52e4679cfb5f3d9641b6f2703c89a4abf6a52d59d'
EXPECTED_EXPORT_SHA='fa4a3dd05ab8152051fc15a13087b2190f6c60d0f1d8715e0311c02f75925174'
CONTROLLER_PID=1169570


def run():
    require('SLURM_JOB_ID' not in os.environ,'This allocation coordinator belongs on the login node')
    require(file_sha256(SEARCH)==EXPECTED_SEARCH_SHA and file_sha256(EXPORT)==EXPECTED_EXPORT_SHA,
            'Different frozen search or export plan')
    require(not COMPLETION.exists(),'Inspect the existing completion proof before starting another coordinator')
    ROOT.mkdir(exist_ok=False)
    source=file_sha256(Path(__file__));plan=json.loads(SEARCH.read_text());search_root=Path(plan['phase_root'])
    def state(status,**fields):
        record={'status':status,'pid':os.getpid(),'search_controller_pid':CONTROLLER_PID,
            'time_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source,
            'search_plan_sha256':EXPECTED_SEARCH_SHA,'export_plan_sha256':EXPECTED_EXPORT_SHA,**fields}
        temporary=ROOT/'state.tmp';write_json(temporary,record);temporary.replace(ROOT/'state.json')
        print(json.dumps(record),flush=True)
    try:
        deadline=time.monotonic()+45*60;state('WAITING_FOR_COMPLETED_SEARCH')
        while True:
            current=json.loads((search_root/'state.json').read_text())
            require(current['controller_pid']==CONTROLLER_PID and current['plan_sha256']==EXPECTED_SEARCH_SHA,
                    'Search controller identity changed')
            if current['status']=='COMPLETED':break
            require(current['status']!='FAILED','Search failed; preserve its evidence without restart')
            os.kill(CONTROLLER_PID,0)
            require(time.monotonic()<deadline,'Search wait exceeded45 minutes')
            time.sleep(15)
        state('VERIFYING_COMPLETE_SEARCH')
        proof=verify_completed_search(SEARCH)
        require(proof['condition']=='harness_only' and proof['seed']==42,'Different completed trial')
        require(not COMPLETION.exists(),'Preserve another completion proof')
        write_json(COMPLETION,proof)
        state('CHECKING_INDEPENDENT_EXPORT',completion_sha256=file_sha256(COMPLETION),accepted=proof['accepted'])
        export=check_export(EXPORT)
        require(export['source_job_id']=='222801' and export['seed']==42 and export['step']==2,
                'Different independent weight-only export')
        require(file_sha256(Path(__file__))==source,'Coordinator source changed during wait')
        require(not subprocess.check_output(['squeue','-h','-u','yihangc','-o','%i'],text=True).strip(),
                'Another job occupies the account slot; do not compete or resubmit')
        state('SUBMITTING_EXPORT_ONCE',completion_sha256=file_sha256(COMPLETION))
        raw=subprocess.check_output(['sbatch','--parsable','ours/run_controlled_checkpoint_export.sh',str(EXPORT)],text=True)
        job=raw.strip().split(';')[0];require(job.isdigit(),'Ambiguous submission reply; inspect squeue before any recovery')
        write_json(ROOT/'submission.json',{'job_id':job,'reply':raw,'plan':str(EXPORT),'plan_sha256':EXPECTED_EXPORT_SHA,
            'completion':str(COMPLETION),'completion_sha256':file_sha256(COMPLETION),'new_model_calls':0,'new_api_calls':0})
        state('EXPORT_SUBMITTED',job_id=job,completion_sha256=file_sha256(COMPLETION),accepted=proof['accepted'])
    except BaseException as error:
        state('STOPPED_FOR_INSPECTION',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':run()
