"""Audit actual native execution while preserving the erroneous preflight list.

The original frozen plan and audit implementation stay unchanged. Only the
audit's expected batch-ID matrix is derived through the actual native loader.
This is explicitly reported as a preflight protocol deviation, never as full
agreement with the original plan. It does not generate or select new samples.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from . import audit_controlled_pilot as original_audit
from .audit_native_training_batch import require
from .controlled_pilot import check
from .native_loader_schedule import native_schedule
from .native_search import write_json
from .visual_task import file_sha256


async def audit(plan_path,directory):
    from omegaconf import OmegaConf
    plan=check(plan_path)
    raw=Path(plan['resolved_config']).read_text()
    config=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
    schedule=native_schedule(config,plan['native_dataset_order']['native_filtered_ids'])
    old_ids=plan['native_dataset_order']['planned_batch_ids']
    require(schedule['batch_ids']!=old_ids,'Use the ordinary auditor when no schedule deviation exists')
    corrected=deepcopy(plan)
    corrected['native_dataset_order']['planned_batch_ids']=schedule['batch_ids']
    # check() already verified original sources/runtime/model/data. The scoped
    # substitution changes a diagnostic expectation, not any trainer setting.
    with patch.object(original_audit,'check',return_value=corrected):
        native=await original_audit.audit(plan_path,directory)
    return {'kind':'controlled_pilot_with_schedule_deviation_audit',
        'status':'PASS_NATIVE_EXECUTION_WITH_PREFLIGHT_DEVIATION',
        'plan_sha256':file_sha256(plan_path),'original_frozen_plan_modified':False,
        'original_preflight_batch_ids':old_ids,'original_preflight_batch_ids_match':False,
        'native_loader_schedule':schedule,'native_configuration_audit':native,
        'deviation':'The preflight iterated RandomSampler directly instead of the native multiprocess DataLoader lifecycle.',
        'new_model_calls':0,'discarded_or_resampled_trajectories':0,
        'source_sha256':{name:file_sha256(Path(name)) for name in
            ('ours/audit_controlled_schedule_deviation.py','ours/native_loader_schedule.py','ours/audit_controlled_pilot.py')},
        'limitations':['This is not full agreement with the original derived preflight ID list.',
            'Native source, complete configuration and seeds were fixed before sampling; their derived schedule is checked here.',
            'A batch audit does not prove optimizer completion, parameter changes, heldout gains or F0 completion.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve previous audit')
    result=asyncio.run(audit(args.plan,args.directory))
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'native_configuration_audit','native_loader_schedule'}}),flush=True)
