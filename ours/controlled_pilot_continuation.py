"""Launch unstarted weight-only seeds with verified native DataLoader order.

Reuse the frozen common E4 configuration and continuous two-batch execution.
This separate entrypoint preserves the original seed42 plan and its disclosed
preflight deviation. Native training and the complete trajectory auditor remain
unchanged; new plans contain the actual dataset/loader-derived batch sequence.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from unittest.mock import patch

from . import controlled_pilot as original
from .audit_native_training_batch import require
from .controlled_data_order import native_data_order
from .visual_task import file_sha256

SOURCES=('ours/controlled_pilot_continuation.py','ours/controlled_data_order.py',
         'ours/native_loader_schedule.py','ours/run_controlled_pilot_continuation.sh')
CONTRACT='native_stateful_dataloader_v1'


def prepare(path,seed):
    require(seed in (43,44),'This continuation writer prepares only unstarted weight-only seeds43/44')
    require(not path.exists() and not path.with_suffix('.yaml').exists(),'Preserve existing trial plan and configuration')
    write=original.write_new
    def frozen_write(destination,plan):
        require(plan['native_dataset_order']['order_contract']==CONTRACT,'Missing actual native loader preflight')
        plan['preflight_order_contract']=CONTRACT
        plan['launch_entrypoint']='ours.controlled_pilot_continuation'
        plan['versions']['torchdata']=version('torchdata')
        plan['source_sha256'].update(plan['native_dataset_order']['independent_index_probe']['runtime_source_sha256'])
        plan['resource_decision']={
            'single_gpu':'Not supported by the validated separate actor/rollout configuration.',
            'two_gpus':'Seed42 measured4098seconds/2H800/2.276667GPUh for two continuous native batches.',
            'four_gpus':'Independent-trial Ray isolation is not validated; keep the same two-GPU topology.',
            'queue_forecast':'Not queried by plan preparation; check current queue before submission.',
            'merge_risk':'One trainer owns both sequential batches, without sampling shards or prefix reuse.'}
        plan['limitations']=[s for s in plan['limitations'] if not s.startswith('Seed propagation into')]+[
            'Actual RLHFDataset and StatefulDataLoader batch IDs/state are verified before sampling.',
            'The completed seed42 trial and its original preflight-order deviation are retained.']
        write(destination,plan)
    with patch.object(original,'native_order',native_data_order), \
         patch.object(original,'SOURCES',(*original.SOURCES,*SOURCES)),patch.object(original,'write_new',frozen_write):
        original.prepare(path,seed)


def check(path):
    plan=original.check(path)
    require(plan['seed'] in (43,44) and plan.get('preflight_order_contract')==CONTRACT and
            plan.get('launch_entrypoint')=='ours.controlled_pilot_continuation','Wrong continuation entrypoint or data-order contract')
    require(set(SOURCES)<=set(plan['source_sha256']) and plan['versions'].get('torchdata')==version('torchdata'),
            'Missing native loader execution bindings')
    order=plan['native_dataset_order']
    require(order['order_contract']==CONTRACT and order['native_task_loader_executed'] and
            order['new_model_calls']==0 and order['order_source_sha256']==file_sha256(Path('ours/controlled_data_order.py')),
            'Wrong loader evidence')
    probe=order['independent_index_probe']
    require(order['planned_batch_ids']==probe['batch_ids'] and
            order['checkpoint_sampler_snapshots']==probe['checkpoint_sampler_snapshots'] and
            all(plan['source_sha256'].get(n)==h for n,h in probe['runtime_source_sha256'].items()),
            'Incomplete independent schedule binding')
    return plan


def preflight(path,plan,*,run=False):
    from omegaconf import OmegaConf
    name=f"controlled-weight-only-seed{plan['seed']}-{os.environ['SLURM_JOB_ID']}" if run else original.CPU_NAME
    if run:
        import torch
        require(torch.cuda.device_count()==2 and all('H800' in torch.cuda.get_device_name(i) for i in range(2)),
                'Expected two H800 devices')
        require(not (original.ROOT/'data/native-rsft'/name).exists(),'Preserve existing training directory')
    raw,config=original.resolve(plan,name)
    expected_raw=Path(plan['resolved_config']).read_text().replace(original.CPU_NAME,name)
    marker='model_engine: dp\n'
    expected=OmegaConf.to_container(OmegaConf.create(expected_raw[expected_raw.index(marker):]),resolve=True)
    require(config==expected,'Actual native Hydra configuration differs from frozen trial')
    require(native_data_order(config)==plan['native_dataset_order'],'Actual native DataLoader order differs')
    report={'kind':'controlled_pilot_preflight','status':'PASS','plan_sha256':file_sha256(path),
        'run_name':name,'seed':plan['seed'],'native_batch_steps':2,'new_model_calls':0,
        'preflight_order_contract':CONTRACT,'entrypoint_source_sha256':file_sha256(Path(__file__))}
    if run:
        output=Path(f"results/controlled-pilot-{os.environ['SLURM_JOB_ID']}");output.mkdir(exist_ok=False)
        original.write_new(output/'start.json',report)
        (output/'effective-config.yaml').write_text(raw)
        os.execvpe('bash',['bash',str(original.ROOT/'ours/run_native_rsft.sh'),*original.training_overrides(plan['seed'])],
                   original.launch_environment(plan,name))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('prepare','check','run'),required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(43,44))
    args=parser.parse_args();os.chdir(original.ROOT)
    if args.phase=='prepare':
        require(args.seed is not None,'Missing unstarted trial seed')
        prepare(args.plan,args.seed)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    else:
        plan=check(args.plan)
        print(json.dumps(preflight(args.plan,plan,run=args.phase=='run')),flush=True)
