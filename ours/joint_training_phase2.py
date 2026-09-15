"""Second E4 batch from completed joint E3 and its original native theta1.

The HF model path constructs the native graph; the original loader restores
FP32 actor parameters, scheduler/RNG and data progress before weight transport
and generation. The selected harness comes only from a completed joint search.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import subprocess
from unittest.mock import patch

from . import controlled_pilot as original
from .audit_native_training_batch import require
from .controlled_data_order import native_data_order
from .joint_training_phase1 import check as check_first
from .native_phase_resume import check_configuration,check_layout,phase_overrides
from .native_resume_observation import observe_loader
from .search_completion import verify_completed_search
from .visual_task import file_sha256

SOURCES=('ours/joint_training_phase2.py','ours/run_joint_training_phase2.sh',
    'ours/native_resume_observation.py','ours/search_completion.py','ours/joint_resume_receipts.py',
    'ours/audit_joint_training_phase2.py','ours/joint_training_phase2_result.py')


def certify_search(search_path,completion_path):
    search=json.loads(search_path.read_text())
    require(search['kind']=='controlled_joint_staged_mh_plan' and search['joint_phase']==1 and
            search['condition'] in ('whale','whale_fst'),'Phase two requires an independent joint search')
    proof=json.loads(completion_path.read_text());verified=verify_completed_search(search_path)
    require(proof==verified,'Completed search proof differs from its full archive')
    export=json.loads(Path(search['joint_handoff']['export_plan']).read_text())
    first_path=Path(export['training_plan']);first=check_first(first_path)
    require(first['condition']==proof['condition']==search['condition'] and
            first['seed']==proof['seed']==search['seed'] and
            first['initial_manifest']==search['joint_handoff']['initial_manifest'],'Mixed training/search lineage')
    handoff={'search_plan':str(search_path.resolve()),'search_plan_sha256':file_sha256(search_path),
        'search_completion':str(completion_path.resolve()),'search_completion_sha256':file_sha256(completion_path),
        'search_result':proof['result_path'],'search_result_sha256':proof['result_sha256'],
        'phase1_plan':str(first_path.resolve()),'phase1_plan_sha256':file_sha256(first_path),
        'harness':proof['harness'],'harness_sha256':proof['harness_sha256'],
        'resume_checkpoint':search['joint_handoff']['resume_checkpoint'],
        'condition':proof['condition'],'seed':proof['seed'],'new_model_calls_by_certifier':0}
    return handoff,first


def run_name(plan,job):
    return f"controlled-{plan['condition']}-seed{plan['seed']}-phase2-{job}"


def overrides(plan,path):
    checkpoint=Path(plan['resume_checkpoint']['directory'])
    return [*phase_overrides(plan['seed'],2,checkpoint),
        'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=ours.native_resume_observation.prepare_worker',
        f'+ray_kwargs.ray_init.runtime_env.env_vars.WHALE_JOINT_RESUME_PLAN={path.resolve()}']


def resolve(plan,name,path):
    from omegaconf import OmegaConf
    raw=subprocess.check_output(['bash',str(original.ROOT/'ours/run_native_rsft.sh'),
        *overrides(plan,path),'--cfg','job','--resolve'],cwd=original.ROOT,
        env=original.launch_environment(plan,name),text=True,stderr=subprocess.STDOUT)
    marker='model_engine: dp\n';require(marker in raw,'Missing native phase-two configuration')
    return raw,OmegaConf.to_container(OmegaConf.create(raw[raw.index(marker):]),resolve=True)


def probe_resume(config,plan):
    """Actual native task loader/restored data; actor RPC is only recorded on CPU."""
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer,AutoProcessor
    from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.dataset.rl_dataset import collate_fn
    cfg=OmegaConf.create(config);directory=check_layout(Path(plan['resume_checkpoint']['directory']))
    check_configuration(cfg,directory)
    observed=[]
    class ActorBoundary:
        def load_checkpoint(self,path,*,del_local_after_load):
            observed.append({'path':path,'del_local_after_load':del_local_after_load})
    python_state,numpy_state=random.getstate(),np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]),patch.dict(os.environ,HARNESS_PATH=plan['harness'],CHESS_PUZZLE_HARNESS_PATH=plan['harness']):
            tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
            processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True)
            dataset=create_rl_dataset(cfg.data.train_files,cfg.data,tokenizer,processor,is_train=True,max_samples=cfg.data.train_max_samples)
            ids=[row['puzzle_id'] for row in dataset.dataframe['extra_info']]
            require(ids==plan['native_dataset_order']['native_filtered_ids'],'Selected harness changed the filtered training dataset')
            trainer=RayPPOTrainer.__new__(RayPPOTrainer);trainer.config=cfg;trainer.global_steps=0;trainer.use_critic=False
            trainer.actor_rollout_wg=ActorBoundary()
            trainer._create_dataloader(dataset,dataset,collate_fn,create_rl_sampler(cfg.data,dataset))
            trainer._load_checkpoint();pending=observe_loader(trainer,directory)
            iterator=iter(trainer.train_dataloader)
            try:next_ids=[row['puzzle_id'] for row in next(iterator)['extra_info']]
            finally:
                if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
    require(next_ids==plan['native_dataset_order']['planned_batch_ids'][1] and
            observed==[{'path':str(directory/'actor'),'del_local_after_load':False}], 'Native restored data order or actor boundary differs')
    return {'kind':'joint_phase2_native_data_preflight','status':'PASS','next_batch_ids':next_ids,
        'pending_loader_observation':pending,'actor_load_rpc':observed,'actor_rpc_executed':False,
        'new_model_calls':0,'new_optimizer_steps':0,'source_sha256':file_sha256(Path(__file__))}


def prepare(path,search_path,completion_path):
    require(not path.exists() and not path.with_suffix('.yaml').exists(),'Preserve existing resumed-phase plan')
    handoff,first=certify_search(search_path,completion_path)
    plan=deepcopy(first);name=run_name(first,'cpu-preflight')
    plan.update(kind='controlled_joint_phase2_plan',created_at_utc=datetime.now(timezone.utc).isoformat(),
        phase=2,cpu_name=name,resume_checkpoint=handoff['resume_checkpoint'],incoming_search=handoff,
        harness=handoff['harness'],harness_sha256=handoff['harness_sha256'],
        resolved_config=str(path.with_suffix('.yaml').resolve()),
        limitations=['Second fresh batch of the independently trained joint condition; no weight-only prefix pooling.',
            'The native FP32 model/extra/data checkpoint is restored before rollout synchronization and generation.',
            'Adam state starts fresh under the released model/extra resume contract.',
            'CPU preflight records the actor RPC without executing it; GPU actor and loader receipts are separate evidence.',
            'Selected harness comes from a completed MH search; no heldout or VETO gain is inferred.'])
    raw,config=resolve(plan,name,path)
    require(native_data_order(config)==first['native_dataset_order'],'Selected harness changed native batch planning')
    plan['resume_data_preflight']=probe_resume(config,plan)
    with path.with_suffix('.yaml').open('x') as stream:stream.write(raw)
    additional=[*SOURCES,str(path.with_suffix('.yaml').resolve()),str(search_path.resolve()),str(completion_path.resolve()),
        handoff['phase1_plan'],plan['harness']]
    plan['source_sha256'].update({name:file_sha256(Path(name)) for name in additional})
    original.write_new(path,plan)
    return plan


def check(path):
    plan=json.loads(path.read_text())
    require(plan['kind']=='controlled_joint_phase2_plan' and plan['phase']==2,'Wrong resumed joint phase')
    handoff,first=certify_search(Path(plan['incoming_search']['search_plan']),Path(plan['incoming_search']['search_completion']))
    require(handoff==plan['incoming_search'] and plan['resume_checkpoint']==handoff['resume_checkpoint'] and
            plan['harness']==handoff['harness'] and plan['harness_sha256']==handoff['harness_sha256'],'Different phase-two handoff')
    shared=('condition','seed','model','dataset','initialization_report','initial_manifest','split_manifest','veto_mode',
        'native_batch_steps','fresh_trajectories','rollouts_per_prompt','max_assistant_output_tokens','max_policy_calls',
        'proposer_calls','randomization','versions','native_dataset_order','native_resume_contract','resources','preflight_order_contract')
    require(all(plan[k]==first[k] for k in shared),'Different joint training budget or shared execution')
    required=set(first['source_sha256'])|set(SOURCES)|{plan['resolved_config'],handoff['search_plan'],handoff['search_completion'],
        handoff['phase1_plan'],plan['harness']}
    require(required<=set(plan['source_sha256']),'Missing phase-two source binding')
    for name,digest in plan['source_sha256'].items():require(file_sha256(original.ROOT/name)==digest,f'Changed phase-two source: {name}')
    require(all(plan['source_sha256'][n]==v for n,v in first['source_sha256'].items()),'Changed first-phase provenance')
    require(plan['cpu_name']==run_name(plan,'cpu-preflight'),'Wrong native phase identity')
    checkpoint=plan['resume_checkpoint'];directory=check_layout(Path(checkpoint['directory']))
    require(checkpoint['step']==1 and len(checkpoint['artifact_sha256'])==5,'Missing native phase-one checkpoint')
    for name,digest in checkpoint['artifact_sha256'].items():require(file_sha256(directory/name)==digest,f'Changed native restore input: {name}')
    return plan


def preflight(path,plan,*,run=False):
    from omegaconf import OmegaConf
    name=run_name(plan,os.environ['SLURM_JOB_ID']) if run else plan['cpu_name']
    if run:
        import torch
        require(torch.cuda.device_count()==2 and all('H800' in torch.cuda.get_device_name(i) for i in range(2)),'Expected two H800 devices')
        require(not (original.ROOT/'data/native-rsft'/name).exists(),'Preserve existing phase-two attempt')
    raw,config=resolve(plan,name,path);cfg=OmegaConf.create(config)
    expected_raw=Path(plan['resolved_config']).read_text().replace(plan['cpu_name'],name);marker='model_engine: dp\n'
    expected=OmegaConf.to_container(OmegaConf.create(expected_raw[expected_raw.index(marker):]),resolve=True)
    require(config==expected,'Actual native second-phase configuration differs')
    check_configuration(cfg,Path(plan['resume_checkpoint']['directory']))
    require(cfg.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=='ours.native_resume_observation.prepare_worker' and
            cfg.ray_kwargs.ray_init.runtime_env.env_vars.WHALE_JOINT_RESUME_PLAN==str(path.resolve()),'Missing actual restore observation hook')
    require(native_data_order(config)==plan['native_dataset_order'] and probe_resume(config,plan)==plan['resume_data_preflight'],
            'Actual native restored data preflight differs')
    report={'kind':'controlled_joint_phase2_preflight','status':'PASS','plan_sha256':file_sha256(path),
        'run_name':name,'condition':plan['condition'],'seed':plan['seed'],'phase':2,'fresh_trajectories':64,
        'resume_checkpoint':plan['resume_checkpoint'],'harness_sha256':plan['harness_sha256'],
        'new_model_calls':0,'entrypoint_source_sha256':file_sha256(Path(__file__))}
    if run:
        output=original.ROOT/f"results/controlled-joint-phase2-{os.environ['SLURM_JOB_ID']}";output.mkdir(exist_ok=False)
        original.write_new(output/'start.json',report);(output/'effective-config.yaml').write_text(raw)
        os.execvpe('bash',['bash',str(original.ROOT/'ours/run_native_rsft.sh'),*overrides(plan,path)],original.launch_environment(plan,name))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','check','run'),required=True)
    for name in ('plan','search-plan','search-completion'):parser.add_argument('--'+name,type=Path,required=name=='plan')
    args=parser.parse_args();os.chdir(original.ROOT)
    if args.phase=='prepare':
        require(args.search_plan is not None and args.search_completion is not None,'Missing completed joint search')
        prepare(args.plan,args.search_plan,args.search_completion)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    else:
        plan=check(args.plan);print(json.dumps(preflight(args.plan,plan,run=args.phase=='run')),flush=True)
