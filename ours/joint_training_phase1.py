"""First E4 batch for the original WHALE and prompt-subspace controls.

Start each joint condition independently at the common untrained theta0, save
native model/extra/data state, then stop at cumulative batch one. The subsequent
MH search and native phase-two resume require their own completed handoff proof.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from . import controlled_pilot as original
from .audit_native_training_batch import require
from .controlled_data_order import native_data_order
from .experiment_randomization import TRIAL_SEEDS
from .native_phase_resume import phase_overrides
from .visual_task import file_sha256

SOURCES=('ours/joint_training_phase1.py','ours/audit_joint_training_phase1.py',
    'ours/run_joint_training_phase1.sh','ours/controlled_data_order.py','ours/native_loader_schedule.py',
    'ours/native_phase_resume.py')


def run_name(plan,job):
    return f"controlled-{plan['condition']}-seed{plan['seed']}-phase1-{job}"


def resolve(plan,name):
    from omegaconf import OmegaConf
    raw=subprocess.check_output(['bash',str(original.ROOT/'ours/run_native_rsft.sh'),
        *phase_overrides(plan['seed'],1),'--cfg','job','--resolve'],cwd=original.ROOT,
        env=original.launch_environment(plan,name),text=True,stderr=subprocess.STDOUT)
    marker='model_engine: dp\n';require(marker in raw,'Missing resolved native phase configuration')
    return raw,OmegaConf.to_container(OmegaConf.create(raw[raw.index(marker):]),resolve=True)


def prepare(path,condition,seed):
    require(condition in ('whale','whale_fst') and seed in TRIAL_SEEDS,'Outside declared joint controls')
    require(not path.exists() and not path.with_suffix('.yaml').exists(),'Preserve existing joint plan')
    cpu_name=f'controlled-{condition}-seed{seed}-phase1-cpu-preflight'
    write=original.write_new
    def frozen_write(destination,plan):
        plan.update(kind='controlled_joint_phase1_plan',condition=condition,phase=1,cpu_name=cpu_name,
            native_batch_steps=1,fresh_trajectories=64,max_assistant_output_tokens=64*8129,max_policy_calls=64*18,
            preflight_order_contract='native_stateful_dataloader_v1',resume_checkpoint=None,
            native_resume_contract={'save_load_contents':['model','extra'],'dataloader_state':'data.pt',
                'adam_moments_saved':False,'phase2_cumulative_target':2,'shared_weight_only_prefix':False},
            resources={'gpus':2,'gpu_type':'H800','cpus':16,'ram_gib':160,'time_limit_seconds':3600,'max_gpu_hours':2.},
            resource_decision={'single_gpu':'Not supported by the validated separate actor/rollout topology.',
                'two_gpus':'Same actor and rollout configuration as the completed controlled weight-only trial.',
                'four_gpus':'Independent-trial Ray isolation is unverified; no trial packing.',
                'queue_forecast':'Not queried by preparation; check active jobs before submission.',
                'merge_risk':'One native trainer, one batch; no sampled-prefix pooling or shard merge.'},
            limitations=['One of the two native training batches of the fixed joint-condition pilot.',
                'Common untrained theta0 and original h0; independently sampled for each condition/seed.',
                'Success filtering can produce zero or multiple optimizer minibatches.',
                'Native model/extra/data state is preserved; a fresh HF initialization is not a phase resume.',
                'Actual joint MH search, phase-two GPU resume and heldout comparison are still separate stages.'])
        plan['versions']['torchdata']=version('torchdata')
        plan['source_sha256'].update(plan['native_dataset_order']['independent_index_probe']['runtime_source_sha256'])
        write(destination,plan)
    with patch.object(original,'CPU_NAME',cpu_name),patch.object(original,'native_order',native_data_order), \
         patch.object(original,'training_overrides',lambda seed:phase_overrides(seed,1)), \
         patch.object(original,'SOURCES',(*original.SOURCES,*SOURCES)),patch.object(original,'write_new',frozen_write):
        original.prepare(path,seed)


def check(path):
    from .local_completion import checkpoint_manifest
    from .rsft_pilot_gate import check_tensor_bucket
    plan=json.loads(path.read_text())
    require(plan['kind']=='controlled_joint_phase1_plan' and plan['condition'] in ('whale','whale_fst') and
            plan['phase']==1 and plan['seed'] in TRIAL_SEEDS and plan['veto_mode']=='off','Wrong joint phase')
    require(plan['native_batch_steps']==1 and plan['fresh_trajectories']==64 and plan['rollouts_per_prompt']==8 and
            plan['max_assistant_output_tokens']==520256 and plan['max_policy_calls']==1152 and
            plan['proposer_calls']==0 and plan['resume_checkpoint'] is None,'Changed phase budget or initialization')
    require(set((*original.SOURCES,*SOURCES))<=set(plan['source_sha256']),'Missing joint-phase source binding')
    for name,digest in plan['source_sha256'].items():
        require(file_sha256(original.ROOT/name)==digest,f'Changed joint phase source: {name}')
    require({n:version(n) for n in plan['versions']}==plan['versions'],'Changed joint runtime')
    require(subprocess.check_output(['git','-C','upstream/WHALE','rev-parse','HEAD'],text=True).strip()==original.PIN and
            not subprocess.check_output(['git','-C','upstream/WHALE','status','--porcelain'],text=True).strip(),'Changed upstream')
    init=json.loads(Path(plan['initialization_report']).read_text())
    require(init['status']=='PASS' and init['optimizer_steps']==init['new_model_calls']==0 and
            init['audit']['exact_source_bf16_cast'] and init['exported']==plan['initial_manifest'] and
            checkpoint_manifest(Path(plan['model']))==plan['initial_manifest'],'Wrong common untrained initialization')
    baseline=original.ROOT/'upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py'
    require(Path(plan['harness']).resolve()==baseline.resolve(),'Joint phase one must use original h0')
    splits=json.loads(Path(plan['split_manifest']).read_text())['splits']
    train,mh,test=[set(splits[k]['puzzle_ids']) for k in ('train','mh_val','test')]
    require(not (train&mh or train&test or mh&test) and plan['dataset']==splits['train']['path'] and
            file_sha256(Path(plan['dataset']))==splits['train']['sha256'],'Changed or overlapping data roles')
    order=plan['native_dataset_order'];probe=order['independent_index_probe']
    require(plan['preflight_order_contract']==order['order_contract']=='native_stateful_dataloader_v1' and
            set(order['native_filtered_ids'])==train and order['planned_batch_ids']==probe['batch_ids'] and
            order['checkpoint_sampler_snapshots']==probe['checkpoint_sampler_snapshots'] and
            order['native_task_loader_executed'] and order['new_model_calls']==0,'Wrong native loader contract')
    require(plan['native_resume_contract']=={'save_load_contents':['model','extra'],'dataloader_state':'data.pt',
            'adam_moments_saved':False,'phase2_cumulative_target':2,'shared_weight_only_prefix':False},'Changed phase-resume contract')
    check_tensor_bucket(plan['model'],'float32',3072)
    return plan


def preflight(path,plan,*,run=False):
    from omegaconf import OmegaConf
    name=run_name(plan,os.environ['SLURM_JOB_ID']) if run else plan['cpu_name']
    if run:
        import torch
        require(torch.cuda.device_count()==2 and all('H800' in torch.cuda.get_device_name(i) for i in range(2)),
                'Expected two H800 devices')
        require(not (original.ROOT/'data/native-rsft'/name).exists(),'Preserve previous joint training attempt')
    raw,config=resolve(plan,name)
    expected_raw=Path(plan['resolved_config']).read_text().replace(plan['cpu_name'],name)
    marker='model_engine: dp\n'
    expected=OmegaConf.to_container(OmegaConf.create(expected_raw[expected_raw.index(marker):]),resolve=True)
    require(config==expected,'Actual native phase configuration differs')
    cfg=OmegaConf.create(config)
    require(cfg.trainer.total_training_steps==1 and cfg.trainer.online_rsft.iterations==0 and
            cfg.trainer.resume_mode=='disable' and not cfg.trainer.del_local_ckpt_after_load and
            list(cfg.actor_rollout_ref.actor.checkpoint.save_contents)==['model','extra'] and
            list(cfg.actor_rollout_ref.actor.checkpoint.load_contents)==['model','extra'],'Changed native save/stop contract')
    require(native_data_order(config)==plan['native_dataset_order'],'Actual task-loader schedule differs')
    report={'kind':'controlled_joint_phase1_preflight','status':'PASS','plan_sha256':file_sha256(path),
        'run_name':name,'condition':plan['condition'],'seed':plan['seed'],'phase':1,'fresh_trajectories':64,
        'new_model_calls':0,'entrypoint_source_sha256':file_sha256(Path(__file__))}
    if run:
        output=Path(f"results/controlled-joint-phase1-{os.environ['SLURM_JOB_ID']}");output.mkdir(exist_ok=False)
        original.write_new(output/'start.json',report);(output/'effective-config.yaml').write_text(raw)
        os.execvpe('bash',['bash',str(original.ROOT/'ours/run_native_rsft.sh'),*phase_overrides(plan['seed'],1)],
                   original.launch_environment(plan,name))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('prepare','check','run'),required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--condition',choices=('whale','whale_fst'))
    parser.add_argument('--seed',type=int,choices=TRIAL_SEEDS)
    args=parser.parse_args();os.chdir(original.ROOT)
    if args.phase=='prepare':
        prepare(args.plan,args.condition,args.seed)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
    else:
        plan=check(args.plan)
        print(json.dumps(preflight(args.plan,plan,run=args.phase=='run')),flush=True)
