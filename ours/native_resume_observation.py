"""Observe the native E3 -> E4 restore without advancing model or data RNG.

The original checkpoint manager still performs loading. This extension checks
all restored actor values, scheduler and RNG before returning, while the driver
inspects the pending loader state directly instead of creating an iterator.
Actual GPU execution remains a separate experiment until these receipts exist.
"""
import hashlib
import json
import os
from pathlib import Path

from .audit_native_training_batch import require
from .native_phase_resume import check_configuration,check_layout
from .native_training_trace import json_value
from .visual_task import file_sha256


SOURCE=Path(__file__).resolve()
ROOT=SOURCE.parent.parent


def state_digest(value):
    return hashlib.sha256(json.dumps(json_value(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def observe_actor(manager,directory,*,expected_device='cuda'):
    """Compare the actual loaded state; CPU mode exists only for isolated tests."""
    import torch
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict
    require(expected_device in ('cuda','cpu'),'Unknown observation device')
    require(manager.rank==0 and manager.world_size==1 and manager.should_load_model and manager.should_load_extra and
            not manager.should_load_optimizer,'Wrong native actor restore contract')
    require(manager.optimizer is not None and not manager.optimizer.state,'Expected a fresh optimizer without restored Adam moments')
    devices={p.device.type for p in manager.model.parameters()}
    require(devices=={expected_device},'Actor is not on the required observation device')
    if expected_device=='cuda':
        require(torch.cuda.is_available() and 'H800' in torch.cuda.get_device_name() and
                manager.model.config.model_type=='qwen3_5','Expected the native H800 Qwen actor')
    model_path=directory/'model_world_size_1_rank_0.pt';extra_path=directory/'extra_state_world_size_1_rank_0.pt'
    expected=torch.load(model_path,map_location='cpu',mmap=True,weights_only=True)
    extra=torch.load(extra_path,map_location='cpu',weights_only=False)
    before=manager.get_rng_state();before_digest=state_digest(before)
    require(set(before)==set(extra['rng']) and before_digest==state_digest(extra['rng']), 'Native RNG was not restored exactly')
    require(expected_device=='cpu' or 'cuda' in before,'Missing actual CUDA RNG restoration')
    require(state_digest(manager.lr_scheduler.state_dict())==state_digest(extra['lr_scheduler']), 'Native scheduler was not restored exactly')
    actual=get_fsdp_full_state_dict(manager.model,offload_to_cpu=True,rank0_only=True)
    require(set(actual)==set(expected),'Restored actor graph differs from its checkpoint')
    records=[]
    for name in sorted(expected):
        a,b=expected[name],actual[name]
        require(type(a) is type(b) is torch.Tensor and a.dtype==b.dtype==torch.float32 and a.shape==b.shape,
                f'Restored actor layout differs: {name}')
        a,b=a.reshape(-1),b.detach().cpu().reshape(-1)
        for offset in range(0,a.numel(),1_000_000):
            require(torch.equal(a[offset:offset+1_000_000],b[offset:offset+1_000_000]) and
                    bool(torch.isfinite(b[offset:offset+1_000_000]).all()),f'Restored actor value differs: {name}')
        records.append({'name':name,'shape':list(expected[name].shape),'dtype':str(expected[name].dtype),'elements':a.numel()})
    after_digest=state_digest(manager.get_rng_state())
    require(after_digest==before_digest,'Observation advanced native RNG state')
    require(not manager.optimizer.state,'Observation initialized optimizer state')
    return {'kind':'native_restored_actor_observation','status':'PASS','device':expected_device,
        'gpu_name':torch.cuda.get_device_name() if expected_device=='cuda' else None,
        'tensor_count_including_aliases':len(records),'elements_including_aliases':sum(v['elements'] for v in records),
        'tensors':records,'all_values_exact_native_fp32':True,'scheduler_exact':True,'rng_exact':True,
        'rng_keys':sorted(before),'rng_before_observation_sha256':before_digest,'rng_after_observation_sha256':after_digest,
        'adam_moments_loaded':False,'optimizer_state_entries':0,
        'artifact_sha256':{str(p):file_sha256(p) for p in (model_path,extra_path)},
        'source_sha256':file_sha256(SOURCE),'new_model_calls':0,'new_optimizer_steps':0,
        'limitations':['Actor state after original load, before native rollout weight synchronization.',
            'This is not a measurement of the in-place vLLM receiver or a task performance gain.']}


def observe_loader(trainer,directory):
    """Read next_iter_state, never call the iterator-creating state_dict method."""
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    require(isinstance(trainer.train_dataloader,StatefulDataLoader),'Expected the native StatefulDataLoader')
    check_configuration(trainer.config,directory)
    require(trainer.global_steps==1 and trainer.train_dataloader._iterator is None and
            not trainer.train_dataloader._initial_iter_for_state_dict,'Unexpected resumed loader lifecycle')
    saved=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    pending=trainer.train_dataloader.next_iter_state
    require(pending is not None and state_digest(pending)==state_digest(saved),'Native loader did not retain exact saved progress')
    main=pending['_snapshot']['_main_snapshot'];sampled=main['_sampler_iter_state']['sampler_iter_state']
    return {'kind':'native_pending_loader_observation','status':'PASS','loaded_global_step':trainer.global_steps,
        'saved_state_sha256':file_sha256(directory/'data.pt'),'pending_state_digest':state_digest(pending),
        'sampler_yielded':sampled['yielded'],'generator_sha256':hashlib.sha256(sampled['generator'].numpy().tobytes()).hexdigest(),
        'base_seed':main['_base_seed'],'iterator_created_by_observer':False,'new_model_calls':0,
        'source_sha256':file_sha256(SOURCE),
        'limitations':['Exact pending native loader state before iteration, not a generated second batch.',
            'Actual next-batch IDs must be checked in the complete phase-two trajectory audit.']}


def resume_plan():
    path=Path(os.environ['WHALE_JOINT_RESUME_PLAN']).resolve();plan=json.loads(path.read_text())
    require(plan['kind']=='controlled_joint_phase2_plan' and plan['condition'] in ('whale','whale_fst') and
            plan['phase']==2 and plan['native_batch_steps']==1 and plan['fresh_trajectories']==64,'Wrong resumed phase')
    require(plan['source_sha256']['ours/native_resume_observation.py']==file_sha256(SOURCE), 'Changed resume observation source')
    checkpoint=plan['resume_checkpoint'];directory=check_layout(Path(checkpoint['directory']))
    require(checkpoint['step']==1,'Wrong native incoming phase')
    require(file_sha256(Path(plan['harness']))==plan['harness_sha256'],'Changed selected harness')
    return path,plan,directory


def publish(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def observed_manager_class(base,*,expected_device='cuda'):
    class ObservedCheckpointManager(base):
        _joint_resume_observation=True

        def load_checkpoint(self,local_path,hdfs_path=None,del_local_after_load=False):
            path,plan,directory=resume_plan()
            require(local_path is not None and Path(local_path).resolve()==directory/'actor' and
                    hdfs_path is None and not del_local_after_load,'Wrong or destructive native restore path')
            require(not getattr(self,'_joint_restore_completed',False),'Refuse repeated actor restoration')
            for name,digest in plan['resume_checkpoint']['artifact_sha256'].items():
                require(file_sha256(directory/name)==digest,f'Changed native restore input: {name}')
            require(not self.should_load_optimizer and self.optimizer is not None and not self.optimizer.state,
                    'Native second phase must start without Adam moments')
            result=super().load_checkpoint(local_path,hdfs_path=hdfs_path,del_local_after_load=del_local_after_load)
            report=observe_actor(self,directory/'actor',expected_device=expected_device)
            job=os.environ['SLURM_JOB_ID'];report.update(job_id=job,plan_sha256=file_sha256(path),
                condition=plan['condition'],seed=plan['seed'],phase=2,harness_sha256=plan['harness_sha256'])
            publish(ROOT/f'results/controlled-joint-phase2-{job}/resume-actor-rank-0.json',report)
            self._joint_restore_completed=True
            return result
    return ObservedCheckpointManager


def install_observation():
    import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint_module
    import verl.workers.fsdp_workers as worker_module
    import verl.trainer.main_textarena_disagg_rsft as trainer_module
    if getattr(checkpoint_module.FSDPCheckpointManager,'_joint_resume_observation',False):
        require(getattr(trainer_module.DisaggregatedRayTrainer,'_joint_resume_observation',False),'Partial resume-hook installation')
        return
    manager=observed_manager_class(checkpoint_module.FSDPCheckpointManager)
    checkpoint_module.FSDPCheckpointManager=manager;worker_module.FSDPCheckpointManager=manager
    class ObservedResumeTrainer(trainer_module.DisaggregatedRayTrainer):
        _joint_resume_observation=True

        def _load_checkpoint(self):
            path,plan,directory=resume_plan();check_configuration(self.config,directory)
            env=self.config.ray_kwargs.ray_init.runtime_env.env_vars
            require(str(env.CHESS_PUZZLE_HARNESS_PATH)==plan['harness'],'Wrong selected training harness')
            result=super()._load_checkpoint()
            report=observe_loader(self,directory);job=os.environ['SLURM_JOB_ID']
            actor=ROOT/f'results/controlled-joint-phase2-{job}/resume-actor-rank-0.json'
            observed=json.loads(actor.read_text())
            require(observed['status']=='PASS' and observed['device']=='cuda' and observed['plan_sha256']==file_sha256(path),
                    'No verified native GPU actor restoration')
            report.update(job_id=job,plan_sha256=file_sha256(path),condition=plan['condition'],seed=plan['seed'],phase=2,
                actor_observation_sha256=file_sha256(actor),harness_sha256=plan['harness_sha256'])
            publish(ROOT/f'results/controlled-joint-phase2-{job}/resume-loader.json',report)
            return result
    trainer_module.DisaggregatedRayTrainer=ObservedResumeTrainer


def prepare_worker():
    from .controlled_training_bootstrap import prepare_worker as shared_setup
    provenance=shared_setup();install_observation()
    print(json.dumps({'kind':'joint_native_resume_worker_setup','pid':os.getpid(),'observation_source_sha256':file_sha256(SOURCE),
                      'original_native_loader_retained':True,'new_model_calls':0}),flush=True)
    return provenance
