"""E4 observations around native four-batch stages; no loss or optimizer changes."""
import asyncio
import json
import os
from pathlib import Path

from .native_resume_observation import observe_actor, publish, state_digest
from .visual_task import file_sha256


def context():
    path = Path(os.environ['VETO_COMPACT_TRAINING_PLAN']).resolve()
    plan = json.loads(path.read_text())
    if plan['kind'] != 'compact_chart_rsft_stage':
        raise ValueError('Wrong compact native stage')
    return path, plan


def checkpoint_class(base):
    class CompactCheckpointManager(base):
        _compact_observer = True

        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            path, plan = context()
            parent = plan['resume_checkpoint']
            if not parent or Path(local_path).resolve() != Path(parent['directory'])/'actor' or hdfs_path or del_local_after_load:
                raise ValueError('Unexpected or destructive native restoration')
            if getattr(self,'_compact_restored',False):
                raise ValueError('Repeated native restoration')
            for name, digest in parent['artifact_sha256'].items():
                if file_sha256(Path(parent['directory'])/name) != digest:
                    raise ValueError('Parent state changed before native load')
            result = super().load_checkpoint(local_path, hdfs_path=None, del_local_after_load=False)
            receipt = observe_actor(self,Path(parent['directory'])/'actor')
            receipt.update(plan_sha256=file_sha256(path),job_id=os.environ['SLURM_JOB_ID'])
            publish(Path(plan['output'])/'state/actor.json',receipt)
            self._compact_restored = True
            return result
    return CompactCheckpointManager


def trainer_class(base):
    class CompactTrainingObserver(base):
        _compact_observer = True

        def _load_checkpoint(self):
            import torch
            path, plan = context()
            cfg = self.config
            if cfg.trainer.del_local_ckpt_after_load or cfg.trainer.max_actor_ckpt_to_keep is not None:
                raise ValueError('Checkpoint deletion is not authorized')
            if cfg.data.visual_harness_path != plan['harness']:
                raise ValueError('Wrong continuation harness')
            result = super()._load_checkpoint()
            expected_step = 4 if plan['stage']==2 else 0
            if self.global_steps != expected_step:
                raise ValueError('Wrong restored training progress')
            loader = self.train_dataloader
            if list(loader.dataset.dataframe['visual_sample_id']) != plan['cpu_preflight']['all_sample_ids']:
                raise ValueError('Actual training dataset differs from the inspected population')
            receipt = {'status':'PASS_COMPACT_NATIVE_LOADER','global_step':expected_step,
                'plan_sha256':file_sha256(path),'iterator_created_by_observer':False}
            if expected_step:
                parent = Path(plan['resume_checkpoint']['directory'])
                saved = torch.load(parent/'data.pt',map_location='cpu',weights_only=False)
                if loader._iterator is not None or loader.next_iter_state is None or state_digest(loader.next_iter_state)!=state_digest(saved):
                    raise ValueError('Native loader did not restore the actual parent sampler')
                receipt['saved_state_digest']=state_digest(saved)
                actor=Path(plan['output'])/'state/actor.json'
                if json.loads(actor.read_text())['plan_sha256']!=file_sha256(path):
                    raise ValueError('Native actor restoration was not observed')
                receipt['actor_receipt_sha256']=file_sha256(actor)
            publish(Path(plan['output'])/'state/loader.json',receipt)
            return result
    return CompactTrainingObserver


def transport_class(base):
    from verl.utils.ray_utils import auto_await
    class CompactTransportObserver(base):
        _compact_observer = True

        @auto_await
        async def update_weights(self,global_steps=None):
            path,plan=context(); start=4 if plan['stage']==2 else 0
            if global_steps not in range(start,start+5) or self.backend!='nccl' or len(self.replicas)!=1:
                raise ValueError('Unexpected native weight transfer')
            output=Path(plan['output'])
            if global_steps==start:
                if not (output/'state/loader.json').exists() or list((output/'requests').glob('*.request.json')):
                    raise ValueError('Rollout started before verified restoration')
            result=await super().update_weights(global_steps)
            receiver=None
            if global_steps in (start,start+4):
                servers=[s for replica in self.replicas for s in replica.servers]
                if len(servers)!=1: raise ValueError('Unexpected rollout receiver count')
                await asyncio.gather(*[s.collective_rpc.remote('record_compact_weights',timeout=180.,
                    kwargs={'global_step':global_steps}) for s in servers])
                files=list((output/'state').glob(f'receiver-step{global_steps}-*.json'))
                if len(files)!=1: raise ValueError('Missing receiver observation')
                r=json.loads(files[0].read_text())
                if r['status']!='PASS_COMPACT_RECEIVER' or r['plan_sha256']!=file_sha256(path):
                    raise ValueError('Receiver weights differ')
                receiver=file_sha256(files[0])
            publish(output/f'state/transport-step{global_steps}.json',{
                'status':'NATIVE_TRANSPORT_RETURNED','global_step':global_steps,
                'plan_sha256':file_sha256(path),'receiver_receipt_sha256':receiver,
                'limitation':'Start/end receiver coordinates checked; intermediate native transfers recorded without checkpoints.'})
            return result
    return CompactTransportObserver


def prepare_worker():
    from .visual_pixel_bootstrap import prepare_worker as original
    result=original()
    from .fast_chart_transport import install_transport_preparation
    install_transport_preparation()
    import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint
    import verl.workers.fsdp_workers as workers
    import verl.trainer.main_textarena_disagg_rsft as trainer
    if not getattr(checkpoint.FSDPCheckpointManager,'_compact_observer',False):
        checkpoint.FSDPCheckpointManager=checkpoint_class(checkpoint.FSDPCheckpointManager)
        workers.FSDPCheckpointManager=checkpoint.FSDPCheckpointManager
        trainer.DisaggregatedRayTrainer=trainer_class(trainer.DisaggregatedRayTrainer)
        trainer.CheckpointEngineManager=transport_class(trainer.CheckpointEngineManager)
    return result
