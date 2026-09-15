"""E4 native restoration and weight-transport observations for visual training.

Wrap the original loaders and checkpoint transport. No new loss, optimizer
state, random draws, candidate selection, data iteration or model generation.
"""
import asyncio
import json
import os
from pathlib import Path

from .native_phase_resume import check_configuration, check_layout
from .native_resume_observation import observe_actor, state_digest, publish
from .visual_task import file_sha256


def context():
    path = Path(os.environ['VETO_VISUAL_RESUME_PLAN']).resolve()
    plan = json.loads(path.read_text())
    assert plan['kind'] == 'native_visual_rsft_resume' and plan['role'] == 'W'
    assert file_sha256(Path(__file__)) == plan['source_sha256']['ours/visual_resume_bootstrap.py']
    directory = check_layout(Path(plan['resume_checkpoint']['directory']))
    assert file_sha256(Path(plan['accepted_harness'])) == plan['harness_sha256']
    assert file_sha256(Path(plan['handoff']['selection_path'])) == plan['handoff']['selection_sha256']
    return path, plan, directory


def observe_visual_loader(trainer, directory, expected_ids):
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    check_configuration(trainer.config, directory)
    loader = trainer.train_dataloader
    assert isinstance(loader, StatefulDataLoader) and loader.num_workers == 0
    assert trainer.global_steps == 1 and loader._iterator is None
    assert list(loader.dataset.dataframe['visual_sample_id']) == expected_ids
    saved = torch.load(directory / 'data.pt', map_location='cpu', weights_only=False)
    assert loader.next_iter_state is not None and state_digest(loader.next_iter_state) == state_digest(saved)
    return {'status': 'PASS_NATIVE_VISUAL_PENDING_LOADER', 'loaded_global_step': 1,
        'saved_state_sha256': file_sha256(directory / 'data.pt'), 'pending_state_digest': state_digest(saved),
        'saved_state_keys': sorted(saved), 'iterator_created_by_observer': False, 'model_calls': 0}


def checkpoint_class(base, *, expected_device='cuda'):
    class VisualResumeCheckpointManager(base):
        _visual_resume_observer = True

        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            path, plan, directory = context()
            assert Path(local_path).resolve() == directory / 'actor' and hdfs_path is None and not del_local_after_load
            assert not getattr(self, '_visual_restored', False)
            assert self.should_load_model and self.should_load_extra and not self.should_load_optimizer
            for name, digest in plan['resume_checkpoint']['artifact_sha256'].items():
                assert file_sha256(directory / name) == digest
            value = super().load_checkpoint(local_path, hdfs_path=hdfs_path, del_local_after_load=False)
            report = observe_actor(self, directory / 'actor', expected_device=expected_device)
            report.update(plan_sha256=file_sha256(path), harness_sha256=plan['harness_sha256'],
                job_id=os.environ['SLURM_JOB_ID'])
            publish(Path(plan['output']) / 'resume/actor.json', report)
            self._visual_restored = True
            return value
    return VisualResumeCheckpointManager


def trainer_class(base):
    class VisualResumeTrainer(base):
        _visual_resume_observer = True

        def _load_checkpoint(self):
            path, plan, directory = context()
            check_configuration(self.config, directory)
            assert self.config.data.visual_harness_path == plan['accepted_harness']
            assert self.config.data.visual_harness_execution == 'isolated'
            value = super()._load_checkpoint()
            report = observe_visual_loader(self, directory, plan['cpu_preflight']['all_sample_ids'])
            actor_path = Path(plan['output']) / 'resume/actor.json'
            actor = json.loads(actor_path.read_text())
            assert actor['status'] == 'PASS' and actor['device'] == 'cuda' and actor['plan_sha256'] == file_sha256(path)
            report.update(plan_sha256=file_sha256(path), actor_receipt_sha256=file_sha256(actor_path),
                harness_sha256=plan['harness_sha256'], job_id=os.environ['SLURM_JOB_ID'])
            publish(Path(plan['output']) / 'resume/loader.json', report)
            return value
    return VisualResumeTrainer


def transport_class(base):
    from verl.utils.ray_utils import auto_await
    class VisualResumeCheckpointTransport(base):
        _visual_resume_observer = True

        @auto_await
        async def update_weights(self, global_steps=None):
            path, plan, _ = context()
            assert global_steps in (1, 2) and self.backend == 'nccl' and len(self.replicas) == 1
            output = Path(plan['output'])
            if global_steps == 1:
                for name in ('actor', 'loader'):
                    assert json.loads((output / f'resume/{name}.json').read_text())['plan_sha256'] == file_sha256(path)
                assert not list((output / 'requests').glob('*.request.json'))
            result = await super().update_weights(global_steps)
            servers = [server for replica in self.replicas for server in replica.servers]
            assert len(servers) == 1
            await asyncio.gather(*[server.collective_rpc.remote('record_native_resumed_weights', timeout=120.,
                kwargs={'global_step': global_steps}) for server in servers])
            receipts = list((output / 'resume').glob(f'receiver-step{global_steps}-*.json'))
            assert len(receipts) == 1
            report = json.loads(receipts[0].read_text())
            assert report['status'] == 'PASS_NATIVE_RECEIVER_COORDINATES' and report['global_step'] == global_steps
            assert report['plan_sha256'] == file_sha256(path)
            publish(output / 'resume' / f'transport-step{global_steps}.json', {
                'status': 'PASS_NATIVE_WEIGHT_TRANSPORT_OBSERVED', 'global_step': global_steps,
                'plan_sha256': file_sha256(path), 'receiver_sha256': file_sha256(receipts[0]),
                'new_model_calls': 0, 'new_optimizer_steps': 0})
            return result
    return VisualResumeCheckpointTransport


def prepare_worker():
    from .visual_pixel_bootstrap import prepare_worker as original
    provenance = original()
    import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint
    import verl.workers.fsdp_workers as worker
    import verl.trainer.main_textarena_disagg_rsft as trainer
    if not getattr(checkpoint.FSDPCheckpointManager, '_visual_resume_observer', False):
        checkpoint.FSDPCheckpointManager = checkpoint_class(checkpoint.FSDPCheckpointManager)
        worker.FSDPCheckpointManager = checkpoint.FSDPCheckpointManager
        trainer.DisaggregatedRayTrainer = trainer_class(trainer.DisaggregatedRayTrainer)
        trainer.CheckpointEngineManager = transport_class(trainer.CheckpointEngineManager)
    assert getattr(trainer.DisaggregatedRayTrainer, '_visual_resume_observer', False)
    assert getattr(trainer.CheckpointEngineManager, '_visual_resume_observer', False)
    return provenance
