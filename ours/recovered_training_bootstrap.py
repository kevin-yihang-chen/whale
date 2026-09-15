"""Resume one interrupted E4 step from its audited on-policy batch.

The native trainer retains filtering, loss, optimizer, save and weight transport.
Generation is replaced exactly once with recorded data from the same base model;
new inference calls are forbidden and source calls remain charged to their job.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from .audit_native_training_batch import load_batch, require
from .native_training_trace import json_value
from .visual_task import file_sha256


def restore_batch(directory, audit_path):
    import numpy as np
    import torch
    from verl import DataProto

    header, rows, receipt = load_batch(Path(directory))
    audit = json.loads(Path(audit_path).read_text())
    require(audit['kind'] == 'native_training_batch_audit' and audit['status'] == 'PASS', 'Batch audit did not pass')
    require(audit['batch_sha256'] == receipt['sha256'], 'Audited batch differs')
    require(audit['trajectories'] == len(rows) == 64 and audit['accepted'] == 2, 'Unexpected recovery batch')
    require(audit['job_id'] == header['context']['job_id'] == '221967', 'Unexpected source job')
    dtypes = {'torch.int64': torch.int64, 'torch.float32': torch.float32}
    tensors = {k: torch.tensor([r['tensors'][k] for r in rows], dtype=dtypes[s['dtype']])
               for k, s in header['tensors'].items()}
    tensors['rm_scores'] = tensors['token_level_scores'].clone()
    metadata = {k: np.array([r['metadata'][k] for r in rows], dtype=object)
                for k in ('extra_info', 'uid', 'extras', 'reward_extra_info')}
    for k in header['meta_info']['reward_extra_keys']:
        metadata[k] = np.array([r['metadata']['reward_extra_info'][k] for r in rows])
    return DataProto.from_dict(tensors=tensors, non_tensors=metadata,
                               meta_info=deepcopy(header['meta_info'])), receipt


class AuditedBatchRecovery:
    def __init__(self, original, batch, receipt, destination, plan):
        self.original, self.batch, self.receipt = original, batch, receipt
        self.destination, self.plan, self.used = Path(destination), plan, False

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate_sequences(self, incoming):
        require(not self.used, 'The interrupted batch can be consumed only once')
        require(len(incoming) == len(self.batch), 'Incoming batch size differs')
        require(json_value(incoming.non_tensor_batch['extra_info']) ==
                json_value(self.batch.non_tensor_batch['extra_info']), 'Incoming task order or content differs')
        require(incoming.meta_info['global_steps'] == 1, 'Recovery is restricted to the first step')
        # Original trainer generates fresh UUIDs for grouping. Preserve those new
        # identifiers for union; exact old identifiers remain in the source trace.
        self.batch.non_tensor_batch['uid'] = incoming.non_tensor_batch['uid'].copy()
        self.used = True
        report = {'kind': 'audited_native_batch_recovery', 'source_job_id': '221967',
                  'job_id': os.environ.get('SLURM_JOB_ID'), 'source_batch_sha256': self.receipt['sha256'],
                  'source_plan_sha256': self.plan['source_plan_sha256'], 'new_model_calls': 0,
                  'cached_trajectories': len(self.batch), 'original_calls_already_accounted': 69,
                  'incoming_order_verified': True, 'fresh_uuid_grouping': True}
        self.destination.mkdir(parents=True, exist_ok=True)
        (self.destination / 'recovery.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return self.batch


def install_recovery():
    import verl.experimental.agent_loop.agent_loop as agent_module
    import verl.trainer.main_textarena_disagg_rsft as trainer_module
    if getattr(trainer_module.DisaggregatedRayTrainer, '_audited_batch_recovery', False):
        return

    class NoInferenceServerManager(agent_module.AsyncLLMServerManager):
        async def chat_completion(self, *args, **kwargs):
            raise RuntimeError('Fresh inference is forbidden during audited batch recovery')

        async def generate(self, *args, **kwargs):
            raise RuntimeError('Fresh inference is forbidden during audited batch recovery')

    class RecoveredDisaggregatedTrainer(trainer_module.DisaggregatedRayTrainer):
        _audited_batch_recovery = True

        def init_workers(self):
            from omegaconf import OmegaConf
            from .recovery_gate import check_config, repository_path
            plan_path = Path(os.environ['WHALE_RSFT_REPLAY_PLAN'])
            plan = json.loads(plan_path.read_text())
            require(file_sha256(repository_path(plan['source_plan'])) == plan['source_plan_sha256'], 'Source plan changed')
            require(file_sha256(repository_path(plan['batch_audit'])) == plan['batch_audit_sha256'], 'Audit changed')
            # Validate the complete effective config before any worker loads weights.
            check_config(plan, OmegaConf.to_container(self.config, resolve=True), runtime=True)
            batch, receipt = restore_batch(repository_path(plan['batch_directory']), repository_path(plan['batch_audit']))
            super().init_workers()
            self.async_rollout_manager = AuditedBatchRecovery(self.async_rollout_manager, batch, receipt,
                Path(self.config.trainer.default_local_dir) / 'recovery', plan)

        def _update_sft_actor(self, batch):
            from .compact_native_batch import compact_batch
            compact, report = compact_batch(batch)
            destination = Path(self.config.trainer.default_local_dir) / 'recovery'
            (destination / 'compaction.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
            return super()._update_sft_actor(compact)

    agent_module.AsyncLLMServerManager = NoInferenceServerManager
    trainer_module.DisaggregatedRayTrainer = RecoveredDisaggregatedTrainer


def prepare_worker():
    from .training_bootstrap import prepare_worker as prepare_original_worker
    provenance = prepare_original_worker()
    install_recovery()
    return {'native_package_repair': provenance, 'audited_batch_recovery': True, 'new_model_calls': 0}
