"""Replay an audited selected-harness E4 batch through the original trainer.

The source checkpoint and complete batch are preserved. Only inference is
replaced once; native filtering, minibatches, optimizer, save and NCCL remain.
The separately frozen config selects WHALE's released Torch fused backend.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from .audit_native_training_batch import load_batch, require
from .native_training_trace import json_value
from .visual_task import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def restore_batch(directory, audit_path, plan):
    import numpy as np
    import torch
    from verl import DataProto
    header, rows, receipt = load_batch(Path(directory))
    audit = json.loads(Path(audit_path).read_text())
    require(audit['kind'] == 'native_alternation_batch_audit' and audit['status'] == 'PASS', 'Batch audit did not pass')
    require(audit['batch_sha256'] == receipt['sha256'] == plan['batch_sha256'], 'Different audited batch')
    require(audit['trajectories'] == len(rows) == plan['cached_trajectories'] == 64, 'Wrong recovery coverage')
    require(audit['accepted'] == plan['accepted'] > 0 and audit['accepted_loss_tokens'] == plan['loss_tokens'],
            'Wrong success-filtered batch')
    require(audit['job_id'] == header['context']['job_id'] == plan['source_job_id'], 'Wrong source job')
    require(header['context'] == audit['context'] and header['context']['configured_model_path'] == plan['model'],
            'Source runtime identity differs')
    dtypes = {'torch.int64': torch.int64, 'torch.float32': torch.float32}
    tensors = {key: torch.tensor([row['tensors'][key] for row in rows], dtype=dtypes[schema['dtype']])
               for key, schema in header['tensors'].items()}
    tensors['rm_scores'] = tensors['token_level_scores'].clone()
    metadata = {key: np.array([row['metadata'][key] for row in rows], dtype=object)
                for key in ('extra_info', 'uid', 'extras', 'reward_extra_info')}
    for key in header['meta_info']['reward_extra_keys']:
        metadata[key] = np.array([row['metadata']['reward_extra_info'][key] for row in rows])
    batch = DataProto.from_dict(tensors=tensors, non_tensors=metadata, meta_info=deepcopy(header['meta_info']))
    accepted = batch.batch['token_level_scores'].sum(-1) > .5
    require(int(accepted.sum()) == plan['accepted'] and
            int(batch.batch['response_mask'][accepted].sum()) == plan['loss_tokens'], 'Restored training objective differs')
    return batch, receipt


class RecordedAlternationBatch:
    def __init__(self, original, batch, receipt, destination, plan):
        self.original, self.batch, self.receipt = original, batch, receipt
        self.destination, self.plan, self.used = Path(destination), plan, False

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate_sequences(self, incoming):
        require(not self.used, 'The recorded batch may be consumed only once')
        require(len(incoming) == len(self.batch), 'Incoming trajectory count differs')
        require(json_value(incoming.non_tensor_batch['extra_info']) ==
                json_value(self.batch.non_tensor_batch['extra_info']), 'Incoming training order or content differs')
        require(incoming.meta_info['global_steps'] == 1, 'Recovery must begin at phase step one')
        self.batch.non_tensor_batch['uid'] = incoming.non_tensor_batch['uid'].copy()
        self.used = True
        self.destination.mkdir(parents=True, exist_ok=False)
        report = {'kind': 'recorded_alternation_batch_recovery', 'source_job_id': self.plan['source_job_id'],
            'job_id': os.environ.get('SLURM_JOB_ID'), 'source_batch_sha256': self.receipt['sha256'],
            'source_plan_sha256': self.plan['source_plan_sha256'], 'cached_trajectories': len(self.batch),
            'original_calls_already_accounted': self.plan['source_model_calls'], 'new_model_calls': 0,
            'incoming_order_verified': True, 'fresh_uuid_grouping': True,
            'limitations': ['Cached-return throughput is not inference throughput.',
                'Restart from the phase incoming checkpoint, not lost in-memory optimizer state.']}
        with (self.destination / 'recovery.json').open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(json.dumps(report), flush=True)
        return self.batch


def install_recovery():
    from .native_fused_chunk import install_chunk_size
    from .native_transport_memory import install_transport_release
    install_chunk_size(int(os.environ['WHALE_RSFT_FUSED_CHUNK_SIZE']))
    install_transport_release()
    import verl.experimental.agent_loop.agent_loop as agent_module
    import verl.trainer.main_textarena_disagg_rsft as trainer_module
    if getattr(trainer_module.DisaggregatedRayTrainer, '_recorded_alternation_recovery', False):
        return

    class NoInferenceServerManager(agent_module.AsyncLLMServerManager):
        async def chat_completion(self, *args, **kwargs):
            raise RuntimeError('Fresh inference is forbidden during recorded alternation recovery')

        async def generate(self, *args, **kwargs):
            raise RuntimeError('Fresh inference is forbidden during recorded alternation recovery')

    class RecordedAlternationTrainer(trainer_module.DisaggregatedRayTrainer):
        _recorded_alternation_recovery = True

        def init_workers(self):
            from omegaconf import OmegaConf
            from .alternation_recovery import check_configuration
            path = Path(os.environ['WHALE_ALTERNATION_REPLAY_PLAN'])
            plan = json.loads(path.read_text())
            for key, digest_key in [('source_plan', 'source_plan_sha256'), ('batch_audit', 'batch_audit_sha256')]:
                require(file_sha256(ROOT / plan[key]) == plan[digest_key], 'Frozen recovery provenance changed')
            check_configuration(plan, OmegaConf.to_container(self.config, resolve=True),
                                f"alternation-replay-{os.environ['SLURM_JOB_ID']}", runtime=True)
            batch, receipt = restore_batch(ROOT / plan['batch_directory'], ROOT / plan['batch_audit'], plan)
            super().init_workers()
            self.async_rollout_manager = RecordedAlternationBatch(self.async_rollout_manager, batch, receipt,
                Path(self.config.trainer.default_local_dir) / 'recovery', plan)

        def _update_sft_actor(self, batch):
            from .compact_native_batch import compact_batch
            compact, report = compact_batch(batch)
            path = Path(self.config.trainer.default_local_dir) / 'recovery/compaction.json'
            with path.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
            print(json.dumps(report), flush=True)
            return super()._update_sft_actor(compact)

    agent_module.AsyncLLMServerManager = NoInferenceServerManager
    trainer_module.DisaggregatedRayTrainer = RecordedAlternationTrainer


def prepare_worker():
    from .training_bootstrap import prepare_worker as original
    provenance = original()
    install_recovery()
    return {'native_package_repair': provenance, 'recorded_alternation_recovery': True,
            'native_fused_chunk_size': int(os.environ['WHALE_RSFT_FUSED_CHUNK_SIZE']), 'new_model_calls': 0}
