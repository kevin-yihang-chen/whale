"""Derive task order through the actual native StatefulDataLoader lifecycle.

Directly iterating RandomSampler omits DataLoader construction/reset draws in
the installed Torch/TorchData combination. This diagnostic uses index-only
rows of the already-verified filtered dataset, not task labels or model output.
It leaves the native sampler and loader behavior unchanged.
"""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import inspect
from pathlib import Path
from unittest.mock import patch

from .audit_native_training_batch import require
from .visual_task import file_sha256


def identity_collation(rows):
    return rows


def native_schedule(config, filtered_ids, steps=2):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import torch
    from verl.trainer.main_ppo import create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from torchdata.stateful_dataloader import StatefulDataLoader
    require(len(filtered_ids)==len(set(filtered_ids))==128 and steps in (1,2),'Unexpected pilot scope')
    cfg=deepcopy(config);dataset=list(range(len(filtered_ids)))
    sampler=create_rl_sampler(cfg.data,dataset)
    original_iter=type(sampler).__iter__
    construction=[]
    digest=lambda tensor:hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    def recorded_iter(instance):
        before=digest(instance.generator.get_state())
        result=original_iter(instance)
        construction.append({'before_sha256':before,'after_sha256':digest(instance.generator.get_state())})
        return result
    batches=[];snapshots=[]
    with torch.random.fork_rng(devices=[]),patch.object(type(sampler),'__iter__',recorded_iter):
        torch.manual_seed(int(cfg.data.seed))
        trainer=RayPPOTrainer.__new__(RayPPOTrainer);trainer.config=cfg
        trainer._create_dataloader(dataset,dataset,identity_collation,sampler)
        iterator=iter(trainer.train_dataloader)
        try:
            for _ in range(steps):
                indices=next(iterator)
                batches.append([filtered_ids[i] for i in indices])
                state=trainer.train_dataloader.state_dict()
                main=state['_snapshot']['_main_snapshot']
                sampled=main['_sampler_iter_state']['sampler_iter_state']
                snapshots.append({'sampler_yielded':sampled['yielded'],
                    'generator_sha256':digest(sampled['generator']),'base_seed':main['_base_seed']})
        finally:
            if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
    paths={str(Path(inspect.getfile(value)).resolve()) for value in
           (torch.utils.data.dataloader._BaseDataLoaderIter,StatefulDataLoader,type(sampler),RayPPOTrainer,create_rl_sampler)}
    return {'kind':'native_dataloader_schedule','seed':int(cfg.data.seed),'batch_ids':batches,
        'checkpoint_sampler_snapshots':snapshots,'sampler_iterator_constructions':construction,
        'runtime_versions':{name:version(name) for name in ('torch','torchdata')},
        'runtime_source_sha256':{name:file_sha256(Path(name)) for name in sorted(paths)},
        'schedule_source_sha256':file_sha256(Path(__file__)),'task_labels_read':False,
        'model_calls':0,'data_rows':'Indices into the preverified filtered dataset; original native loader lifecycle.'}
