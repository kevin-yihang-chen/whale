"""Bind E4 batches to the real native dataset and StatefulDataLoader.

The sampler's bare permutation is not the executed order under this runtime.
Materialize the two planned native batches and independently compare them with
the index-only loader probe. No trajectories, rewards or model calls are made.
"""
import hashlib
import os
from pathlib import Path
import random
from unittest.mock import patch

from .audit_native_training_batch import require
from .native_loader_schedule import native_schedule
from .visual_task import file_sha256


def native_data_order(config):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor,AutoTokenizer
    from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.dataset.rl_dataset import collate_fn
    cfg=OmegaConf.create(config)
    model=cfg.actor_rollout_ref.model.path
    harness=cfg.ray_kwargs.ray_init.runtime_env.env_vars.CHESS_PUZZLE_HARNESS_PATH
    python_state,numpy_state=random.getstate(),np.random.get_state()
    batches=[];snapshots=[]
    try:
        with torch.random.fork_rng(devices=[]),patch.dict(os.environ,HARNESS_PATH=harness,CHESS_PUZZLE_HARNESS_PATH=harness):
            tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
            processor=AutoProcessor.from_pretrained(model,local_files_only=True)
            dataset=create_rl_dataset(cfg.data.train_files,cfg.data,tokenizer,processor,
                                      is_train=True,max_samples=cfg.data.train_max_samples)
            ids=[row['puzzle_id'] for row in dataset.dataframe['extra_info']]
            require(len(ids)==len(set(ids))==128,'Incomplete native filtered training dataset')
            torch.manual_seed(int(cfg.data.seed))
            trainer=RayPPOTrainer.__new__(RayPPOTrainer);trainer.config=cfg
            trainer._create_dataloader(dataset,dataset,collate_fn,create_rl_sampler(cfg.data,dataset))
            iterator=iter(trainer.train_dataloader)
            try:
                for _ in range(2):
                    batch=next(iterator)
                    batches.append([row['puzzle_id'] for row in batch['extra_info']])
                    main=trainer.train_dataloader.state_dict()['_snapshot']['_main_snapshot']
                    sampled=main['_sampler_iter_state']['sampler_iter_state']
                    snapshots.append({'sampler_yielded':sampled['yielded'],
                        'generator_sha256':hashlib.sha256(sampled['generator'].numpy().tobytes()).hexdigest(),
                        'base_seed':main['_base_seed']})
            finally:
                if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
            independent=native_schedule(cfg,ids)
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
    require(all(len(batch)==8 for batch in batches) and len(set(sum(batches,[])))==16,
            'Unexpected native batch coverage')
    require(batches==independent['batch_ids'] and snapshots==independent['checkpoint_sampler_snapshots'],
            'Actual task DataLoader differs from index-only lifecycle probe')
    return {'order_contract':'native_stateful_dataloader_v1','native_dataset_class':type(dataset).__name__,
        'native_filtered_ids':ids,'planned_batch_ids':batches,'checkpoint_sampler_snapshots':snapshots,
        'independent_index_probe':independent,'order_source_sha256':file_sha256(Path(__file__)),
        'native_task_loader_executed':True,'new_model_calls':0,
        'scope':'First two real training batches before sampling; no bare-sampler order is substituted.'}
