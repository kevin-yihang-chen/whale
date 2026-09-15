"""E4 continuation reference derived from the supplied native sampler state."""
from copy import deepcopy
import json
from pathlib import Path

from .native_phase_resume import check_configuration, check_layout
from .native_resume_observation import state_digest
from .visual_task import file_sha256


def restore_next_batch(loader, state_path):
    import torch
    saved = torch.load(state_path, map_location='cpu', weights_only=False)
    loader.load_state_dict(saved)
    if state_digest(loader.next_iter_state) != state_digest(saved):
        raise ValueError('Independent loader did not restore the actual saved sampler state')
    with torch.random.fork_rng(devices=[]):
        batch = next(iter(loader))
    return batch, {'saved_state_sha256': file_sha256(Path(state_path)), 'saved_state_digest': state_digest(saved)}


def reference(first):
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    cfg = OmegaConf.create(deepcopy(first['config']))
    directory = check_layout(Path(first['output']) / 'checkpoints/global_step_1')
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(directory)
    cfg.trainer.total_training_steps = 2
    cfg.trainer.del_local_ckpt_after_load = False
    check_configuration(cfg, directory)
    for key in ('manifest', 'parquet'):
        if file_sha256(Path(first['training_data'][key])) != first['training_data'][key + '_sha256']:
            raise ValueError('The first-stage training inputs changed')
    model = cfg.actor_rollout_ref.model.path
    dataset = create_rl_dataset(cfg.data.train_files, cfg.data,
        AutoTokenizer.from_pretrained(model, local_files_only=True),
        AutoProcessor.from_pretrained(model, local_files_only=True), is_train=True,
        max_samples=cfg.data.train_max_samples)
    ids = list(dataset.dataframe['visual_sample_id'])
    if ids != first['cpu_preflight']['all_sample_ids']:
        raise ValueError('Original-harness dataset filtering changed the sampler population')
    loader = StatefulDataLoader(dataset, batch_size=cfg.data.train_batch_size, num_workers=0,
        drop_last=True, collate_fn=collate_fn, sampler=create_rl_sampler(cfg.data, dataset))
    batch, report = restore_next_batch(loader, directory / 'data.pt')
    next_ids = batch['visual_sample_id'].tolist()
    previous = json.loads((directory.parent / 'audit/rollout-step1.json').read_text())
    previous_ids = list(dict.fromkeys(previous['sample_ids']))
    if len(next_ids) != cfg.data.train_batch_size or len(set(next_ids)) != len(next_ids):
        raise ValueError('Unexpected next-batch coverage')
    if set(next_ids) & set(previous_ids) or not set(next_ids) <= set(ids):
        raise ValueError('The native next batch repeats or leaves the first-stage population')
    report.update(first_plan_sha256=file_sha256(Path(first['output']) / 'plan.json'),
        original_harness_sha256=file_sha256(Path(cfg.data.visual_harness_path)),
        next_eight_ids=next_ids, previous_eight_ids=previous_ids, model_calls=0)
    return report
