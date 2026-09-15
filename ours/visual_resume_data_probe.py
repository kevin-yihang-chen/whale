"""Read the actual visual E4 sampler checkpoint through the native loader.

No model is loaded, no trajectory is generated, and no training is performed.
The actor RPC is a recorded CPU boundary. This probes data continuation only;
GPU actor/scheduler/RNG restoration still requires a real subsequent E4 run.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from .native_phase_resume import check_configuration, check_layout
from .native_resume_observation import state_digest
from .native_visual_service import ROOT, write_new
from .visual_task import file_sha256


def probe(plan_path, output):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, AutoProcessor
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.dataset.rl_dataset import collate_fn
    plan = json.loads(plan_path.read_text())
    assert plan['kind'] == 'native_visual_rsft_engineering' and plan['role'] == 'W'
    for path, sha in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        assert file_sha256(ROOT / path) == sha, path
    directory = check_layout(Path(plan['output']) / 'checkpoints/global_step_1')
    cfg = OmegaConf.create(deepcopy(plan['config']))
    cfg.trainer.total_training_steps = 2
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(directory)
    cfg.trainer.del_local_ckpt_after_load = False
    check_configuration(cfg, directory)
    assert cfg.data.dataloader_num_workers == 0
    model = cfg.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    dataset = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, processor,
        is_train=True, max_samples=cfg.data.train_max_samples)
    assert list(dataset.dataframe['visual_sample_id']) == plan['cpu_preflight']['all_sample_ids']
    observed = []
    class ActorBoundary:
        def load_checkpoint(self, path, *, del_local_after_load):
            observed.append({'path': path, 'del_local_after_load': del_local_after_load})
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config, trainer.global_steps, trainer.use_critic = cfg, 0, False
    trainer.actor_rollout_wg = ActorBoundary()
    trainer._create_dataloader(dataset, dataset, collate_fn, create_rl_sampler(cfg.data, dataset))
    trainer._load_checkpoint()
    saved = torch.load(directory / 'data.pt', map_location='cpu', weights_only=False)
    pending = trainer.train_dataloader.next_iter_state
    assert trainer.global_steps == 1 and pending is not None and state_digest(pending) == state_digest(saved)
    # Do not assume the multiprocessing loader's _snapshot schema: this visual
    # configuration uses the native zero-worker StatefulDataLoader schema.
    assert trainer.train_dataloader._iterator is None
    expected_rpc = [{'path': str(directory / 'actor'), 'del_local_after_load': False}]
    assert observed == expected_rpc
    batch = next(iter(trainer.train_dataloader))
    ids = batch['visual_sample_id'].tolist()
    first = json.loads((Path(plan['output']) / 'checkpoints/audit/rollout-step1.json').read_text())
    previous = list(dict.fromkeys(first['sample_ids']))
    assert len(ids) == len(set(ids)) == len(previous) == 8 and not set(ids) & set(previous)
    assert set(ids) <= set(plan['cpu_preflight']['all_sample_ids'])
    assert all(any(p.get('type') == 'image' for message in prompt if isinstance(message['content'], list)
                   for p in message['content']) for prompt in batch['raw_prompt'])
    report = {'status': 'PASS_NATIVE_VISUAL_DATA_RESUME_CPU', 'training_plan_sha256': file_sha256(plan_path),
        'checkpoint': str(directory), 'saved_state_sha256': file_sha256(directory / 'data.pt'),
        'pending_state_digest': state_digest(saved), 'saved_state_keys': sorted(saved),
        'loaded_global_step': 1, 'previous_eight_ids': previous, 'next_eight_ids': ids,
        'actor_load_rpc': observed, 'actor_rpc_executed': False, 'model_calls': 0, 'optimizer_steps': 0,
        'source_sha256': file_sha256(Path(__file__)), 'images_present_in_next_batch': True,
        'limitations': ['Original incoming h0 and W dataset only; selected-harness filtering must be checked again before actual resume.',
            'Native CPU loader restore, not GPU actor/Adam/scheduler/RNG or training-performance evidence.']}
    write_new(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    print(json.dumps(probe(args.plan.resolve(), args.output.resolve())), flush=True)
