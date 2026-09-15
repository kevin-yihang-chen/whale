"""Bind E3 -> E4 to native model/extra and sampler-state continuation.

The released alternating launcher changes the cumulative training-step target.
Its checkpoint contract restores model, scheduler and RNG, but excludes Adam
moments. This module preserves that contract and rejects missing sampler state,
which the upstream loader otherwise only warns about. It does not authorize
sharing a weight-only prefix or replace a complete training/checkpoint audit.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from .audit_native_training_batch import require
from .controlled_conditions import native_training_overrides
from .experiment_randomization import TRIAL_SEEDS
from .native_search import write_json
from .visual_task import file_sha256


def phase_overrides(seed, target_step, checkpoint=None):
    require(seed in TRIAL_SEEDS and target_step in (1,2), 'Outside controlled two-phase pilot')
    require((target_step==1) == (checkpoint is None), 'Phase two requires its native phase-one checkpoint')
    values=[*native_training_overrides(seed),'trainer.online_rsft.iterations=0',
        f'trainer.total_training_steps={target_step}','trainer.max_actor_ckpt_to_keep=2',
        'trainer.del_local_ckpt_after_load=false']
    if checkpoint is None:
        return [*values,'trainer.resume_mode=disable']
    directory=Path(checkpoint).resolve()
    require(directory.name=='global_step_1','Expected native phase-one global-step directory')
    return [*values,'trainer.resume_mode=resume_path',f'trainer.resume_from_path={directory}']


def check_layout(directory):
    directory=Path(directory).resolve()
    require(directory.name=='global_step_1','Unexpected incoming global step')
    required=('data.pt','actor/model_world_size_1_rank_0.pt',
        'actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','actor/huggingface/config.json')
    for name in required:
        require((directory/name).is_file(),f'Missing native resume artifact: {name}')
    require(json.loads((directory/'actor/fsdp_config.json').read_text())['world_size']==1,
            'Different actor topology')
    require(not list((directory/'actor').glob('optim_world_size_*')),'Unexpected optimizer-state artifact')
    return directory


def check_configuration(config, directory):
    require(config.trainer.total_training_steps==2 and config.trainer.online_rsft.iterations==0,
            'Resume must stop at cumulative native batch two')
    require(config.trainer.resume_mode=='resume_path' and
            Path(config.trainer.resume_from_path).resolve()==directory.resolve(), 'Wrong native checkpoint path')
    require(not config.trainer.del_local_ckpt_after_load,'Resume must preserve its source checkpoint')
    checkpoint=config.actor_rollout_ref.actor.checkpoint
    require(list(checkpoint.load_contents)==list(checkpoint.save_contents)==['model','extra'] and
            not checkpoint.async_save,'Different native checkpoint-content contract')


def probe_saved_data_progress(plan_path, directory):
    """Use actual RLHFDataset/loader and native resume; actor RPC is recorded only."""
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer,AutoProcessor
    from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.dataset.rl_dataset import collate_fn
    from .controlled_pilot import check
    from .native_loader_schedule import native_schedule
    plan=check(plan_path)
    directory=check_layout(directory)
    raw=Path(plan['resolved_config']).read_text()
    config=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
    config.trainer.resume_mode='resume_path';config.trainer.resume_from_path=str(directory)
    config.trainer.del_local_ckpt_after_load=False
    check_configuration(config,directory)
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True)
    with patch.dict('os.environ',HARNESS_PATH=plan['harness'],CHESS_PUZZLE_HARNESS_PATH=plan['harness']):
        dataset=create_rl_dataset(config.data.train_files,config.data,tokenizer,processor,
                                  is_train=True,max_samples=config.data.train_max_samples)
    filtered_ids=[row['puzzle_id'] for row in dataset.dataframe['extra_info']]
    require(filtered_ids==plan['native_dataset_order']['native_filtered_ids'],'Filtered dataset order changed')
    schedule=native_schedule(config,filtered_ids)
    observed=[]
    class ActorBoundary:
        def load_checkpoint(self,path,*,del_local_after_load):
            observed.append({'path':path,'del_local_after_load':del_local_after_load})
    trainer=RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config=config;trainer.global_steps=0;trainer.use_critic=False
    trainer.actor_rollout_wg=ActorBoundary()
    trainer._create_dataloader(dataset,dataset,collate_fn,create_rl_sampler(config.data,dataset))
    trainer._load_checkpoint()
    iterator=iter(trainer.train_dataloader)
    try:
        batch=next(iterator)
        ids=[row['puzzle_id'] for row in batch['extra_info']]
    finally:
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
    expected_ids=schedule['batch_ids'][1]
    require(trainer.global_steps==1 and ids==expected_ids,
            f'Native resume differs: step={trainer.global_steps}, actual={ids}, expected={expected_ids}')
    require(observed==[{'path':str(directory/'actor'),'del_local_after_load':False}],
            'Native trainer called a different actor resume boundary')
    old_ids=plan['native_dataset_order']['planned_batch_ids'][1]
    return {'kind':'native_saved_data_progress_probe',
        'status':'PASS_NATIVE_RESUME' if ids==old_ids else 'PASS_NATIVE_RESUME_WITH_PREFLIGHT_SCHEDULE_DEVIATION',
        'training_plan_sha256':file_sha256(plan_path),'original_preflight_next_ids':old_ids,
        'matches_original_preflight_ids':ids==old_ids,'native_loader_schedule':schedule,
        'checkpoint':str(directory),'loaded_global_step':trainer.global_steps,'next_puzzle_ids':ids,
        'actor_load_rpc':observed,'native_dataset_class':type(dataset).__name__,
        'artifact_sha256':{name:file_sha256(directory/name) for name in
            ('data.pt','actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json')},
        'source_sha256':{name:file_sha256(Path(name)) for name in ('ours/native_phase_resume.py',
            'upstream/WHALE/domains/chess_puzzles/verl/trainer/ppo/ray_trainer.py')},
        'new_model_calls':0,'new_optimizer_steps':0,
        'limitations':['Actual saved sampler state and native dataset were loaded on CPU.',
            'Actor RPC was recorded, not executed; this does not prove GPU model/RNG loading.',
            'Any disagreement with the original preflight remains a recorded protocol deviation.',
            'This probe alone does not authorize reusing this weight-only prefix for a joint condition.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve existing resume evidence')
    report=probe_saved_data_progress(args.plan,args.checkpoint)
    write_json(args.output,report)
    print(json.dumps(report),flush=True)
