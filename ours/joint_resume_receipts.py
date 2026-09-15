"""Verify the actual native restore receipts before auditing resumed E4 data.

The actor receipt is emitted only around the original checkpoint loader. This
reader checks its complete tensor layout and saved scheduler/RNG/data hashes;
it does not independently observe a GPU or replace the task trajectory audit.
"""
import json
from pathlib import Path

from .audit_native_training_batch import require
from .native_phase_resume import check_layout
from .native_resume_observation import state_digest
from .visual_task import file_sha256

CHECKPOINT_FILES=('data.pt','actor/model_world_size_1_rank_0.pt','actor/extra_state_world_size_1_rank_0.pt',
                  'actor/fsdp_config.json','actor/huggingface/config.json')
SOURCE=Path(__file__).resolve()


def checkpoint_layout(directory,step):
    directory=Path(directory).resolve()
    require(step in (1,2) and directory.name==f'global_step_{step}','Wrong checkpoint step')
    if step==1:return check_layout(directory)
    for name in CHECKPOINT_FILES:require((directory/name).is_file(),f'Missing native checkpoint artifact: {name}')
    require(json.loads((directory/'actor/fsdp_config.json').read_text())['world_size']==1,'Different native actor topology')
    require(not list((directory/'actor').glob('optim_world_size_*')),'Unexpected optimizer-state artifact')
    return directory


def verify_restore_receipts(plan,plan_path,job,receipt_root):
    """The caller first certifies the full phase-two plan and checkpoint bytes."""
    import torch
    require(plan['kind']=='controlled_joint_phase2_plan' and plan['phase']==2,'Wrong native resumed phase')
    checkpoint=plan['resume_checkpoint'];directory=checkpoint_layout(checkpoint['directory'],1)
    require(checkpoint['step']==1 and set(checkpoint['artifact_sha256'])==set(CHECKPOINT_FILES),'Incomplete incoming checkpoint binding')
    actor_path=receipt_root/'resume-actor-rank-0.json';loader_path=receipt_root/'resume-loader.json'
    actor=json.loads(actor_path.read_text());loader=json.loads(loader_path.read_text())
    observer=Path('ours/native_resume_observation.py');source_sha=file_sha256(observer)
    require(plan['source_sha256']['ours/native_resume_observation.py']==source_sha,'Changed bound restore observer')
    for record in (actor,loader):
        require(record['status']=='PASS' and record['job_id']==job and record['phase']==2 and
                record['condition']==plan['condition'] and record['seed']==plan['seed'] and
                record['plan_sha256']==file_sha256(plan_path) and record['source_sha256']==source_sha and
                record['harness_sha256']==plan['harness_sha256'] and record['new_model_calls']==0,'Wrong native restoration receipt identity')
    require(actor['kind']=='native_restored_actor_observation' and actor['device']=='cuda' and
            isinstance(actor['gpu_name'],str) and 'H800' in actor['gpu_name'] and
            actor['all_values_exact_native_fp32'] is True and actor['scheduler_exact'] is True and actor['rng_exact'] is True and
            actor['adam_moments_loaded'] is False and actor['optimizer_state_entries']==0 and actor['new_optimizer_steps']==0,
            'Missing exact native GPU actor restoration')
    model_path=directory/'actor/model_world_size_1_rank_0.pt';extra_path=directory/'actor/extra_state_world_size_1_rank_0.pt'
    expected_artifacts={str(directory/n):checkpoint['artifact_sha256'][n] for n in CHECKPOINT_FILES[1:3]}
    require(actor['artifact_sha256']==expected_artifacts,'Actor was restored from a different checkpoint')
    tensors=torch.load(model_path,map_location='cpu',mmap=True,weights_only=True)
    require(bool(tensors) and all(type(t) is torch.Tensor and t.dtype==torch.float32 for t in tensors.values()),
            'Different native checkpoint tensor interface')
    expected=[{'name':n,'shape':list(t.shape),'dtype':str(t.dtype),'elements':t.numel()} for n,t in sorted(tensors.items())]
    require(actor['tensors']==expected and actor['tensor_count_including_aliases']==len(expected) and
            actor['elements_including_aliases']==sum(t['elements'] for t in expected),'Incomplete restored actor layout')
    extra=torch.load(extra_path,map_location='cpu',weights_only=False)
    rng=extra['rng'];digest=state_digest(rng)
    require(set(rng)=={'cpu','cuda','numpy','random'} and actor['rng_keys']==sorted(rng) and
            actor['rng_before_observation_sha256']==actor['rng_after_observation_sha256']==digest,'Restored RNG differs from saved native state')
    saved=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    require(loader['kind']=='native_pending_loader_observation' and loader['loaded_global_step']==1 and
            loader['iterator_created_by_observer'] is False and loader['actor_observation_sha256']==file_sha256(actor_path) and
            loader['saved_state_sha256']==checkpoint['artifact_sha256']['data.pt'] and
            loader['pending_state_digest']==state_digest(saved),'Saved data progress was not restored without iteration')
    snapshot=plan['native_dataset_order']['checkpoint_sampler_snapshots'][0]
    require(all(loader[k]==v for k,v in snapshot.items()),'Observed incoming sampler differs from the native preflight')
    return {'kind':'joint_native_resume_receipt_audit','status':'PASS_RESTORED_NATIVE_STATE_RECEIPTS',
        'job_id':job,'condition':plan['condition'],'seed':plan['seed'],'plan_sha256':file_sha256(plan_path),
        'harness_sha256':plan['harness_sha256'],'resume_checkpoint':checkpoint,
        'tensor_count_including_aliases':len(expected),'elements_including_aliases':sum(t['elements'] for t in expected),
        'saved_rng_sha256':digest,'pending_loader_state_sha256':state_digest(saved),
        'artifact_sha256':{str(p):file_sha256(p) for p in (actor_path,loader_path)},
        'source_sha256':{str(SOURCE):file_sha256(SOURCE),str(observer):source_sha},'new_model_calls':0,
        'limitations':['Checks source-bound native GPU restoration receipts; this reader does not execute GPU loading.',
            'Actual next-batch IDs and replies require the full trajectory audit; vLLM receiver values are not measured here.']}
