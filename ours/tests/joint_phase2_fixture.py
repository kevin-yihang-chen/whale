"""Synthetic phase-two provenance around unchanged archived native data.

All generated records stay in test temporary directories. GPU identity/terminal
fields are synthetic fixtures and never constitute a real joint experiment.
"""
import hashlib
import json
from pathlib import Path
import shutil

from ours.joint_resume_receipts import CHECKPOINT_FILES
from ours.native_resume_observation import state_digest
from ours.native_search import write_json
from ours.visual_task import file_sha256


def incoming_checkpoint(root,source):
    import torch
    directory=root/'global_step_1';(directory/'actor/huggingface').mkdir(parents=True)
    for name in CHECKPOINT_FILES:
        if name!='actor/model_world_size_1_rank_0.pt':shutil.copyfile(source/name,directory/name)
    torch.save({'fixture.weight':torch.arange(6,dtype=torch.float32).reshape(2,3)},directory/'actor/model_world_size_1_rank_0.pt')
    return {'step':1,'directory':str(directory),'artifact_sha256':{n:file_sha256(directory/n) for n in CHECKPOINT_FILES}}


def restore_receipts(plan,plan_path,job,root):
    import torch
    root.mkdir(parents=True,exist_ok=True);directory=Path(plan['resume_checkpoint']['directory'])
    model=torch.load(directory/'actor/model_world_size_1_rank_0.pt',map_location='cpu',weights_only=True)
    extra=torch.load(directory/'actor/extra_state_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
    data=torch.load(directory/'data.pt',map_location='cpu',weights_only=False)
    main=data['_snapshot']['_main_snapshot'];sampled=main['_sampler_iter_state']['sampler_iter_state']
    records=[{'name':n,'shape':list(t.shape),'dtype':str(t.dtype),'elements':t.numel()} for n,t in sorted(model.items())]
    shared={'status':'PASS','job_id':job,'phase':2,'condition':plan['condition'],'seed':plan['seed'],
        'plan_sha256':file_sha256(plan_path),'harness_sha256':plan['harness_sha256'],
        'source_sha256':file_sha256(Path('ours/native_resume_observation.py')),'new_model_calls':0,
        'fixture_only':'Synthetic GPU identity/provenance for CPU validation; no actual GPU restore.'}
    actor={**shared,'kind':'native_restored_actor_observation','device':'cuda','gpu_name':'H800 synthetic CPU receipt fixture',
        'all_values_exact_native_fp32':True,'scheduler_exact':True,'rng_exact':True,'adam_moments_loaded':False,
        'optimizer_state_entries':0,'new_optimizer_steps':0,'tensors':records,'tensor_count_including_aliases':len(records),
        'elements_including_aliases':sum(v['elements'] for v in records),'rng_keys':sorted(extra['rng']),
        'rng_before_observation_sha256':state_digest(extra['rng']),'rng_after_observation_sha256':state_digest(extra['rng']),
        'artifact_sha256':{str(directory/n):plan['resume_checkpoint']['artifact_sha256'][n] for n in CHECKPOINT_FILES[1:3]}}
    write_json(root/'resume-actor-rank-0.json',actor)
    loader={**shared,'kind':'native_pending_loader_observation','loaded_global_step':1,'iterator_created_by_observer':False,
        'actor_observation_sha256':file_sha256(root/'resume-actor-rank-0.json'),'saved_state_sha256':file_sha256(directory/'data.pt'),
        'pending_state_digest':state_digest(data),'sampler_yielded':sampled['yielded'],
        'generator_sha256':hashlib.sha256(sampled['generator'].numpy().tobytes()).hexdigest(),'base_seed':main['_base_seed']}
    write_json(root/'resume-loader.json',loader)
    return actor,loader
