"""Audit every first-phase joint-training trajectory and native request.

Replay the original Chess agent with recorded replies, then independently
verify board moves and training masks. A completed batch may have no accepted
rows. This audit does not assert optimizer completion or a serving handoff.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from .audit_alternation_training import replay_row
from .audit_native_training_batch import RequestPool,load_batch,require
from .joint_training_phase1 import check,run_name
from .native_search import write_json
from .visual_task import file_sha256


def runtime_configuration(plan,job):
    from omegaconf import OmegaConf
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    require(re.fullmatch(r'[0-9]+',job) is not None,'Invalid job identity')
    raw=Path(plan['resolved_config']).read_text().replace(plan['cpu_name'],run_name(plan,job))
    cfg=migrate_legacy_reward_impl(OmegaConf.create(raw[raw.index('model_engine: dp\n'):]))
    require(cfg.trainer.total_training_steps==1 and cfg.trainer.online_rsft.iterations==0 and
            cfg.trainer.resume_mode=='disable','Wrong first-phase stopping/resume rule')
    require(not cfg.reward.reward_model.enable_resource_pool,'Unexpected reward resource pool')
    cfg.reward.reward_model.nnodes=cfg.trainer.nnodes
    cfg.reward.reward_model.n_gpus_per_node=cfg.trainer.n_gpus_per_node
    cfg.actor_rollout_ref.actor.optim.total_training_steps=cfg.trainer.total_training_steps
    cfg.critic.optim.total_training_steps=cfg.trainer.total_training_steps
    context={'configured_model_path':plan['model'],'job_id':job,'recording_only':True,
        'configuration_sha256':hashlib.sha256(json.dumps(OmegaConf.to_container(cfg,resolve=True),sort_keys=True).encode()).hexdigest()}
    return cfg,context


async def audit(plan_path,directory):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from autoharness_chess_puzzle.harness import load_harness
    plan=check(plan_path)
    require(sorted(p.name for p in directory.glob('batch-*.jsonl.gz'))==['batch-1.jsonl.gz'],'Missing or extra sampled batch')
    header,rows,receipt=load_batch(directory,step=1);job=header['context']['job_id']
    config,context=runtime_configuration(plan,job)
    require(directory.resolve()==(Path(config.trainer.default_local_dir)/'audit').resolve(),'Wrong joint audit directory')
    start_path=Path(f'results/controlled-joint-phase1-{job}/start.json')
    start=json.loads(start_path.read_text())
    require(start['status']=='PASS' and start['plan_sha256']==file_sha256(plan_path) and
            start['condition']==plan['condition'] and start['seed']==plan['seed'] and start['phase']==1 and
            start['run_name']==run_name(plan,job),'Wrong joint start receipt')
    require(header['context']==context and len(rows)==plan['fresh_trajectories']==64,'Wrong batch context or row count')
    env=OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars,resolve=True)
    require(env['CHESS_PUZZLE_HARNESS_PATH']==plan['harness'],'Wrong phase-one harness')
    os.environ.update({k:str(v) for k,v in env.items() if k.startswith('CHESS_PUZZLE_')})
    harness=load_harness(Path(plan['harness']));tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    data=pq.read_table(plan['dataset'],use_threads=False).to_pylist();by_id={r['extra_info']['puzzle_id']:r['extra_info'] for r in data}
    require(len(by_id)==len(data)==128 and set(by_id)==set(plan['native_dataset_order']['native_filtered_ids']),
            'Wrong frozen training coverage')
    expected=plan['native_dataset_order']['planned_batch_ids'][0]
    require([r['metadata']['extra_info']['puzzle_id'] for r in rows]==[key for key in expected for _ in range(8)],
            'Actual sampled rows differ from frozen native DataLoader order')
    prompt,response=[int(config.actor_rollout_ref.rollout[k]) for k in ('prompt_length','response_length')];width=prompt+response
    shapes={'prompts':[prompt],'responses':[response],'response_mask':[response],'attention_mask':[width],
            'input_ids':[width],'position_ids':[4,width],'token_level_scores':[response]}
    require(set(header['tensors'])==set(shapes),'Wrong batch tensor interface')
    for key,shape in shapes.items():
        dtype='torch.float32' if key=='token_level_scores' else 'torch.int64'
        require(header['tensors'][key]=={'shape':[64]+shape,'dtype':dtype},f'Wrong tensor layout: {key}')
    require(receipt['recorder_sha256']==plan['source_sha256']['ours/native_training_trace.py'],'Wrong recorder')
    pool=RequestPool(directory,plan['model']);require(all(c==context for c in pool.contexts),'Wrong request contexts')
    checked=[await replay_row(row,pool=pool,config=config,tokenizer=tokenizer,harness=harness,by_id=by_id) for row in rows]
    require(Counter(r['puzzle_id'] for r in checked)==Counter({key:8 for key in expected}),'Wrong row multiplicities')
    require(not any(pool.calls.values()),'Unmatched actual requests')
    require(pool.accounting['recorded_starts']<=plan['max_policy_calls'] and
            pool.accounting['known_completion_tokens']<=plan['max_assistant_output_tokens'],'Exceeded frozen generation budget')
    artifacts=[*sorted(directory.glob('requests-*.jsonl')),directory/receipt['file'],directory/'batch-1.receipt.json',start_path]
    sources=('ours/audit_joint_training_phase1.py','ours/joint_training_phase1.py','ours/audit_alternation_training.py',
             'ours/audit_native_training_batch.py','ours/native_training_trace.py')
    return {'kind':'controlled_joint_phase1_batch_audit','status':'PASS','job_id':job,'condition':plan['condition'],
        'seed':plan['seed'],'phase':1,'plan_sha256':file_sha256(plan_path),'context':context,'trajectories':64,
        'puzzle_ids':expected,'accepted':sum(r['accepted'] for r in checked),
        'accepted_loss_tokens':sum(r['assistant_reencoded_tokens'] for r in checked if r['accepted']),
        'trajectory_checks':checked,'request_accounting':pool.accounting,
        'artifact_sha256':{str(p):file_sha256(p) for p in artifacts},'audit_source_sha256':{p:file_sha256(Path(p)) for p in sources},
        'new_model_calls':0,'limitations':['Complete recorded reply/token/mask replay and independent board checks.',
            'No optimizer completion, checkpoint-value change or heldout performance is inferred.',
            'Duplicate identical replies establish multiplicities, not unique request-task lineage.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('plan','directory','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing first-phase audit')
    report=asyncio.run(audit(args.plan,args.directory));write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','condition','seed','trajectories','accepted','accepted_loss_tokens')}),flush=True)
