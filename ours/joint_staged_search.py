"""Certified first joint E3 phase using the actual preceding E4 checkpoint.

The original five-round search and complete two-shard evaluator are inherited.
A joint-specific handoff binds completed training/export evidence and native
resume state; no engineering or common-theta0 score substitutes for theta1.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from . import staged_evaluation, staged_search
from .audit_native_training_batch import require
from .experiment_randomization import TRIAL_SEEDS, TrialRandomization
from .glm_gateway import MODEL
from .joint_checkpoint_export import close as close_export
from .local_completion import checkpoint_manifest
from .native_search import write_json
from .probe_updated_vllm import changed_embedding_coordinates, inference_contract
from .staged_candidate_recovery import compatible_proposal
from .visual_task import file_sha256

SOURCES=('ours/joint_staged_search.py','ours/run_joint_staged_evaluation.sh',
    'ours/joint_checkpoint_export.py','ours/joint_training_result.py',
    'ours/staged_metadata_recovery.py','ours/staged_search_recovery.py','ours/staged_candidate_recovery.py',
    'ours/prompts/chess_prompt_subspace.md')
WRAPPER='ours/run_joint_staged_evaluation.sh'
ENTRYPOINT='ours.joint_staged_search'


def certify_handoff(export_plan_path,export_result_path):
    result=json.loads(export_result_path.read_text())
    require(result['kind']=='controlled_joint_phase1_export_result' and result['status']=='COMPLETED_CANONICAL_EXPORT',
            'Joint search requires an independently completed joint export')
    verified=close_export(export_plan_path,Path(result['slurm_terminal_path']))
    require(result==verified,'Joint export result differs from its completed evidence')
    export=json.loads(export_plan_path.read_text())
    training_path=Path(export['training_plan']);training=json.loads(training_path.read_text())
    completed=json.loads(Path(export['training_result']).read_text())
    require(training['kind']=='controlled_joint_phase1_plan' and completed['kind']=='controlled_joint_phase1_training_result' and
            completed['status']=='COMPLETED_NATIVE_PHASE' and completed['original_preflight_batch_ids_match'] and
            completed['phase']==1 and completed['plan_sha256']==file_sha256(training_path) and
            completed['job_id']==export['source_job_id']==result['source_job_id'] and
            completed['condition']==training['condition']==export['condition']==result['condition'] and
            completed['seed']==training['seed']==export['seed']==result['seed'] and
            completed['checkpoints']==[result['resume_checkpoint']]==[export['resume_checkpoint']],
            'Joint phase lineage differs')
    require(result['exported']['path']==export['target'] and export['base']==training['initial_manifest'],
            'Wrong common initialization or exported target')
    sources={**export['source_sha256'],**result['artifact_sha256']}
    for p in (export_plan_path,export_result_path):sources[str(p)]=file_sha256(p)
    handoff={'export_plan':str(export_plan_path),'export_result':str(export_result_path),
        'export_plan_sha256':file_sha256(export_plan_path),'export_result_sha256':file_sha256(export_result_path),
        'source_job_id':result['source_job_id'],'export_job_id':result['job_id'],
        'condition':result['condition'],'seed':result['seed'],'initial_manifest':training['initial_manifest'],
        'exported':result['exported'],'resume_checkpoint':result['resume_checkpoint'],
        'optimizer_steps':completed['optimizer_steps'],'new_model_calls_by_certifier':0}
    return handoff,sources


def prepare(path,root,export_plan_path,export_result_path):
    require(not path.exists() and not root.exists(),'Preserve an existing joint search plan or trial')
    handoff,sources=certify_handoff(export_plan_path,export_result_path)
    require(handoff['condition'] in ('whale','whale_fst') and handoff['seed'] in TRIAL_SEEDS,'Outside declared joint controls')
    # The original writer supplies common data/decoding/budget settings only.
    # Its untrained plan is discarded; the joint plan is independently checked.
    with tempfile.TemporaryDirectory(prefix='whale-joint-search-settings-') as tmp:
        plan=staged_search.prepare(Path(tmp)/'common-settings.json',handoff['seed'],root)
    target=handoff['exported'];reference=plan['worker_probe_reference']
    inference_contract(Path(handoff['initial_manifest']['path']),Path(target['path']))
    plan.update(kind='controlled_joint_staged_mh_plan',condition=handoff['condition'],joint_phase=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),target_manifest=target,
        joint_handoff=handoff,joint_source_sha256=sources,
        evaluation_entrypoint=ENTRYPOINT,evaluation_wrapper=WRAPPER,
        worker_probe_coordinates=changed_embedding_coordinates(Path(reference['path']),Path(target['path'])),
        metadata_contract='bare_list_or_candidates_and_agreeing_name_slot_harness_id_candidate_v2',
        limitations=['Independent joint theta1 from completed first-phase training and canonical export.',
            'Original five-round/three-proposal MH32 optimization; no heldout or visual result is inferred.',
            'The numerical worker reference locates distinguishing coordinates and is never an initialization.',
            'Fresh two-engine evaluation per candidate; no persistent engine/RNG/cache equivalence is claimed.',
            'GLM uses the same globally bounded 45 CNY journal and authorized MH32 evidence.',
            'Proposer metadata aliases/current report padding are normalized before native import; candidate bytes stay unchanged.',
            'WHALE-FST uses the disclosed reconstructed prompt contract and native AST restriction.',
            'No automatic controller restart; all failed and invalid attempts remain recorded.'])
    plan['target_config'].update(model=target['path'],expected_weights_sha256=target['weights_sha256'])
    plan['source_sha256'].update({n:file_sha256(Path(n)) for n in SOURCES})
    write_json(path,plan);check_plan(path);return plan


def check_plan(path,*,verify_weights=True):
    from autoharness_chess_puzzle import runner
    from .vllm_runtime import VLLMConfiguration
    plan=json.loads(path.read_text());handoff=plan['joint_handoff']
    require(plan['kind']=='controlled_joint_staged_mh_plan' and plan['joint_phase']==1 and
            plan['condition'] in ('whale','whale_fst') and plan['seed'] in TRIAL_SEEDS,'Wrong joint search scope')
    require(plan['condition']==handoff['condition'] and plan['seed']==handoff['seed'] and
            plan['target_manifest']==handoff['exported'],'Mixed joint condition, seed or model')
    require(plan['evaluation_entrypoint']==ENTRYPOINT and plan['evaluation_wrapper']==WRAPPER and
            plan['metadata_contract']=='bare_list_or_candidates_and_agreeing_name_slot_harness_id_candidate_v2',
            'Changed joint execution boundary')
    require(set((*staged_search.SOURCES,*SOURCES))<=set(plan['source_sha256']),'Missing shared/joint execution source')
    for name,digest in plan['source_sha256'].items():require(file_sha256(Path(name))==digest,f'Frozen joint source changed: {name}')
    require(plan['upstream_pin']==staged_search.PIN and
            subprocess.check_output(['git','-C','upstream/WHALE','rev-parse','HEAD'],text=True).strip()==staged_search.PIN and
            not subprocess.check_output(['git','-C','upstream/WHALE','status','--porcelain'],text=True).strip(),'Changed upstream')
    require({n:version(n) for n in plan['versions']}==plan['versions'],'Changed search runtime')
    require(plan['data_role']=='mh_val' and plan['veto_mode']=='off' and plan['examples']==32 and
            plan['replicas']==2 and plan['iterations']==5 and plan['proposals_per_iter']==3,'Changed pilot search scope')
    splits=json.loads(Path(plan['split_manifest']).read_text())['splits'];ids=plan['puzzle_ids']
    require(ids==splits['mh_val']['puzzle_ids'] and len(set(ids))==32 and
            not set(ids)&(set(splits['train']['puzzle_ids'])|set(splits['test']['puzzle_ids'])),'Mixed MH data roles')
    require(plan['dataset']==splits['mh_val']['path'] and file_sha256(Path(plan['dataset']))==splits['mh_val']['sha256'] and
            plan['shard_ids']==[sorted(ids)[i::2] for i in (0,1)],'Changed dataset or shard assignment')
    require(Path(plan['incoming_harness']).resolve()==staged_search.BASELINE.resolve() and
            file_sha256(Path(plan['incoming_harness']))==plan['incoming_harness_sha256']==file_sha256(staged_search.BASELINE),
            'First joint search must start at original h0')
    target=handoff['exported'];seed=plan['seed']
    expected={'model':target['path'],'provider':'local-vllm','max_tokens':8129,'temperature':1.,'top_p':1.,'top_k':20,
        'batch_concurrency':4,'chat_template_kwargs':{'enable_thinking':True},'seed':seed,
        'expected_weights_sha256':target['weights_sha256'],'request_timeout_seconds':420}
    require(plan['target_config']==expected,'Changed joint inference contract')
    VLLMConfiguration(**expected,base_url='http://127.0.0.1:1')
    require(plan['server_arguments']==['--dtype','bfloat16','--max-model-len','32768','--max-num-seqs','8',
        '--max-num-batched-tokens','16384','--gpu-memory-utilization','0.60','--enforce-eager',
        '--enable-prefix-caching','--enable-chunked-prefill','--gdn-prefill-backend','triton',
        '--distributed-executor-backend','uni','--seed',str(seed),'--worker-cls','ours.phase_vllm_worker.CheckpointVerifiedWorker'],
        'Changed actual engine configuration')
    require(plan['proposer_model']==MODEL and plan['max_api_requests']==plan['max_cli_turns']==12 and
            plan['propose_timeout_seconds']==300 and plan['startup_timeout_seconds']==360,'Changed proposer/startup limits')
    require(plan['bounds']=={'evaluations':16,'trajectories':512,'assistant_tokens':512*8129,'policy_calls':512*18,
            'proposer_api_requests':60,'gpu_hours':16*2*2400/3600},'Changed phase budget')
    require(plan['resources_per_evaluation']=={'gpus':2,'type':'h800','cpus':16,'ram_gib':128,'time_limit_seconds':2400,
            'mail_user':'kevin.yihang.chan@gmail.com','mail_type':'ALL'},'Changed evaluation resources')
    examples=runner._read_examples(plan['dataset'],limit=32,seed=seed)
    require({e.example_id for e in examples}==set(ids) and
            {runner._public_example_id(e) for e in examples}=={'redacted'},'Wrong native task coverage or redaction')
    export_path,result_path=Path(handoff['export_plan']),Path(handoff['export_result'])
    require(file_sha256(export_path)==handoff['export_plan_sha256'] and
            file_sha256(result_path)==handoff['export_result_sha256'],'Changed completed handoff records')
    export=json.loads(export_path.read_text());result=json.loads(result_path.read_text())
    expected_sources={**export['source_sha256'],**result['artifact_sha256'],str(export_path):file_sha256(export_path),
                      str(result_path):file_sha256(result_path)}
    require(expected_sources==plan['joint_source_sha256'],'Incomplete joint provenance binding')
    require(result['kind']=='controlled_joint_phase1_export_result' and result['status']=='COMPLETED_CANONICAL_EXPORT' and
            result['condition']==plan['condition'] and result['seed']==seed and result['exported']==target and
            result['resume_checkpoint']==handoff['resume_checkpoint'] and result['source_job_id']==handoff['source_job_id'] and
            result['job_id']==handoff['export_job_id'] and export['base']==handoff['initial_manifest'],
            'Different completed joint handoff')
    # Parent/controller validates all weights; workers repeat all light provenance
    # checks and prove eight distinguishing values inside the actual loaded model.
    native_model=(Path(handoff['resume_checkpoint']['directory'])/'actor/model_world_size_1_rank_0.pt').resolve()
    for name,digest in expected_sources.items():
        if not verify_weights and Path(name).resolve()==native_model:continue
        require(file_sha256(Path(name))==digest,f'Changed completed joint evidence: {name}')
    reference=plan['worker_probe_reference']
    require(reference==json.loads(staged_search.REFERENCE.read_text())['exported'],'Wrong numerical reference')
    if verify_weights:
        verified,sources=certify_handoff(export_path,result_path)
        require(verified==handoff and sources==expected_sources,'Joint training/export lineage changed')
        require(checkpoint_manifest(Path(target['path']))==target and
                checkpoint_manifest(Path(reference['path']))==reference,'Changed served target or numerical reference')
        require(changed_embedding_coordinates(Path(reference['path']),Path(target['path']))==plan['worker_probe_coordinates'],
                'Changed worker discrimination coordinates')
        inference_contract(Path(handoff['initial_manifest']['path']),Path(target['path']))
    return plan


class JointStagedSearch(staged_search.StagedSearch):
    def evaluate(self,**kwargs):
        original=staged_search.subprocess.check_output
        def command(argv,*args,**options):
            if list(argv[:3])==['sbatch','--parsable','ours/run_staged_evaluation.sh']:
                argv=[*argv[:2],WRAPPER,*argv[3:]]
            return original(argv,*args,**options)
        with patch.object(staged_search.subprocess,'check_output',command):
            return super().evaluate(**kwargs)

    def run(self):
        with patch.object(staged_search,'isolated_native_proposal',compatible_proposal):
            return super().run()


def run_evaluation(args,plan):
    original=staged_evaluation.subprocess.Popen
    def launch(argv,*positional,**options):
        if len(argv)>=3 and argv[1:3]==['-m','ours.staged_search']:
            argv=[argv[0],'-m',ENTRYPOINT,*argv[3:]]
        return original(argv,*positional,**options)
    with patch.object(staged_evaluation.subprocess,'Popen',launch):
        return staged_evaluation.run_evaluation(args,plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','check','coordinate','evaluate','worker'),required=True)
    for name in ('plan','root','export-plan','export-result','evaluation'):parser.add_argument('--'+name,type=Path,required=name=='plan')
    parser.add_argument('--replica',type=int,choices=(0,1));parser.add_argument('--submit-evaluations',action='store_true')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(all(getattr(args,k) is not None for k in ('root','export_plan','export_result')),'Missing completed joint handoff')
        prepare(args.plan,args.root,args.export_plan,args.export_result)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True);return
    plan=check_plan(args.plan,verify_weights=args.phase!='worker')
    if args.phase=='check':print(json.dumps({'status':'PASS','plan_sha256':file_sha256(args.plan)}),flush=True)
    elif args.phase=='coordinate':
        require('SLURM_JOB_ID' not in os.environ,'Coordinator belongs on the networked login node')
        JointStagedSearch(plan,args.plan,submit=args.submit_evaluations).run()
    else:
        require(args.evaluation is not None and (args.phase!='worker' or args.replica is not None),'Missing worker request')
        os.environ.update(TrialRandomization(plan['seed']).environment())
        (staged_evaluation.run_worker if args.phase=='worker' else run_evaluation)(args,plan)


if __name__=='__main__':main()
