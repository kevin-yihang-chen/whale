"""Keep the original MH loop resident while offline Slurm workers evaluate h.

This is the execution boundary for E3: h_next = MH(theta_fixed, h_incoming).
Scores enter the native frontier only after complete evidence and terminal job
verification. A controller interruption preserves its state and forbids an
implicit fresh restart. The initial plan writer supports the harness-only arm;
joint-arm checkpoint provenance must be implemented before launching that arm.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from unittest.mock import patch

from .audit_native_training_batch import require
from .controlled_conditions import native_search_context
from .experiment_randomization import TRIAL_SEEDS, TrialRandomization
from .glm_gateway import BudgetJournal, MODEL
from .local_completion import checkpoint_manifest
from .native_search import tree_hashes, write_json
from .preflight import PIN
from .scoped_proposer import isolated_native_proposal
from .visual_task import file_sha256

INITIALIZATION = Path('results/canonical-initialization-20260910.json')
SPLIT = Path('results/chess-pilot-manifest-20260909.json')
REFERENCE = Path('results/alternation-checkpoint-transition-222640.json')
BASELINE = Path('upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py')
SOURCES = ('ours/staged_search.py', 'ours/staged_evaluation.py', 'ours/audit_staged_evaluation.py',
    'ours/run_staged_evaluation.sh', 'ours/scoped_proposer.py', 'ours/controlled_conditions.py',
    'ours/experiment_randomization.py', 'ours/prompt_subspace.py', 'ours/mh_phase.py',
    'ours/native_search.py', 'ours/phase_vllm_worker.py', 'ours/probe_updated_vllm.py',
    'ours/vllm_runtime.py', 'ours/local_completion.py', 'ours/visual_task.py', 'ours/evidence.py',
    'ours/audit_mh_phase.py', 'ours/audit_native_training_batch.py', 'ours/glm_gateway.py',
    'ours/confined_exec.py', 'ours/preflight.py', 'ours/controlled_pilot_protocol.md',
    'ours/compat/autoharness_textarena/__init__.py', 'ours/compat/autoharness_textarena/config.py',
    'ours/compat/autoharness_textarena/llm.py', 'data/claude-runtime-2.1.236/claude')


def prepare(path, seed, root):
    from .probe_updated_vllm import changed_embedding_coordinates, inference_contract
    require(seed in TRIAL_SEEDS, 'Outside predeclared pilot seeds')
    require(not path.exists() and not root.exists(), 'Preserve existing plan or trial')
    init = json.loads(INITIALIZATION.read_text())
    reference = json.loads(REFERENCE.read_text())['exported']
    split = json.loads(SPLIT.read_text())['splits']['mh_val']
    model = init['exported']
    # Reference weights only locate discriminating coordinates. They are never
    # used to initialize this untrained harness-only condition.
    coordinates = changed_embedding_coordinates(Path(reference['path']), Path(model['path']))
    inference_contract(Path(init['base']['path']), Path(model['path']))
    sources = [*SOURCES, str(INITIALIZATION), str(REFERENCE), str(SPLIT), split['path']]
    plan = {'kind': 'controlled_staged_mh_plan', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'condition': 'harness_only', 'phase_root': str(root.resolve()), 'data_role': 'mh_val',
        'split_manifest': str(SPLIT), 'dataset': split['path'], 'examples': 32,
        'puzzle_ids': split['puzzle_ids'], 'shard_ids': [sorted(split['puzzle_ids'])[i::2] for i in (0,1)],
        'replicas': 2, 'seed': seed, 'iterations': 5, 'proposals_per_iter': 3, 'veto_mode': 'off',
        'initialization_report': str(INITIALIZATION), 'target_manifest': model,
        'incoming_harness': str(BASELINE.resolve()), 'incoming_harness_sha256': file_sha256(BASELINE),
        'worker_probe_reference': reference, 'worker_probe_coordinates': coordinates,
        'worker_probe_reference_role': 'Numerical discrimination only; not training initialization.',
        'proposer_model': MODEL, 'max_api_requests': 12, 'max_cli_turns': 12,
        'propose_timeout_seconds': 300, 'startup_timeout_seconds': 360,
        'target_config': {'model': model['path'], 'provider': 'local-vllm', 'max_tokens': 8129,
            'temperature': 1., 'top_p': 1., 'top_k': 20, 'batch_concurrency': 4,
            'chat_template_kwargs': {'enable_thinking': True}, 'seed': seed,
            'expected_weights_sha256': model['weights_sha256'], 'request_timeout_seconds': 420},
        'server_arguments': ['--dtype', 'bfloat16', '--max-model-len', '32768', '--max-num-seqs', '8',
            '--max-num-batched-tokens', '16384', '--gpu-memory-utilization', '0.60', '--enforce-eager',
            '--enable-prefix-caching', '--enable-chunked-prefill', '--gdn-prefill-backend', 'triton',
            '--distributed-executor-backend', 'uni', '--seed', str(seed),
            '--worker-cls', 'ours.phase_vllm_worker.CheckpointVerifiedWorker'],
        'versions': {name: version(name) for name in ('torch','vllm','transformers','chess','pandas','pyarrow')},
        'source_sha256': {name: file_sha256(Path(name)) for name in sources}, 'upstream_pin': PIN,
        'resources_per_evaluation': {'gpus': 2, 'type': 'h800', 'cpus': 16, 'ram_gib': 128,
            'time_limit_seconds': 2400, 'mail_user': 'kevin.yihang.chan@gmail.com', 'mail_type': 'ALL'},
        'bounds': {'evaluations': 16, 'trajectories': 512, 'assistant_tokens': 512 * 8129,
            'policy_calls': 512 * 18, 'proposer_api_requests': 60, 'gpu_hours': 16 * 2 * 2400 / 3600},
        'limitations': ['Optimization-set pilot, not heldout or visual evidence.',
            'GLM substitutes for the released proposer under the persistent global 45 CNY journal.',
            'Native perfect-score stopping, invalid candidates and all failed attempts remain visible.',
            'Each evaluation starts two fresh engines; cache and continuous engine state are not claimed identical.',
            'No automatic controller restart or resampling after interruption.']}
    write_json(path, plan)
    check_plan(path)
    return plan


def check_plan(path, *, verify_weights=True):
    from .probe_updated_vllm import changed_embedding_coordinates
    from .vllm_runtime import VLLMConfiguration
    from autoharness_chess_puzzle import runner
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'controlled_staged_mh_plan' and plan['condition'] == 'harness_only',
            'This writer currently certifies only common-initialization harness-only trials')
    require(plan['data_role'] == 'mh_val' and plan['veto_mode'] == 'off' and plan['examples'] == 32 and
            plan['seed'] in TRIAL_SEEDS and plan['replicas'] == 2 and
            plan['iterations'] == 5 and plan['proposals_per_iter'] == 3, 'Changed search scope')
    require(set(SOURCES) <= set(plan['source_sha256']), 'Missing frozen execution source')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Frozen source changed: {name}')
    require(plan['upstream_pin'] == PIN and
            subprocess.check_output(['git','-C','upstream/WHALE','rev-parse','HEAD'],text=True).strip() == PIN and
            not subprocess.check_output(['git','-C','upstream/WHALE','status','--porcelain'],text=True).strip(),
            'Changed upstream')
    require({name: version(name) for name in plan['versions']} == plan['versions'], 'Changed runtime')
    init = json.loads(Path(plan['initialization_report']).read_text())
    require(init['status'] == 'PASS' and init['optimizer_steps'] == init['new_model_calls'] == 0 and
            init['audit']['exact_source_bf16_cast'] and init['exported'] == plan['target_manifest'],
            'No certified common untrained initialization')
    split = json.loads(Path(plan['split_manifest']).read_text())['splits']
    ids = plan['puzzle_ids']
    require(ids == split['mh_val']['puzzle_ids'] and len(set(ids)) == 32 and
            not set(ids) & (set(split['train']['puzzle_ids']) | set(split['test']['puzzle_ids'])), 'Wrong data roles')
    require(plan['dataset'] == split['mh_val']['path'] and
            file_sha256(Path(plan['dataset'])) == split['mh_val']['sha256'], 'Changed MH data')
    require(plan['shard_ids'] == [sorted(ids)[i::2] for i in (0,1)], 'Changed shard assignment')
    require(file_sha256(Path(plan['incoming_harness'])) == plan['incoming_harness_sha256'] == file_sha256(BASELINE),
            'Harness-only trial must start at original h0')
    target = plan['target_config']
    VLLMConfiguration(**target, base_url='http://127.0.0.1:1')
    require(target == {'model': init['exported']['path'], 'provider': 'local-vllm', 'max_tokens': 8129,
        'temperature': 1., 'top_p': 1., 'top_k': 20, 'batch_concurrency': 4,
        'chat_template_kwargs': {'enable_thinking': True}, 'seed': plan['seed'],
        'expected_weights_sha256': init['exported']['weights_sha256'], 'request_timeout_seconds': 420},
        'Changed inference contract')
    require(plan['proposer_model'] == MODEL and plan['max_api_requests'] == plan['max_cli_turns'] == 12 and
            plan['propose_timeout_seconds'] == 300, 'Changed proposer budget')
    opts = plan['server_arguments']
    require(opts[opts.index('--seed')+1] == str(plan['seed']), 'Wrong engine seed')
    examples = runner._read_examples(plan['dataset'], limit=32, seed=plan['seed'])
    require({e.example_id for e in examples} == set(ids) and
            {runner._public_example_id(e) for e in examples} == {'redacted'}, 'Native reader or redaction differs')
    if verify_weights:
        require(checkpoint_manifest(Path(target['model'])) == plan['target_manifest'], 'Changed target bytes')
        reference = plan['worker_probe_reference']
        require(checkpoint_manifest(Path(reference['path'])) == reference, 'Changed probe reference')
        require(changed_embedding_coordinates(Path(reference['path']), Path(target['model'])) ==
                plan['worker_probe_coordinates'], 'Wrong discriminating coordinates')
    return plan


def native_arguments(plan, root):
    config = {'task_profiles': {'mh_val': {'dataset_path': plan['dataset'], 'limit': 32}},
        'models': [plan['target_config']], 'seeds': [plan['seed']],
        'eval': {'assistant_token_budget': 8129, 'policy_max_tokens': 8129}}
    write_json(root/'native-config.json', config)
    return argparse.Namespace(run_name='search', config=str(root/'native-config.json'), iterations=5,
        proposals_per_iter=3, proposer_model=MODEL, proposer_effort='low', propose_timeout=300,
        early_stop_success_rate=1., fresh=False, force=False, start_iteration=1,
        early_stop_min_iters=0, early_stop_patience=2, eval_only=False, use_api_key=True,
        prompt_only=plan['condition']=='whale_fst')


def terminal_job(raw, job_id):
    fields = dict(re.findall(r'(\w+)=([^\s]+)', raw))
    require(fields.get('JobId') == str(job_id), 'Wrong Slurm job identity')
    state = fields['JobState']
    if state in {'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED'}:
        return None
    require(state == 'COMPLETED' and fields.get('ExitCode') == '0:0', f'Evaluation job ended with {state}')
    allocated = dict(item.split('=',1) for item in fields['AllocTRES'].split(','))
    require(allocated.get('gres/gpu') == allocated.get('gres/gpu:h800') == '2', 'Wrong evaluation GPU allocation')
    duration = fields['RunTime']; days, _, clock = duration.rpartition('-')
    h,m,s = map(int,clock.split(':')); seconds = int(days or 0)*86400+h*3600+m*60+s
    return {'job_id': str(job_id), 'state': state, 'exit_code': fields['ExitCode'],
            'seconds': seconds, 'gpu_hours': 2*seconds/3600, 'alloc_tres': fields['AllocTRES']}


def verify_evaluation(private, plan, plan_path):
    request = json.loads((private/'request.json').read_text())
    result = json.loads((private/'result.json').read_text())
    require(result['status'] == 'PASS' and result['plan_sha256'] == request['plan_sha256'] == file_sha256(plan_path) and
            result['request_sha256'] == file_sha256(private/'request.json'), 'Evaluation receipt identity differs')
    require({'audit.json','merged/val.json','merged/trajectories.jsonl'} <= set(result['artifact_sha256']),
            'Incomplete evidence manifest')
    for name,digest in result['artifact_sha256'].items():
        path = private/name
        require(path.resolve().is_relative_to(private.resolve()) and file_sha256(path) == digest,
                f'Changed evaluation evidence: {name}')
    audit = json.loads((private/'audit.json').read_text())
    require(audit['status'] == 'PASS' and audit['examples'] == 32 and audit['seed'] == plan['seed'] and
            audit['plan_sha256'] == file_sha256(plan_path) and
            audit['audit_source_sha256'] == file_sha256(Path('ours/audit_staged_evaluation.py')) and
            audit['harness_sha256'] == request['harness_sha256'] == file_sha256(Path(request['kwargs']['harness_path'])),
            'Audit or harness identity differs')
    terminal = terminal_job((private/'slurm-terminal.txt').read_text(), result['job_id'])
    require(terminal is not None, 'Evaluation allocation is still active')
    summary = json.loads((private/'merged/val.json').read_text())
    require(summary['num_examples'] == 32 and summary['solved_examples'] == audit['solved'] == result['solved'] and
            audit['calls'] == result['new_model_calls'] and audit['generated_tokens'] == result['generated_tokens'],
            'Audited scores or usage differ')
    return summary, terminal


class StagedSearch:
    def __init__(self, plan, plan_path, *, submit=False):
        self.plan, self.plan_path = plan, plan_path.resolve()
        self.root, self.submit = Path(plan['phase_root']).resolve(), submit
        self.evaluations = []

    def state(self, status, **fields):
        path = self.root/'state.json'; temporary = path.with_suffix('.tmp')
        write_json(temporary, {'status': status, 'controller_pid': os.getpid(),
            'time_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': file_sha256(self.plan_path), **fields})
        temporary.replace(path)

    def wait_for_submission_slot(self, name, private):
        # Account currently permits one submitted job. Waiting happens before
        # sbatch; an ambiguous real submission is never automatically repeated.
        while True:
            jobs = subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split()
            if not jobs:
                return
            self.state('WAITING_SUBMISSION_SLOT',harness=name,evaluation=str(private),existing_jobs=jobs)
            time.sleep(30)

    def evaluate(self, **kwargs):
        name = Path(kwargs['harness_path']).parent.name
        require(re.fullmatch(r'h(?:[0-9]|1[0-5])',name) is not None, 'Outside frozen candidate allocation')
        require(len(self.evaluations) < 16, 'Evaluation budget exhausted')
        private = self.root/'evaluations'/name; private.mkdir(parents=True,exist_ok=False)
        serial = {k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()}
        write_json(private/'request.json', {'plan_sha256': file_sha256(self.plan_path),
            'harness_sha256': file_sha256(Path(kwargs['harness_path'])), 'kwargs': serial})
        self.state('WAITING_GPU_EVALUATION', harness=name, evaluation=str(private))
        if self.submit:
            self.wait_for_submission_slot(name,private)
            raw = subprocess.check_output(['sbatch','--parsable','ours/run_staged_evaluation.sh',
                str(self.plan_path),str(private)],text=True)
            job_id = raw.strip()
            require(job_id.isdigit(), 'Ambiguous sbatch response; inspect allocation before any retry')
            write_json(private/'submission.json',{'job_id':job_id,'sbatch_response':raw,
                'request_sha256':file_sha256(private/'request.json')})
        while True:
            start = private/'submission.json'
            if not start.exists(): start = private/'start.json'
            if start.exists():
                job_id = json.loads(start.read_text())['job_id']
                self.state('WAITING_GPU_EVALUATION', harness=name, evaluation=str(private), job_id=job_id)
                raw = subprocess.check_output(['scontrol','show','job',str(job_id),'-o'],text=True)
                try:
                    terminal = terminal_job(raw, job_id)
                except ValueError:
                    (private/'slurm-terminal.txt').write_text(raw)
                    raise
                if terminal is not None:
                    (private/'slurm-terminal.txt').write_text(raw)
                    break
            time.sleep(30)
        summary, terminal = verify_evaluation(private,self.plan,self.plan_path)
        output = Path(kwargs['output_dir'])
        # Native run_sweep creates the destination before invoking this hook.
        output.mkdir(parents=True,exist_ok=True)
        require(not any(output.iterdir()), 'Refuse to overwrite native evaluation')
        shutil.copytree(private/'merged',output,dirs_exist_ok=True)
        self.evaluations.append({'harness':name,'directory':str(private),'public_directory':str(output),
            'public_sha256':tree_hashes(output),'solved':summary['solved_examples'],**terminal})
        write_json(self.root/'evaluations.json',self.evaluations)
        return summary

    def verify_public(self):
        require(bool(self.evaluations), 'No audited MH evidence')
        for value in self.evaluations:
            verify_evaluation(Path(value['directory']),self.plan,self.plan_path)
            require(tree_hashes(Path(value['public_directory'])) == value['public_sha256'],
                    'Public search scores or traces changed after audit')

    def run(self):
        self.root.mkdir(parents=True,exist_ok=False)
        (self.root/'plan.json').write_bytes(self.plan_path.read_bytes())
        journal = BudgetJournal(Path('data/glm-budget/ledger.jsonl'))
        self.state('STARTING',budget=journal.summary())
        args = native_arguments(self.plan,self.root)
        with native_search_context(self.plan['condition'],self.root,self.plan['incoming_harness']) as (native,benchmark,provenance):
            original_sweep = native.run_sweep
            def sweep(config,harnesses,logs_dir,**kwargs):
                rows = original_sweep(config,harnesses,logs_dir,**kwargs)
                require(len(rows)==len(harnesses) and all(ok for _,ok in rows),'Native evaluation failed; inspect preserved evidence')
                for name,_ in harnesses:
                    require(name in {value['harness'] for value in self.evaluations},'Unaudited or cached native score')
                self.verify_public()
                return rows
            def propose(**kwargs):
                self.verify_public()
                number = kwargs['iteration']
                request = self.root/f'proposal-request-{number}.json'
                require(not request.exists(),'Refuse repeated paid proposal')
                write_json(request,{k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()})
                self.state('PROPOSING',iteration=number,slots=kwargs['next_names'],budget=journal.summary())
                return isolated_native_proposal(native,self.plan,self.root/f'proposal-{number}',journal,**kwargs)
            try:
                with patch.object(benchmark,'evaluate_harness',self.evaluate),patch.object(native,'run_sweep',sweep), \
                     patch.object(native,'propose_claude_with_retries',propose):
                    native.run_evolve(args)
                self.verify_public()
                accepted = native.get_accepted_harness(self.root/'search')
                write_json(self.root/'result.json', {'status':'COMPLETED_NATIVE_SEARCH','plan_sha256':file_sha256(self.plan_path),
                    'accepted':accepted,'accepted_harness_sha256':file_sha256(self.root/f'search/harnesses/{accepted}/harness.py'),
                    'search_sha256':tree_hashes(self.root/'search'),'evaluations':self.evaluations,
                    'budget':journal.summary(),'provenance':provenance,'limitations':self.plan['limitations']})
                self.state('COMPLETED',accepted=accepted)
            except BaseException as error:
                self.state('FAILED',error_type=type(error).__name__,error=str(error),budget=journal.summary())
                raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','check','coordinate','evaluate','worker'),required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--replica',type=int,choices=(0,1))
    parser.add_argument('--submit-evaluations',action='store_true')
    args=parser.parse_args()
    if args.phase=='prepare':
        require(args.root is not None,'A new trial root is required')
        prepare(args.plan,args.seed,args.root)
        print(json.dumps({'status':'PREPARED','plan_sha256':file_sha256(args.plan)}),flush=True)
        return
    plan=check_plan(args.plan,verify_weights=args.phase!='worker')
    if args.phase=='check':
        print(json.dumps({'status':'PASS','plan_sha256':file_sha256(args.plan)}),flush=True)
    elif args.phase=='coordinate':
        require('SLURM_JOB_ID' not in os.environ,'Proposer coordinator belongs on the networked login node')
        StagedSearch(plan,args.plan,submit=args.submit_evaluations).run()
    else:
        from .staged_evaluation import run_evaluation,run_worker
        require(args.evaluation is not None,'Missing evaluation request directory')
        require(args.phase!='worker' or args.replica is not None,'Missing shard index')
        os.environ.update(TrialRandomization(plan['seed']).environment())
        (run_worker if args.phase=='worker' else run_evaluation)(args,plan)


if __name__=='__main__':
    main()
