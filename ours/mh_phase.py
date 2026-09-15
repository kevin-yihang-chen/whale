"""Stage the original MH loop across offline GPU evaluation and a login proposer.

E4 -> E3 execution: the trained checkpoint evaluates all32 fixed MH examples.
Two deterministic shards use the original runner; summaries are recomputed from
all native outputs. Only the native redacted search archive enters the proposer.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import uuid

from .audit_native_training_batch import require
from .glm_gateway import BudgetJournal, MODEL
from .local_completion import checkpoint_manifest
from .native_search import isolated_proposal, normalize_metadata, target_service, tree_hashes, write_json
from .preflight import PIN
from .visual_task import file_sha256
from .vllm_runtime import RecordedVLLMClient, VLLMConfiguration


class ProposalBoundary(Exception):
    """The GPU baseline has completed; its proposer must run on the login node."""


def check_plan(path, *, verify_weights=True):
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'updated_checkpoint_mh_phase' and plan['data_role'] == 'mh_val', 'Wrong phase role')
    require(plan['examples'] == 32 and plan['seed'] == 42 and plan['replicas'] == 2, 'Wrong pilot scope')
    require(plan['veto_mode'] == 'off' and plan['proposer_model'] == MODEL, 'Wrong search condition')
    require(plan['iterations'] == plan['proposals_per_iter'] == 1, 'Expected one native iteration')
    require(1 <= plan['max_api_requests'] <= 12 and 1 <= plan['max_cli_turns'] <= 12, 'Unbounded proposer')
    require(plan['propose_timeout_seconds'] == 300, 'Unexpected proposer deadline')
    require(subprocess.check_output(['git', '-C', 'upstream/WHALE', 'rev-parse', 'HEAD'], text=True).strip() == PIN,
            'Wrong upstream revision')
    require(not subprocess.check_output(['git', '-C', 'upstream/WHALE', 'status', '--porcelain'], text=True).strip(),
            'Modified upstream')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Frozen phase source changed: {name}')
    require({name: version(name) for name in plan['versions']} == plan['versions'], 'Changed runtime')
    split = json.loads(Path(plan['split_manifest']).read_text())['splits']
    require(plan['dataset'] == split['mh_val']['path'] and split['mh_val']['examples'] == 32, 'Not the declared MH data')
    ids = split['mh_val']['puzzle_ids']
    require(len(set(ids)) == 32 and ids == plan['puzzle_ids'], 'Changed MH IDs')
    require(not set(ids) & (set(split['train']['puzzle_ids']) | set(split['test']['puzzle_ids'])), 'Data roles overlap')
    require(file_sha256(Path(plan['dataset'])) == split['mh_val']['sha256'], 'Changed MH data bytes')
    require(plan['shard_ids'] == [sorted(ids)[::2], sorted(ids)[1::2]], 'Changed deterministic assignment')
    handoff = json.loads(Path(plan['handoff']).read_text())
    require(handoff['status'] == 'PASS_ENGINEERING_HANDOFF_ONLY' and handoff['fresh_updated_checkpoint_inference'],
            'Updated inference not verified')
    observed = json.loads(Path(plan['worker_probe_source']).read_text())
    require(plan['worker_probe_coordinates'] == observed['changed_coordinates'] and
            len(plan['worker_probe_coordinates']) == 8, 'Worker probe differs from the audited handoff')
    transition = json.loads(Path(plan['transition']).read_text())
    require(transition['native_fp32']['status'] == 'CHANGED' and transition['export_exact_native_bf16_cast'],
            'No verified native weight update')
    target = plan['target_config']
    require(target['model'] == transition['exported']['path'] and
            target['expected_weights_sha256'] == transition['exported']['weights_sha256'], 'Wrong updated checkpoint')
    VLLMConfiguration(**target, base_url='http://127.0.0.1:1')
    require(target['max_tokens'] == 8129 and target['seed'] == 42 and target['batch_concurrency'] == 4,
            'Changed generation scope')
    require(target['chat_template_kwargs'] == {'enable_thinking': True}, 'Changed thinking mode')
    if verify_weights:
        require(checkpoint_manifest(Path(target['model'])) == transition['exported'], 'Checkpoint bytes changed')
    from autoharness_chess_puzzle import runner
    examples = runner._read_examples(plan['dataset'], limit=32, seed=42)
    require({e.example_id for e in examples} == set(ids), 'Original dataset reader coverage differs')
    require({runner._public_example_id(e) for e in examples} == {'redacted'}, 'Changed native ID redaction')
    return plan


def verify_receipt(root, name, plan_path, *, verify_live=True):
    receipt = json.loads((root / name).read_text())
    require(receipt['plan_sha256'] == file_sha256(plan_path), 'Receipt belongs to another plan')
    for relative, digest in receipt['artifact_sha256'].items():
        p = root / relative
        require(p.resolve().is_relative_to(root.resolve()) and file_sha256(p) == digest, f'Changed phase artifact: {relative}')
    if verify_live:
        require(tree_hashes(root / 'search') == receipt['live_search_sha256'], 'Live search differs from sealed stage')
    return receipt


def seal(root, name, status, plan_path, **fields):
    hashes = tree_hashes(root / 'search')
    snapshot = root / 'snapshots' / Path(name).stem
    shutil.copytree(root / 'search', snapshot)
    write_json(root / name, {'status': status, 'plan_sha256': file_sha256(plan_path),
               'time_utc': datetime.now(timezone.utc).isoformat(),
               'live_search_sha256': hashes,
               'artifact_sha256': {str(snapshot.relative_to(root) / k): v for k, v in hashes.items()}, **fields})


def merge_outputs(plan, shards):
    """Use private ordered IDs while retaining the native redacted output IDs."""
    from autoharness_chess_puzzle import runner
    expected = [e.example_id for e in runner._read_examples(plan['dataset'], limit=32, seed=42)]
    pairs = [(key, row) for shard in shards for key, row in zip(shard['ids'], shard['outputs'], strict=True)]
    by_id = dict(pairs)
    require(len(pairs) == len(by_id) == len(expected) and set(by_id) == set(expected), 'Duplicate or missing shard output')
    require(all(row['example_id'] == 'redacted' for row in by_id.values()), 'Native output ID was exposed')
    return [by_id[key] for key in expected]


def aggregate_evaluation(plan, private, kwargs):
    from autoharness_chess_puzzle import runner
    request = json.loads((private / 'request.json').read_text())
    shards, metadata = [], []
    for replica in range(2):
        directory = private / f'replica-{replica}'
        receipt = json.loads((directory / 'result.json').read_text())
        require(receipt['status'] == 'COMPLETED' and receipt['replica'] == replica, 'Incomplete replica')
        require(receipt['plan_sha256'] == request['plan_sha256'] and
                receipt['harness_sha256'] == request['harness_sha256'], 'Mixed evaluation identities')
        for name, digest in receipt['artifact_sha256'].items():
            require(file_sha256(directory / name) == digest, f'Changed replica artifact: {name}')
        require(receipt['assigned_ids'] == plan['shard_ids'][replica], 'Wrong shard assignment')
        shards.append({'outputs': json.loads((directory / 'native-outputs.json').read_text()),
                       'ids': json.loads((directory / 'ordered-ids.json').read_text())})
        metadata.append(json.loads((directory / 'evaluation/val.json').read_text())['metadata'])
    for key in metadata[0]:
        require(metadata[0][key] == metadata[1][key], f'Shard metadata differ: {key}')
    outputs = merge_outputs(plan, shards)
    meta = dict(metadata[0], limit=32, dataset_path=plan['dataset'], harness_path=str(kwargs['harness_path']))
    summary = runner.summarize_outputs(outputs, meta)
    output = Path(kwargs['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'val.json', summary)
    runner.write_trajectories(outputs, output)
    with (output / 'ChessPuzzle-v0_policy_trace.jsonl').open('x') as stream:
        for row in outputs:
            for event in row['harness_trace']:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    write_json(private / 'merge.json', {'status': 'COMPLETE_COVERAGE', 'examples': 32,
               'shard_assignments': plan['shard_ids'], 'summary': summary,
               'ordered_private_ids': [e.example_id for e in runner._read_examples(plan['dataset'], limit=32, seed=42)]})
    return summary


def distributed_evaluation(plan, plan_path, root, **kwargs):
    require(kwargs['dataset_path'] == plan['dataset'] and kwargs['limit'] == 32 and kwargs['seed'] == 42,
            'Native sweep context changed')
    name = Path(kwargs['harness_path']).parent.name
    require(name in {'h0', 'h1'}, 'Unexpected harness slot')
    private = root / 'evaluations' / name
    private.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
    request.update(plan_sha256=file_sha256(plan_path), harness_sha256=file_sha256(Path(kwargs['harness_path'])))
    write_json(private / 'request.json', request)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(devices) == 2 and len(set(devices)) == 2 and all(devices), 'Expected exactly two allocated GPUs')
    children, logs = [], []
    try:
        for replica, device in enumerate(devices):
            log = (private / f'replica-{replica}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            children.append(subprocess.Popen([sys.executable, '-m', 'ours.mh_phase', '--plan', str(plan_path),
                '--phase', 'worker', '--replica', str(replica), '--evaluation', str(private)],
                env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            require(not any(p.poll() not in (None, 0) for p in children), 'A shard failed; stop its peer and preserve all evidence')
            time.sleep(2)
        require(all(p.returncode == 0 for p in children), 'A shard failed')
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    return aggregate_evaluation(plan, private, kwargs)


def run_worker(args, plan):
    import torch
    from autoharness_chess_puzzle import runner
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Worker must see one GPU')
    request = json.loads((args.evaluation / 'request.json').read_text())
    require(request['plan_sha256'] == file_sha256(args.plan), 'Wrong evaluation plan')
    require(file_sha256(Path(request['harness_path'])) == request['harness_sha256'], 'Harness changed')
    output = args.evaluation / f'replica-{args.replica}'
    output.mkdir(exist_ok=False)
    clients, captured, ordered_ids = [], [], []
    journal_lock = threading.Lock()
    def record(event):
        with journal_lock, (output / 'request-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    class Client(RecordedVLLMClient):
        def __init__(self, config):
            super().__init__(config)
            clients.append(self)
        def complete_response(self, messages, *, max_tokens=None):
            ident = uuid.uuid4().hex
            record({'event': 'start', 'id': ident, 'messages': [asdict(m) for m in messages], 'max_tokens': max_tokens})
            try:
                value = super().complete_response(messages, max_tokens=max_tokens)
            except BaseException as exc:
                record({'event': 'error', 'id': ident, 'type': type(exc).__name__, 'usage': None})
                raise
            record({'event': 'complete', 'id': ident, 'usage': value.usage})
            return value
    original_read, original_summary = runner._read_examples, runner.summarize_outputs
    def shard_read(path, *, limit, seed):
        require(str(path) == plan['dataset'] and limit == 32 and seed == 42, 'Unexpected dataset request')
        values = [e for e in original_read(path, limit=limit, seed=seed) if e.example_id in plan['shard_ids'][args.replica]]
        ordered_ids.extend(e.example_id for e in values)
        return values
    def capture(rows, metadata):
        captured.extend(rows)
        return original_summary(rows, metadata)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'MH worker received {signum}')
    signal.signal(signal.SIGTERM, terminate)
    os.environ.update(WHALE_MH_PLAN=str(args.plan.resolve()), WHALE_MH_WORKER_RECEIPT=str((output / 'worker-weights.json').resolve()))
    started = time.monotonic()
    try:
        with target_service(plan, output) as endpoint:
            proof = json.loads((output / 'worker-weights.json').read_text())
            require(proof['status'] == 'PASS' and proof['plan_sha256'] == file_sha256(args.plan), 'Worker proof missing')
            kwargs = {k: v for k, v in request.items() if k not in {'plan_sha256', 'harness_sha256'}}
            kwargs['output_dir'] = output / 'evaluation'
            kwargs['llm_config'] = dict(plan['target_config'], base_url=endpoint, trace_path=str(output / 'generations.jsonl'))
            with patch.object(runner, 'LLMClient', Client), patch.object(runner, 'LLMConfig', VLLMConfiguration), \
                 patch.object(runner, '_read_examples', shard_read), patch.object(runner, 'summarize_outputs', capture):
                summary = runner.evaluate_harness(**kwargs)
            require(summary['num_examples'] == len(captured) == 16, 'Incomplete native shard')
            write_json(output / 'native-outputs.json', captured)
            write_json(output / 'ordered-ids.json', ordered_ids)
    finally:
        for client in clients:
            client.close()
    write_json(output / 'result.json', {'status': 'COMPLETED', 'replica': args.replica,
               'plan_sha256': file_sha256(args.plan), 'harness_sha256': request['harness_sha256'],
               'assigned_ids': plan['shard_ids'][args.replica], 'seconds': time.monotonic() - started,
               'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'artifact_sha256': tree_hashes(output)})


@contextmanager
def native_context(plan, root):
    from meta_harness import chess_puzzle_benchmark as benchmark
    from meta_harness import meta_harness_chess_puzzle as native
    original_results = benchmark.load_results
    def ordered_results(path):
        # Same scores/frontier; deterministic h0-before-h1 ingestion on objective ties.
        return dict(sorted(original_results(path).items()))
    env = {'CHESS_PUZZLE_DEFAULT_MAX_TURNS': '9', 'CHESS_PUZZLE_MAX_TURNS_CAP': '18',
           'CHESS_PUZZLE_FORMAT_RETRIES': '1', 'CHESS_PUZZLE_ILLEGAL_RETRIES': '1',
           'BASELINE_HARNESS_OVERRIDE': '', 'CHESS_PUZZLE_PROPOSER_RETRIES': '1'}
    with patch.dict(os.environ, env), patch.object(native, 'RUNS_DIR', root), \
         patch.object(native, 'PROMPT_ONLY', False), patch.object(benchmark, 'load_results', ordered_results):
        yield native, benchmark


def run_gpu_phase(args, plan):
    root = Path(plan['phase_root']).resolve()
    if args.phase == 'baseline':
        root.mkdir(parents=True, exist_ok=False)
        (root / 'plan.json').write_bytes(args.plan.read_bytes())
    else:
        verify_receipt(root, 'candidate-ready.json', args.plan)
    config = {'task_profiles': {'mh_val': {'dataset_path': plan['dataset'], 'limit': 32}},
              'models': [plan['target_config']], 'seeds': [42],
              'eval': {'assistant_token_budget': 8129, 'policy_max_tokens': 8129}}
    write_json(root / 'native-config.json', config)
    native_args = argparse.Namespace(run_name='search', config=str(root / 'native-config.json'), iterations=1,
        proposals_per_iter=1, proposer_model=MODEL, proposer_effort='low',
        propose_timeout=300, early_stop_success_rate=1., fresh=False, force=False,
        start_iteration=1, early_stop_min_iters=0, early_stop_patience=2, eval_only=False, use_api_key=True, prompt_only=False)
    with native_context(plan, root) as (native, benchmark):
        original_sweep = native.run_sweep
        def sweep(config, harnesses, logs_dir, **kwargs):
            result = original_sweep(config, harnesses, logs_dir, **kwargs)
            require(len(result) == len(harnesses) and all(ok for _, ok in result), 'Native evaluation failed')
            for name, _ in harnesses:
                paths = list(logs_dir.glob(f'*/{name}/*/val.json'))
                require(len(paths) == 1, 'Missing native summary')
                require(json.loads(paths[0].read_text())['num_examples'] == 32, 'Incomplete MH coverage')
            return result
        def proposal(**kwargs):
            require(kwargs['next_names'] == ['h1'] and kwargs['iteration'] == 1, 'Wrong proposal slot')
            if args.phase == 'baseline':
                write_json(root / 'proposal-request.json', {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()})
                raise ProposalBoundary()
            request = json.loads((root / 'proposal-request.json').read_text())
            require(kwargs['task_prompt'] == request['task_prompt'], 'Changed native proposer prompt')
            sessions = root / 'proposer-workspace/logs/claude_sessions'
            session, = sessions.iterdir()
            meta = json.loads((session / 'meta.json').read_text())
            return native.claude_wrapper.parse_stream_events((session / 'events.jsonl').read_text(),
                kwargs['task_prompt'], MODEL, meta['duration_seconds'], meta['exit_code'], cwd=meta['cwd'])
        def slots(run_dir, count):
            require(count == 1 and set(p.name for p in (run_dir / 'harnesses').iterdir()) <= {'h0', 'h1'}, 'Unexpected resumed slots')
            return ['h1']
        with patch.object(benchmark, 'evaluate_harness', lambda **kw: distributed_evaluation(plan, args.plan, root, **kw)), \
             patch.object(native, 'run_sweep', sweep), patch.object(native, 'propose_claude_with_retries', proposal), \
             patch.object(native, 'next_harness_names', slots):
            try:
                native.run_evolve(native_args)
            except ProposalBoundary:
                require(args.phase == 'baseline', 'Unexpected phase boundary')
                seal(root, 'baseline-ready.json', 'WAITING_LOGIN_PROPOSER', args.plan,
                     proposal_request_sha256=file_sha256(root / 'proposal-request.json'), new_api_calls=0,
                     slurm_job_id=os.environ.get('SLURM_JOB_ID'))
                return
    require(args.phase == 'candidate', 'Baseline unexpectedly passed proposal boundary')
    accepted = native.get_accepted_harness(root / 'search')
    seal(root, 'search-complete.json', 'COMPLETED_NATIVE_MH_ITERATION', args.plan, accepted=accepted,
         accepted_harness_sha256=file_sha256(root / f'search/harnesses/{accepted}/harness.py'),
         new_api_calls=0, slurm_job_id=os.environ.get('SLURM_JOB_ID'), training_steps=0,
         limitations=plan['limitations'])


def run_proposer(args, plan):
    require(not os.environ.get('SLURM_JOB_ID'), 'Proposer belongs on the networked login node')
    root = Path(plan['phase_root']).resolve()
    receipt = verify_receipt(root, 'baseline-ready.json', args.plan)
    audit = json.loads((root / 'baseline-audit.json').read_text())
    require(audit['kind'] == 'mh_phase_audit' and audit['status'] == 'PASS' and audit['harness'] == 'h0',
            'Complete baseline audit required before the paid proposer')
    require(audit['plan_sha256'] == file_sha256(args.plan) and
            audit['baseline_receipt_sha256'] == file_sha256(root / 'baseline-ready.json'), 'Wrong baseline audit')
    for relative, digest in audit['artifact_sha256'].items():
        p = root / relative
        require(p.resolve().is_relative_to(root) and file_sha256(p) == digest, f'Audited artifact changed: {relative}')
    require(file_sha256(root / 'proposal-request.json') == receipt['proposal_request_sha256'], 'Changed proposal request')
    request = json.loads((root / 'proposal-request.json').read_text())
    request['run_dir'] = Path(request['run_dir'])
    require(request['run_dir'] == root / 'search', 'Wrong search archive')
    journal = BudgetJournal(Path('data/glm-budget/ledger.jsonl'))
    before = journal.summary()
    with native_context(plan, root) as (native, _):
        result = isolated_proposal(native, plan, root, journal, **request)
        require(result.exit_code == 0, 'Proposer did not complete; preserve artifacts and usage')
        native.validate_candidate(root / 'search', 'h1')
    payload = normalize_metadata(json.loads((root / 'search/pending_eval.json').read_text()))
    write_json(root / 'search/pending_eval.json', payload)
    seal(root, 'candidate-ready.json', 'WAITING_CANDIDATE_GPU_EVALUATION', args.plan,
         budget_before=before, budget_after=journal.summary(),
         candidate_sha256=file_sha256(root / 'search/harnesses/h1/harness.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('baseline', 'propose', 'candidate', 'worker'), required=True)
    parser.add_argument('--replica', type=int, choices=(0, 1))
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = check_plan(args.plan)
    if args.check_only:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan_sha256': file_sha256(args.plan), 'phase': args.phase}))
    elif args.phase == 'worker':
        run_worker(args, plan)
    elif args.phase == 'propose':
        run_proposer(args, plan)
    else:
        run_gpu_phase(args, plan)
