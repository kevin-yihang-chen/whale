"""Native visual E3 search with a final E2-E3 training-harness boundary.

Offline GPU evaluations and the networked proposer are explicit resumable
boundaries of the original run_evolve. The public proposer archive contains H
only. C receipts and training acceptance remain outside that archive. This
one-iteration engineering plan is not a formal multi-stage VETO experiment.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from .acceptance import CandidateMetrics, EvidenceConstrainedAcceptance
from .adapter import WHALEAcceptanceAdapter
from .evidence import EvaluationIdentity, PairPrediction, VisualPairEvaluator, fingerprint
from .glm_gateway import BudgetJournal, MODEL
from .isolated_visual_harness import load_isolated_visual_harness
from .native_search import tree_hashes
from .native_visual_service import ROOT, audit_requests, write_new
from .scoped_proposer import isolated_native_proposal
from .visual_dataset_materialization import single_rows
from .visual_native_evaluation import pair_inputs
from .visual_task import binary_answer_verifier, file_sha256 as path_sha256

CONTRACT = ROOT / 'ours/visual_search_contract.md'
SOURCES = ('ours/visual_search_bridge.py', 'ours/visual_search_contract.md', 'ours/adapter.py',
    'ours/acceptance.py', 'ours/evidence.py', 'ours/scoped_proposer.py', 'ours/glm_gateway.py',
    'ours/confined_exec.py', 'upstream/WHALE/domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py',
    'upstream/WHALE/domains/chess_puzzles/meta_harness/chess_puzzle_benchmark.py',
    'upstream/WHALE/domains/chess_puzzles/meta_harness/claude_wrapper.py')


def read(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    return path_sha256(Path(path))


def once(path, value):
    if path.exists():
        assert fingerprint(read(path)) == fingerprint(value), f'Changed boundary: {path}'
    else:
        write_new(path, value)


def certify(plan_path, allocation_path, *, with_audit=True):
    """Recompute H/E1 from complete actual records and bind their native evidence."""
    plan, allocation = read(plan_path), read(allocation_path)
    from .research_budget import terminal_allocation
    terminal = Path(allocation_path).parent / 'slurm-terminal.txt'
    assert file_sha256(terminal) == allocation['terminal_sha256']
    observed_allocation = terminal_allocation(terminal.read_text(), allocation['job_id'])
    assert observed_allocation['state'] == allocation['state'] and observed_allocation['seconds'] == allocation['seconds']
    root = Path(plan['output'])
    result, h = read(root / 'result.json'), read(root / 'H/result.json')
    assert allocation['state'] == 'COMPLETED' and allocation['job_id'] == result['job_id']
    assert result['status'] == 'COMPLETE_VISUAL_SEARCH_CANDIDATE'
    assert result['plan_sha256'] == file_sha256(plan_path) and result['identity'] == plan['identity']
    identity = EvaluationIdentity(**result['identity'])
    for path, digest in plan['source_sha256'].items():
        assert file_sha256(ROOT / path) == digest, path
    harness_path = Path(plan['config']['data']['visual_harness_path'])
    assert file_sha256(harness_path) == result['harness_sha256'] == plan['harness_sha256']
    for item in plan['partitions'].values():
        for key in ('manifest', 'parquet'):
            assert file_sha256(item[key]) == item[key + '_sha256']
    _, h_rows = single_rows(plan['partitions']['H']['manifest'])
    expected = {r['visual_sample_id']: r for r in h_rows}
    actual = {r['sample_id']: r for r in h['records']}
    assert len(h['records']) == len(actual) == len(expected) == 128 and set(actual) == set(expected)
    for key, row in actual.items():
        assert row['correct'] == int(binary_answer_verifier(row['committed_answer'], expected[key]['reward_model']['ground_truth']))
    candidate = CandidateMetrics(plan['candidate'], plan['harness_sha256'],
        sum(r['correct'] for r in actual.values()) / 128, sum(r['native_turns'] for r in actual.values()) / 128)
    assert asdict(candidate) == result['candidate'] and candidate.accuracy == h['accuracy']
    for name, digest in h['batch_artifact_sha256'].items():
        assert file_sha256(root / 'H' / name) == digest
    receipt = None
    if with_audit:
        c = read(root / 'C/result.json')
        manifest, pairs, c_rows = pair_inputs(plan['partitions']['C']['manifest'])
        assert manifest['role'] == 'C'
        c_actual = {r['sample_id']: r for r in c['records']}
        assert len(c['records']) == len(c_actual) == len(c_rows) == 128
        assert set(c_actual) == {r['visual_sample_id'] for r in c_rows}
        predictions = [PairPrediction(p.pair_id, tuple(c_actual[fingerprint({'pair_id': p.pair_id, 'side': side})]['committed_answer']
            for side in (0, 1))) for p in pairs]
        receipt = VisualPairEvaluator(binary_answer_verifier).evaluate(pairs, predictions,
            identity=identity, harness_sha256=candidate.harness_sha256, role='C')
        assert fingerprint(asdict(receipt)) == fingerprint(result['audit']) == fingerprint(c['audit'])
        for row in c_rows:
            assert c_actual[row['visual_sample_id']]['correct'] == int(binary_answer_verifier(
                c_actual[row['visual_sample_id']]['committed_answer'], row['reward_model']['ground_truth']))
        for name, digest in c['batch_artifact_sha256'].items():
            assert file_sha256(root / 'C' / name) == digest
    observed = result['observed']
    assert tree_hashes(root / 'requests') == observed['request_artifact_sha256']
    assert audit_requests(plan, root, observed) == observed
    return {'plan': plan, 'result': result, 'candidate': candidate, 'audit': receipt, 'h': h, 'h_rows': h_rows}


def boundary(adapter, native, run_dir, archive, identity, *, stage, frontier, rows=(), valid_names=(), resume=None):
    """Use the original selector verbatim when off; cover the early-stop bypass."""
    def original():
        hit = native.find_early_stop_candidate(rows, valid_names, 1.) if stage != 'initial' else None
        return hit['harness'] if hit else native.pick_accepted(frontier, run_dir)
    selected, receipt = adapter.decide(stage=stage, original_selector=original, incoming='h0',
        archive_provider=lambda: [x['candidate'] for x in archive],
        audit_provider=lambda c: next(x['audit'] for x in archive if x['candidate'].name == c.name),
        identity=identity, resume_receipt=resume)
    return {'accepted_harness': selected, 'upstream_harness': original(), 'receipt': receipt,
        'stop': {'search_finished': stage != 'initial', 'upstream_perfect_candidate':
                 native.find_early_stop_candidate(rows, valid_names, 1.),
                 'training_harness': selected, 'stage': stage}}


def init(root, baseline_plan, allocation):
    assert not root.exists()
    base = certify(baseline_plan, allocation)
    assert base['candidate'].name == 'h0' and base['plan']['repeatability_check']
    root.mkdir()
    plan = {'kind': 'native_visual_one_iteration_search', 'incoming': 'h0', 'identity': base['result']['identity'],
        'iterations': 1, 'proposals_per_iter': 1, 'slots': ['h1'], 'max_api_requests': 12, 'max_cli_turns': 12,
        'propose_timeout_seconds': 300, 'proposer_model': MODEL, 'mode': 'paired', 'epsilon': 0.,
        'source_sha256': {p: file_sha256(ROOT / p) for p in SOURCES},
        'baseline': {'plan': str(baseline_plan), 'allocation': str(allocation),
                     'plan_sha256': file_sha256(baseline_plan), 'allocation_sha256': file_sha256(allocation),
                     'result_sha256': file_sha256(Path(base['plan']['output']) / 'result.json')},
        'selection_pass': 'First complete H/C pass; never choose a better repeat.',
        'repeatability_passed': base['result']['repeatability_passed'],
        'limitations': ['One fixed theta1, one proposed harness, no formal h0/LR calibration or scientific effect claim.',
                       'Observed inference variability remains a limitation; repeated passes are diagnostic only.',
                       'C is optimization data. The proposer receives H only. No V, R or T are accessed.',
                       'The final accepted receipt, not a raw native accepted_harness.txt, must govern E4 resume.']}
    write_new(root / 'plan.json', plan)
    shutil.copyfile(CONTRACT, root / 'contract.md')
    run = root / 'search'
    (run / 'harnesses/h0').mkdir(parents=True)
    shutil.copyfile(base['plan']['config']['data']['visual_harness_path'], run / 'harnesses/h0/harness.py')
    (run / 'logs/claude_sessions').mkdir(parents=True)
    write_new(root / 'native-config.json', {'task_profiles': {'H': {'limit': 128}},
        'models': [{'model': 'qwen35-4b'}], 'seeds': [42]})


def load(root):
    plan = read(root / 'plan.json')
    assert plan['kind'] == 'native_visual_one_iteration_search'
    for name, digest in plan['source_sha256'].items():
        assert file_sha256(ROOT / name) == digest, name
    assert file_sha256(root / 'contract.md') == plan['source_sha256']['ours/visual_search_contract.md']
    base = plan['baseline']
    assert file_sha256(base['plan']) == base['plan_sha256'] and file_sha256(base['allocation']) == base['allocation_sha256']
    assert file_sha256(Path(read(base['plan'])['output']) / 'result.json') == base['result_sha256']
    return plan


def public_h_feedback(run, item):
    """Expose only H data through a positive field allowlist, without file paths."""
    name = item['candidate'].name
    target = run / 'logs/H' / name / 'qwen35-4b'
    target.mkdir(parents=True, exist_ok=True)
    h = item['h']
    value = {'num_examples': 128, 'success_rate': h['accuracy'], 'avg_reward': h['accuracy'],
        'mean_turn_count': h['mean_native_turns'], 'role': 'H', 'source': 'complete real visual first-pass evaluation'}
    once(target / 'val.json', value)
    records = {r['sample_id']: r for r in h['records']}
    feedback = [{'example': i, 'question': row['prompt'][-1]['content'].removeprefix('<image>\n'),
                 'model_answer': records[row['visual_sample_id']]['raw_answer'],
                 'correct': records[row['visual_sample_id']]['correct'],
                 'expected_answer': row['reward_model']['ground_truth']}
                for i, row in enumerate(item['h_rows'])]
    once(target / 'feedback.json', {'role': 'H', 'images_supplied_to_target_model': True, 'examples': feedback})


class PhaseBoundary(Exception):
    pass


def advance(root, candidate_plan=None, candidate_allocation=None):
    from meta_harness import meta_harness_chess_puzzle as native
    from meta_harness import chess_puzzle_benchmark as benchmark
    plan = load(root)
    base = certify(plan['baseline']['plan'], plan['baseline']['allocation'])
    identity = EvaluationIdentity(**plan['identity'])
    archive = [base]
    if candidate_plan:
        candidate = certify(candidate_plan, candidate_allocation)
        assert candidate['result']['identity'] == plan['identity'] and candidate['candidate'].name == 'h1'
        assert candidate['candidate'].harness_sha256 == file_sha256(root / 'search/harnesses/h1/harness.py')
        archive.append(candidate)
        once(root / 'candidate-evaluation.json', {'plan': str(candidate_plan), 'allocation': str(candidate_allocation),
            'result_sha256': file_sha256(Path(candidate['plan']['output']) / 'result.json')})
    adapter = WHALEAcceptanceAdapter(EvidenceConstrainedAcceptance(plan['mode'], epsilon=plan['epsilon']))
    run = root / 'search'
    original_pick, original_results = native.pick_accepted, benchmark.load_results
    initial = True
    def pick(frontier, run_dir):
        nonlocal initial
        winner = original_pick(frontier, run_dir)
        if initial:
            initial = False
            # Restore the original function while invoking the shared boundary.
            with patch.object(native, 'pick_accepted', original_pick):
                value = boundary(adapter, native, run_dir, [base], identity, stage='initial', frontier=frontier)
            once(root / 'initial-acceptance.json', value)
            return value['accepted_harness']
        return winner
    def sweep(config, harnesses, logs_dir, **kwargs):
        for name, path in harnesses:
            item = next((x for x in archive if x['candidate'].name == name), None)
            if item is None:
                once(root / 'evaluation-request.json', {'candidate': name, 'harness': str(path),
                    'harness_sha256': file_sha256(path), 'identity': plan['identity'],
                    'reference_evaluation_plan': plan['baseline']['plan'], 'with_audit': True,
                    'repeatability_check': True})
                raise PhaseBoundary('WAITING_CANDIDATE_GPU_EVALUATION')
            assert file_sha256(path) == item['candidate'].harness_sha256
            public_h_feedback(run, item)
        return [(name, True) for name, _ in harnesses]
    def propose(**kwargs):
        request = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
        once(root / 'proposal-request.json', request)
        ready = root / 'proposal-ready.json'
        if not ready.exists():
            raise PhaseBoundary('WAITING_NETWORKED_PROPOSER')
        receipt = read(ready)
        assert receipt['request_sha256'] == file_sha256(root / 'proposal-request.json')
        for name, digest in receipt['imported_sha256'].items():
            assert file_sha256(run / name) == digest
        session = Path(receipt['session'])
        assert tree_hashes(session) == receipt['session_sha256']
        metadata = read(session / 'meta.json')
        return native.claude_wrapper.parse_stream_events((session / 'events.jsonl').read_text(),
            kwargs['task_prompt'], MODEL, metadata['duration_seconds'], metadata['exit_code'], cwd=metadata['cwd'])
    args = argparse.Namespace(run_name='search', config=str(root / 'native-config.json'), iterations=1,
        proposals_per_iter=1, proposer_model=MODEL, proposer_effort='low', propose_timeout=300,
        early_stop_success_rate=1., fresh=False, force=False, start_iteration=1,
        early_stop_min_iters=0, early_stop_patience=2, eval_only=False, use_api_key=True, prompt_only=False)
    task = ('Run iteration 1 of visual harness search. Read the supplied H feedback and h0. '
            'Generate exactly one standalone candidate at harnesses/h1/harness.py and pending_eval.json. '
            'Improve ordinary chart-question accuracy within the supplied contract. No shell or external data.')
    with ExitStack() as stack:
        for obj, key, value in [(native, 'RUNS_DIR', root), (native, 'BASELINE_HARNESS', run / 'harnesses/h0/harness.py'),
            (native, 'PROMPT_ONLY', False), (native, 'SKILL_DIR', root / 'contract.md'),
            (native, 'render_task_prompt', lambda iteration, names: task),
            (native, 'next_harness_names', lambda run_dir, count: ['h1']), (native, 'run_sweep', sweep),
            (native, 'pick_accepted', pick), (native, 'load_harness', load_isolated_visual_harness),
            (native, 'propose_claude_with_retries', propose),
            (benchmark, 'load_results', lambda path: dict(sorted(original_results(path).items())))]:
            stack.enter_context(patch.object(obj, key, value))
        stack.enter_context(patch.dict('os.environ', {'BASELINE_HARNESS_OVERRIDE': ''}))
        try:
            native.run_evolve(args)
        except PhaseBoundary as pending:
            print(json.dumps({'status': str(pending), 'root': str(root)}), flush=True)
            return
    comparison = read(run / 'logs/iteration_001/comparison.json')
    stage = 'early_stop' if comparison['early_stop'] else 'ordinary'
    value = boundary(adapter, native, run, archive, identity, stage=stage, frontier=comparison['frontier'],
        rows=comparison['summary'], valid_names=['h1'])
    assert value['upstream_harness'] == native.get_accepted_harness(run)
    value.update(status='COMPLETE_SEARCH_AND_TRAINING_ACCEPTANCE', plan_sha256=file_sha256(root / 'plan.json'),
        evaluations={x['candidate'].name: {'candidate': asdict(x['candidate']), 'audit': asdict(x['audit']),
            'repeatability_passed': x['result']['repeatability_passed']} for x in archive},
        accepted_harness_path=str(run / 'harnesses' / value['accepted_harness'] / 'harness.py'),
        accepted_harness_sha256=file_sha256(run / 'harnesses' / value['accepted_harness'] / 'harness.py'),
        limitations=plan['limitations'])
    once(root / 'accepted-for-training.json', value)
    restored = boundary(adapter, native, run, archive, identity, stage='resume', frontier=comparison['frontier'],
        rows=comparison['summary'], valid_names=['h1'], resume=value['receipt'])
    once(root / 'acceptance-resume.json', restored)
    print(json.dumps({k: value[k] for k in ('status', 'accepted_harness', 'upstream_harness')}), flush=True)


def propose(root):
    from meta_harness import meta_harness_chess_puzzle as native
    plan = load(root)
    assert not (root / 'paid-proposal').exists() and not (root / 'proposal-ready.json').exists()
    request_path = root / 'proposal-request.json'
    request = read(request_path)
    assert request['next_names'] == ['h1'] and request['iteration'] == 1
    assert request['run_dir'] == str(root / 'search')
    request['run_dir'] = Path(request['run_dir'])
    # Revalidate actual evidence before crossing the paid, networked boundary.
    certify(plan['baseline']['plan'], plan['baseline']['allocation'])
    journal = BudgetJournal(ROOT / 'data/glm-budget/ledger.jsonl')
    budget_before = journal.summary()
    system = ('Generate visual chart-question harnesses using only the supplied H archive. '
        'Follow the visual contract. No shell commands, external paths, benchmark answers or hidden data. '
        'Write only allocated harnesses, pending_eval.json and the iteration report.')
    with patch.object(native, 'SKILL_DIR', root / 'contract.md'), patch.object(native, 'PROMPT_ONLY', False), \
         patch.object(native, 'PROPOSER_SYSTEM_PROMPT', system):
        result = isolated_native_proposal(native, {**plan, 'data_role': 'mh_val'}, root / 'paid-proposal', journal, **request)
    # mh_val is the reused transport's legacy role name; only visual H was copied.
    assert result.exit_code == 0, 'Preserve failed proposal artifacts and fees; no automatic paid retry'
    harness = load_isolated_visual_harness(root / 'search/harnesses/h1/harness.py')
    assert isinstance(harness.invoke('format_observation', question='Which bar is higher? A: left; B: right.'), str)
    sessions = list((root / 'search/logs/claude_sessions').iterdir())
    assert len(sessions) == 1
    integrity = read(root / 'paid-proposal/proposal-integrity.json')
    write_new(root / 'proposal-ready.json', {'status': 'VALIDATED_REAL_GLM_VISUAL_CANDIDATE',
        'request_sha256': file_sha256(request_path), 'session': str(sessions[0]),
        'session_sha256': tree_hashes(sessions[0]),
        'imported_sha256': {p: file_sha256(root / 'search' / p) for p in set(integrity['imported']) | {'pending_eval.json'}},
        'actual_data_role': 'H', 'legacy_transport_role': 'mh_val',
        'budget_before': budget_before, 'budget_after': journal.summary()})
    print(json.dumps({'status': 'VALIDATED_REAL_GLM_VISUAL_CANDIDATE', 'budget': journal.summary()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('init', 'advance', 'propose'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    for name in ('baseline-plan', 'allocation', 'candidate-plan', 'candidate-allocation'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'init':
        init(root, args.baseline_plan.resolve(), args.allocation.resolve())
    elif args.phase == 'propose':
        propose(root)
    else:
        advance(root, args.candidate_plan.resolve() if args.candidate_plan else None,
                args.candidate_allocation.resolve() if args.candidate_allocation else None)
