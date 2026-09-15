"""Check original-runner shard equivalence and the original staged search loop.

All responses/evaluation scores here are CPU fixtures, with zero model/API calls.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('chess') and importlib.util.find_spec('transformers'),
                     'Requires native Chess runtime')
class MHPhaseTests(unittest.TestCase):
    def test_actual_worker_subclass_delegates_loading_then_reads_model_values(self):
        import torch
        from vllm.v1.worker.gpu_worker import Worker
        from ours.phase_vllm_worker import CheckpointVerifiedWorker
        class Qwen3_5ForConditionalGeneration(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[1., 2.]], dtype=torch.bfloat16))
            def embed_input_ids(self, ids):
                return self.weight[ids]
        model = Qwen3_5ForConditionalGeneration()
        worker = CheckpointVerifiedWorker.__new__(CheckpointVerifiedWorker)
        worker.model_runner = SimpleNamespace(get_model=lambda: model)
        with tempfile.TemporaryDirectory(prefix='whale-mh-worker-') as tmp:
            path, receipt = Path(tmp) / 'plan.json', Path(tmp) / 'receipt.json'
            path.write_text(json.dumps({'worker_probe_coordinates': [{'token_id': 0, 'column': 1, 'base': 0., 'updated': 2.}]}))
            with patch.dict(os.environ, WHALE_MH_PLAN=str(path), WHALE_MH_WORKER_RECEIPT=str(receipt)), \
                 patch.object(Worker, 'load_model', autospec=True) as load:
                worker.load_model()
                load.assert_called_once_with(worker, load_dummy_weights=False)
                self.assertEqual(json.loads(receipt.read_text())['actual_worker']['values'], [2.])
                with self.assertRaisesRegex(ValueError, 'real weights'):
                    worker.load_model(load_dummy_weights=True)

    def test_native_runner_shards_preserve_full_outputs_and_redaction(self):
        import chess
        from autoharness_chess_puzzle import runner
        from ours.local_completion import CompletionResponse
        from ours.mh_phase import merge_outputs
        from ours.vllm_runtime import VLLMConfiguration
        dataset = 'data/chess-pilot-v1/lichess_puzzles_mh_val_32.parquet'
        original_read = runner._read_examples
        examples = original_read(dataset, limit=32, seed=42)
        ids = [e.example_id for e in examples]
        actions = {}
        for index, example in enumerate(examples):
            board = chess.Board(example.start_fen)
            for step, move in enumerate(example.solution_moves):
                if step % 2 == 0:
                    actions[board.fen()] = f'<move>{move}</move>' if index % 2 == 0 else 'fixture_no_move'
                board.push_uci(move)
        class FixtureClient:
            def __init__(self, config):
                self.config = config
            def complete_response(self, messages, *, max_tokens=None):
                fen = re.search(r'^FEN: (.+)$', messages[-1].content, re.M).group(1)
                return CompletionResponse(actions[fen], {'completion_tokens': 1, 'prompt_tokens': 1, 'total_tokens': 2})
        original_summary = runner.summarize_outputs
        captured = []
        def capture(rows, metadata):
            captured.append(rows)
            return original_summary(rows, metadata)
        with tempfile.TemporaryDirectory(prefix='whale-native-mh-shards-') as tmp, \
             patch.object(runner, 'LLMClient', FixtureClient), patch.object(runner, 'LLMConfig', VLLMConfiguration), \
             patch.object(runner, 'summarize_outputs', capture):
            kwargs = dict(harness_path='upstream/WHALE/domains/chess_puzzles/environments/chess_puzzle/base_harness.py',
                dataset_path=dataset, llm_config={'provider': 'local-vllm', 'model': 'fixture',
                'base_url': 'http://127.0.0.1:1', 'batch_concurrency': 4},
                limit=32, seed=42, assistant_token_budget=8129, policy_max_tokens=8129)
            full = runner.evaluate_harness(**kwargs, output_dir=Path(tmp) / 'full')
            full_outputs = captured.pop()
            shards = []
            for replica in range(2):
                assigned = sorted(ids)[replica::2]
                selected = [e for e in examples if e.example_id in assigned]
                with patch.object(runner, '_read_examples', return_value=selected):
                    runner.evaluate_harness(**kwargs, output_dir=Path(tmp) / f'shard-{replica}')
                shards.append({'ids': [e.example_id for e in selected], 'outputs': captured.pop()})
            combined = merge_outputs({'dataset': dataset}, shards)
            self.assertEqual(combined, full_outputs)
            self.assertEqual(original_summary(combined, full['metadata']), full)
            self.assertEqual(full['solved_examples'], 16)
            self.assertEqual({o['example_id'] for o in combined}, {'redacted'})
            bad = [shards[0], shards[0]]
            with self.assertRaisesRegex(ValueError, 'Duplicate or missing'):
                merge_outputs({'dataset': dataset}, bad)

    def test_original_loop_pauses_resumes_without_repeating_h0_and_has_fixed_ties(self):
        from autoharness_chess_puzzle import runner
        from meta_harness import meta_harness_chess_puzzle as native
        from ours.mh_phase import run_gpu_phase, run_proposer, seal, verify_receipt
        from ours.native_search import write_json
        for candidate_score, expected in ((0., 'h0'), (1., 'h1')):
            with self.subTest(candidate_score=candidate_score), tempfile.TemporaryDirectory(prefix='whale-staged-mh-') as tmp:
                root = Path(tmp) / 'phase'
                plan_path = Path(tmp) / 'plan.json'
                plan = {'phase_root': str(root), 'dataset': 'fixture-only', 'target_config': {'model': 'fixture'},
                        'limitations': ['CPU fixture, no inference or API calls']}
                write_json(plan_path, plan)
                evaluated = []
                def evaluate(_plan, _plan_path, _root, **kwargs):
                    name = Path(kwargs['harness_path']).parent.name
                    evaluated.append(name)
                    output = Path(kwargs['output_dir'])
                    output.mkdir(parents=True)
                    score = 0. if name == 'h0' else candidate_score
                    rows = [{'example_id': 'redacted', 'reward': score, 'turn_count': 1,
                             'metrics': {'assistant_tokens_used': 1}, 'prompt': [], 'harness_trace': []} for _ in range(32)]
                    summary = runner.summarize_outputs(rows, {'seed': 42, 'assistant_token_budget': 8129})
                    write_json(output / 'val.json', summary)
                    runner.write_trajectories(rows, output)
                    return summary
                args = SimpleNamespace(plan=plan_path, phase='baseline')
                with patch('ours.mh_phase.distributed_evaluation', evaluate):
                    run_gpu_phase(args, plan)
                    self.assertEqual(evaluated, ['h0'])
                    self.assertEqual(verify_receipt(root, 'baseline-ready.json', plan_path)['status'], 'WAITING_LOGIN_PROPOSER')
                    with patch('ours.mh_phase.isolated_proposal') as paid:
                        with self.assertRaises(FileNotFoundError):
                            run_proposer(SimpleNamespace(plan=plan_path), plan)
                        paid.assert_not_called()
                    candidate = root / 'search/harnesses/h1'
                    candidate.mkdir()
                    shutil.copyfile(root / 'search/harnesses/h0/harness.py', candidate / 'harness.py')
                    write_json(root / 'search/pending_eval.json', {'candidates': [{'name': 'h1', 'axis': 'fixture'}]})
                    session = root / 'proposer-workspace/logs/claude_sessions/fixture'
                    session.mkdir(parents=True)
                    write_json(session / 'meta.json', {'duration_seconds': 0., 'exit_code': 0, 'cwd': str(root)})
                    (session / 'events.jsonl').write_text('fixture\n')
                    seal(root, 'candidate-ready.json', 'WAITING_CANDIDATE_GPU_EVALUATION', plan_path)
                    with patch.object(native.claude_wrapper, 'parse_stream_events', return_value=SimpleNamespace(show=lambda: None)):
                        run_gpu_phase(SimpleNamespace(plan=plan_path, phase='candidate'), plan)
                    self.assertEqual(evaluated, ['h0', 'h1'])
                    self.assertEqual(verify_receipt(root, 'search-complete.json', plan_path)['accepted'], expected)
                    # Historical snapshot remains verifiable after the live search evolves.
                    verify_receipt(root, 'baseline-ready.json', plan_path, verify_live=False)
                    with self.assertRaisesRegex(ValueError, 'Live search differs'):
                        verify_receipt(root, 'baseline-ready.json', plan_path)


if __name__ == '__main__':
    unittest.main()
