"""Reject altered training masks, rewards, positions and incomplete request pools."""
import asyncio
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ours.audit_native_training_batch import RequestPool, check_board, check_tensors, load_batch
from ours.native_training_trace import NativeTrainingJournal


def fixture():
    replay = SimpleNamespace(prompt_ids=[7], response_ids=[8, 9], response_mask=[1, 0], reward_score=1.)
    tensors = {'prompts': [0, 7], 'responses': [8, 9, 0], 'response_mask': [1, 0, 0],
               'attention_mask': [0, 1, 1, 1, 0], 'input_ids': [0, 7, 8, 9, 0],
               'position_ids': [[1, 0, 1, 2, 1] for _ in range(4)], 'token_level_scores': [0., 1., 0.]}
    return tensors, replay


class TensorAuditTests(unittest.TestCase):
    def test_exact_native_tokens_and_mask_pass(self):
        check_tensors(*fixture(), 0)

    def test_tool_or_padding_tokens_cannot_enter_loss(self):
        for index in (1, 2):
            tensors, replay = fixture()
            tensors['response_mask'][index] = 1
            with self.assertRaisesRegex(ValueError, 'response_mask'):
                check_tensors(tensors, replay, 0)

    def test_relocated_reward_is_rejected(self):
        tensors, replay = fixture()
        tensors['token_level_scores'] = [1., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'Reward tensor'):
            check_tensors(tensors, replay, 0)

    def test_changed_rotary_position_is_rejected(self):
        tensors, replay = fixture()
        tensors['position_ids'][2][2] = 8
        with self.assertRaisesRegex(ValueError, 'positions differ'):
            check_tensors(tensors, replay, 0)

    def test_batch_receipt_and_shape_are_both_checked(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            batch = root / 'batch-1.jsonl.gz'
            header = {'kind': 'native_training_batch_header', 'step': 1, 'count': 1,
                      'tensors': {'responses': {'shape': [1, 3], 'dtype': 'torch.int64'}}}
            row = {'kind': 'native_training_trajectory', 'index': 0, 'tensors': {'responses': [8, 9, 0]}}
            def write(row):
                with gzip.open(batch, 'wt') as stream:
                    stream.write(json.dumps(header) + '\n' + json.dumps(row) + '\n')
                receipt = {'kind': 'native_training_batch_receipt', 'status': 'COMPLETE', 'count': 1,
                           'step': 1, 'file': batch.name, 'sha256': hashlib.sha256(batch.read_bytes()).hexdigest()}
                (root / 'batch-1.receipt.json').write_text(json.dumps(receipt))
            write(row)
            self.assertEqual(load_batch(root)[1][0], row)
            row['tensors']['responses'].pop()
            write(row)
            with self.assertRaisesRegex(ValueError, 'shape differs'):
                load_batch(root)
            batch.write_bytes(b'changed after receipt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_batch(root)


class RequestPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_responses_keep_request_budgets_distinct(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            journal = NativeTrainingJournal(root)
            messages = [{'role': 'user', 'content': 'fixture'}]
            response = {'content': 'A', 'usage': {'completion_tokens': 1, 'prompt_tokens': 3, 'total_tokens': 4},
                        'raw_response': {'model': 'fixture', 'choices': [{'message': {'content': 'A'}}],
                                         'usage': {'completion_tokens': 1, 'prompt_tokens': 3, 'total_tokens': 4}}}
            async def invoke(**kwargs):
                return response
            for budget in (9, 3):
                await journal.call(invoke, request_id=str(budget), messages=messages,
                                   sampling_params={'max_tokens': budget}, context={'configured_model_path': 'fixture'})
            pool = RequestPool(root, 'fixture')
            manager, pending, used = pool.manager(['A', 'A'])
            for budget in (3, 9):
                await manager.chat_completion('replay', messages=messages, sampling_params={'max_tokens': budget})
            self.assertFalse(pending)
            self.assertEqual([pair[0]['sampling_params']['max_tokens'] for pair in used], [3, 9])
            self.assertFalse(any(pool.calls.values()))

    async def test_partial_request_cannot_pass_as_complete_batch(self):
        with tempfile.TemporaryDirectory() as root:
            NativeTrainingJournal(root).event({'kind': 'native_training_request', 'trace_id': 'p',
                                               'request_id': 'p', 'stage': 'started', 'context': {}})
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                RequestPool(Path(root), 'fixture')


@unittest.skipUnless(importlib.util.find_spec('chess'), 'Requires native chess dependency')
class IndependentBoardTests(unittest.TestCase):
    def test_unfinished_reasoning_follows_original_parser_without_final_answer_claim(self):
        info = {'start_fen': '8/6k1/1R5p/5p1P/5P1K/6P1/8/r7 b - - 3 58', 'solution_moves': ['a1h1']}
        events = [{'actor': 'assistant', 'raw_response': 'Thinking about [a1h1]', 'parsed_action': 'a1h1'},
                  {'actor': 'verifier', 'terminal': True, 'stop_condition': 'solved'}]
        check_board(info, events, 1., 'solved')
        corrupted = deepcopy(events)
        corrupted[0]['raw_response'] = 'Thinking about [a1a2]'
        corrupted[0]['parsed_action'] = 'a1a2'
        with self.assertRaisesRegex(ValueError, 'board verdict differs'):
            check_board(info, corrupted, 1., 'solved')
