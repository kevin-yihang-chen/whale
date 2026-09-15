"""Recording must preserve calls, failures, masks and native return values."""
import asyncio
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from ours.native_training_trace import NativeTrainingJournal, install_native_recorders, summarize_requests


class RequestJournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_response_and_arguments_are_preserved(self):
        messages = [{'role': 'user', 'content': 'training fixture'}]
        sampling = {'max_tokens': 19, 'temperature': 1.0}
        result = {'content': 'A', 'usage': {'completion_tokens': 1}, 'raw_response': {'id': 'fixture'}}
        async def invoke(**kwargs):
            self.assertIs(kwargs['messages'], messages)
            self.assertIs(kwargs['sampling_params'], sampling)
            return result
        with tempfile.TemporaryDirectory() as root:
            output = await NativeTrainingJournal(root).call(invoke, request_id='x', messages=messages,
                                                            sampling_params=sampling, context={})
            self.assertIs(output, result)
            rows = [json.loads(s) for s in next(Path(root).glob('requests-*.jsonl')).read_text().splitlines()]
            self.assertEqual([r['stage'] for r in rows], ['started', 'completed'])
            self.assertEqual(rows[1]['response'], result)
            self.assertEqual(rows[0]['trace_id'], rows[1]['trace_id'])
            accounting = summarize_requests(root)
            self.assertTrue(accounting['observed_requests_fully_accounted'])
            self.assertEqual(accounting['known_completion_tokens'], 1)

    async def test_failure_is_recorded_and_reraised(self):
        error = RuntimeError('fixture transport failure')
        async def invoke(**kwargs):
            raise error
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError) as caught:
                await NativeTrainingJournal(root).call(invoke, request_id='x', messages=[],
                                                       sampling_params={}, context={})
            self.assertIs(caught.exception, error)
            rows = [json.loads(s) for s in next(Path(root).glob('requests-*.jsonl')).read_text().splitlines()]
            self.assertEqual([r['stage'] for r in rows], ['started', 'error'])
            self.assertFalse(summarize_requests(root)['observed_requests_fully_accounted'])

    async def test_cancellation_keeps_a_started_request(self):
        async def invoke(**kwargs):
            raise asyncio.CancelledError()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(asyncio.CancelledError):
                await NativeTrainingJournal(root).call(invoke, request_id='x', messages=[],
                                                       sampling_params={}, context={})
            rows = [json.loads(s) for s in next(Path(root).glob('requests-*.jsonl')).read_text().splitlines()]
            self.assertEqual(rows[1]['error_type'], 'CancelledError')

    async def test_partial_journal_cannot_claim_zero_consumption(self):
        with tempfile.TemporaryDirectory() as root:
            NativeTrainingJournal(root).event({'kind': 'native_training_request', 'trace_id': 'pending',
                                               'request_id': 'x', 'stage': 'started', 'context': {}})
            path = next(Path(root).glob('requests-*.jsonl'))
            with path.open('ab') as stream:
                stream.write(b'{"incomplete')
            report = summarize_requests(root)
            self.assertEqual(report['pending_or_interrupted'], 1)
            self.assertEqual(report['partial_tail_records'], 1)
            self.assertFalse(report['observed_requests_fully_accounted'])

    async def test_orphan_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            NativeTrainingJournal(root).event({'kind': 'native_training_request', 'trace_id': 'orphan',
                                               'request_id': 'x', 'stage': 'completed', 'context': {}})
            with self.assertRaisesRegex(ValueError, 'Orphan'):
                summarize_requests(root)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('tensordict'),
                     'Requires native training runtime')
class NativeRecordingTests(unittest.IsolatedAsyncioTestCase):
    def setup_native(self, root):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from omegaconf import OmegaConf
        import verl.experimental.agent_loop.agent_loop as agent
        import verl.trainer.main_textarena_disagg_rsft as trainer
        config = OmegaConf.create({'trainer': {'default_local_dir': root},
                                   'actor_rollout_ref': {'model': {'path': 'fixture-only'}}})
        original = (agent.AsyncLLMServerManager, trainer.DisaggregatedRayTrainer)
        self.addCleanup(setattr, agent, 'AsyncLLMServerManager', original[0])
        self.addCleanup(setattr, trainer, 'DisaggregatedRayTrainer', original[1])
        return config, agent, trainer, original

    async def test_native_server_subclass_delegates_exactly_once(self):
        with tempfile.TemporaryDirectory() as root:
            config, agent, trainer, original = self.setup_native(root)
            expected = {'content': 'fixture', 'usage': {'completion_tokens': 2}}
            with patch.object(original[0], 'chat_completion', new=AsyncMock(return_value=expected)) as delegate:
                install_native_recorders()
                installed = agent.AsyncLLMServerManager
                install_native_recorders()
                self.assertIs(agent.AsyncLLMServerManager, installed)
                manager = installed.__new__(installed)
                manager.config = config
                messages, sampling = [{'role': 'user', 'content': 'fixture'}], {'max_tokens': 2}
                result = await manager.chat_completion('id', messages=messages, sampling_params=sampling)
                self.assertIs(result, expected)
                delegate.assert_awaited_once_with(request_id='id', messages=messages, sampling_params=sampling)

    async def test_native_batch_keeps_padded_tensors_and_events(self):
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl.protocol import DataProto
        with tempfile.TemporaryDirectory() as root:
            config, agent, trainer, original = self.setup_native(root)
            tensors = {'prompts': torch.tensor([[0, 7]]), 'responses': torch.tensor([[8, 9, 0]]),
                       'response_mask': torch.tensor([[1, 0, 0]]), 'attention_mask': torch.tensor([[0, 1, 1, 1, 0]]),
                       'input_ids': torch.tensor([[0, 7, 8, 9, 0]]),
                       'position_ids': torch.tensor([[0, 0, 1, 2, 0]]),
                       'token_level_scores': torch.tensor([[0., 1., 0.]])}
            events = {'events': [{'actor': 'assistant', 'raw_response': 'fixture'}]}
            batch = DataProto(batch=TensorDict(tensors, batch_size=[1]),
                              non_tensor_batch={'extras': np.array([events], dtype=object)},
                              meta_info={'temperature': 1.0})
            before = {k: v.clone() for k, v in tensors.items()}
            with patch.object(original[1], '_log_rollout_data', return_value='native-return') as delegate:
                install_native_recorders()
                instance = trainer.DisaggregatedRayTrainer.__new__(trainer.DisaggregatedRayTrainer)
                instance.config, instance.global_steps = config, 1
                extras, timing = {}, {}
                self.assertEqual(instance._log_rollout_data(batch, extras, timing, 'native-dir'), 'native-return')
                delegate.assert_called_once_with(batch, extras, timing, 'native-dir')
            for k, v in before.items():
                self.assertTrue(torch.equal(batch.batch[k], v))
            with gzip.open(Path(root) / 'audit/batch-1.jsonl.gz', 'rt') as stream:
                header, row = [json.loads(s) for s in stream]
            self.assertEqual(header['count'], 1)
            self.assertEqual(row['metadata']['extras'], events)
            self.assertEqual(row['tensors']['response_mask'], [[1, 0, 0]][0])
            self.assertEqual(row['tensors']['input_ids'], [0, 7, 8, 9, 0])
            self.assertTrue((Path(root) / 'audit/batch-1.receipt.json').exists())
