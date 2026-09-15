"""Selected-harness native CPU replay, with actual tokenizer and recorded fixtures.

Replies are explicit fixtures, never model measurements. No GPU or API calls.
"""
from collections import deque
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('chess'),
                     'Requires native Chess/training runtime')
class AlternationBatchAuditTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from transformers import AutoTokenizer
        from autoharness_chess_puzzle.harness import load_harness
        from ours.audit_alternation_training import runtime_configuration
        cls.plan = json.loads(Path('results/alternation-rsft-plan-20260909.json').read_text())
        cls.config, cls.context = runtime_configuration(cls.plan, '222292')
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.plan['model'], local_files_only=True)
        cls.harness = load_harness(cls.plan['harness'])

    def setUp(self):
        from omegaconf import OmegaConf
        self.config = deepcopy(type(self).config)
        self.environment = patch.dict(os.environ, {k: str(v) for k, v in OmegaConf.to_container(
            self.config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True).items()
            if k.startswith('CHESS_PUZZLE_')})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='whale-h1-audit-fixture-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    async def record(self, replies, solution):
        import chess
        from verl.experimental.agent_loop.agent_loop import DictConfigWrap
        from verl.experimental.agent_loop.chess_puzzle_agent_loop import ChessPuzzleAgentLoop
        from ours.native_training_trace import NativeTrainingJournal
        tokenizer, config = self.tokenizer, self.config
        expected = deque(replies)
        journal = NativeTrainingJournal(self.directory)
        context = {**self.context, 'fixture_only': True}
        async def invoke(*, messages, **kwargs):
            text = expected.popleft()
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=False, **dict(config.data.apply_chat_template_kwargs))
            count = len(tokenizer.encode(text, add_special_tokens=False))
            usage = {'prompt_tokens': len(prompt), 'completion_tokens': count,
                     'total_tokens': len(prompt) + count}
            return {'content': text, 'usage': usage, 'raw_response': {
                'model': self.plan['model'], 'usage': usage, 'choices': [{'message': {'content': text}}]}}
        class FixtureManager:
            async def chat_completion(self, **kwargs):
                return await journal.call(invoke, context=context, **kwargs)
        loop = ChessPuzzleAgentLoop(trainer_config=DictConfigWrap(config), server_manager=FixtureManager(),
            tokenizer=tokenizer, processor=None, dataset_cls=None, data_config=DictConfigWrap(config.data))
        info = {'puzzle_id': 'offline-selected-harness-fixture', 'start_fen': chess.STARTING_FEN,
                'solution_moves': solution}
        result = await loop.run({'temperature': 1., 'top_p': 1., 'top_k': 20,
                                 'repetition_penalty': 1., 'logprobs': False}, extra_info=info)
        self.assertFalse(expected)
        pwidth, rwidth = config.actor_rollout_ref.rollout.prompt_length, config.actor_rollout_ref.rollout.response_length
        prompt, response = result.prompt_ids[-pwidth:], result.response_ids
        pad = tokenizer.pad_token_id
        prompts = [pad] * (pwidth - len(prompt)) + prompt
        responses = response + [pad] * (rwidth - len(response))
        attention = [0] * (pwidth - len(prompt)) + [1] * (len(prompt) + len(response)) + [0] * (rwidth - len(response))
        positions, position = [], 0
        for active in attention:
            positions.append(position if active else 1)
            position += active
        scores = [0.] * rwidth
        if response:
            scores[len(response) - 1] = result.reward_score
        row = {'index': 0, 'metadata': {'extra_info': info, **result.extra_fields}, 'tensors': {
            'prompts': prompts, 'responses': responses, 'input_ids': prompts + responses,
            'response_mask': result.response_mask + [0] * (rwidth - len(response)),
            'attention_mask': attention, 'position_ids': [positions[:] for _ in range(4)],
            'token_level_scores': scores}}
        return row, result

    async def replay(self, row):
        from ours.audit_alternation_training import replay_row
        from ours.audit_native_training_batch import RequestPool
        pool = RequestPool(self.directory, self.plan['model'])
        info = row['metadata']['extra_info']
        report = await replay_row(row, pool=pool, config=self.config, tokenizer=self.tokenizer,
                                  harness=self.harness, by_id={info['puzzle_id']: info})
        self.assertFalse(any(pool.calls.values()))
        return report

    async def test_native_multiturn_retry_and_tampered_mask_reward_request(self):
        row, result = await self.record(['no move yet', '<move>e2e4</move>', '<move>g1f3</move>'],
                                         ['e2e4', 'e7e5', 'g1f3'])
        report = await self.replay(row)
        self.assertTrue(report['accepted'])
        self.assertEqual([s['reason'] for s in report['steps']], ['format_retry', 'continue', 'solved'])
        corrupted = deepcopy(row)
        tool_index = result.response_mask.index(0)
        corrupted['tensors']['response_mask'][tool_index] = 1
        with self.assertRaisesRegex(ValueError, 'response_mask'):
            await self.replay(corrupted)
        corrupted = deepcopy(row)
        corrupted['metadata']['reward_extra_info']['score'] = 0.
        with self.assertRaisesRegex(ValueError, 'outcome differs'):
            await self.replay(corrupted)
        path, = self.directory.glob('requests-*.jsonl')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]['messages'][0]['content'] += '\nAn unrecorded prompt mutation.'
        path.write_text(''.join(json.dumps(e) + '\n' for e in records))
        with self.assertRaisesRegex(ValueError, 'Unused expected replies|events differ'):
            await self.replay(row)

    async def test_independent_board_rejects_fake_solved_action(self):
        from ours.audit_alternation_training import board_continuation
        row, result = await self.record(['<move>e2e4</move>'], ['e2e4'])
        self.assertTrue((await self.replay(row))['board_solved'])
        wrong = deepcopy(result)
        wrong.extra_fields['extras']['events'][0]['parsed_action'] = 'd2d4'
        with self.assertRaisesRegex(ValueError, 'board verdict differs'):
            board_continuation(row['metadata']['extra_info'], wrong, self.harness)

    async def test_budget_exhaustion_and_solved_feedback_truncation(self):
        text = '<move>e2e4</move>'
        count = len(self.tokenizer.encode(text, add_special_tokens=False))
        self.config.actor_rollout_ref.rollout.multi_turn.max_assistant_tokens = count
        row, _ = await self.record([text], ['e2e4', 'e7e5', 'g1f3'])
        report = await self.replay(row)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['stop'], 'assistant_token_budget')
        for path in self.directory.iterdir():
            path.unlink()
        self.config.actor_rollout_ref.rollout.response_length = count + 1
        row, _ = await self.record([text], ['e2e4'])
        report = await self.replay(row)
        self.assertTrue(report['board_solved'])
        self.assertTrue(report['solved_but_feedback_truncated'])
        self.assertFalse(report['accepted'])

    async def test_reply_discarded_before_event_recording_cannot_pass(self):
        text = '<move>e2e4</move>'
        self.config.actor_rollout_ref.rollout.response_length = len(self.tokenizer.encode(text, add_special_tokens=False))
        row, result = await self.record([text], ['e2e4'])
        self.assertEqual(result.extra_fields['extras']['events'], [])
        with self.assertRaisesRegex(ValueError, 'events differ'):
            await self.replay(row)


if __name__ == '__main__':
    unittest.main()
