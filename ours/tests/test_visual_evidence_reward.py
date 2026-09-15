"""Native visual reward and RSFT selection with explicitly scripted replies.

No task model or optimizer runs. Only Ray RPC transport and actor/checkpoint
operations are stubs; native image preprocessing, reward loading/decoding,
score placement and the original RSFT selection loop execute on CPU.
"""
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.visual_evidence_reward import DATA_SOURCE, compute_score


class VisualRewardProtocolTests(unittest.TestCase):
    def test_invalid_dataset_contracts_fail_and_claimed_reward_cannot_override_truth(self):
        for source, truth in [('chess_puzzles', 'A'), (DATA_SOURCE, 'a'), (DATA_SOURCE, ''), (DATA_SOURCE, None)]:
            with self.subTest(source=source, truth=truth), self.assertRaises(ValueError):
                compute_score(source, 'A', truth)
        claims = {'score': 100., 'acc': 1., 'ground_truth': 'A', 'rollout_reward_scores': {'accuracy': 1.}}
        before = deepcopy(claims)
        self.assertEqual(compute_score(DATA_SOURCE, 'A', 'B', claims), {'score': 0., 'acc': 0.})
        self.assertEqual(claims, before)
        with self.assertRaises(TypeError):
            compute_score(DATA_SOURCE, ['A'], 'A')


AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'qwen_vl_utils'))


@unittest.skipUnless(AVAILABLE, 'Requires native runtime and isolated visual overlay')
class VisualRewardNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing engineering pair and processor fixture; do not make
        # a new benchmark or read the frozen test split.
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        VisualDatasetNativeTests.setUpClass.__func__(cls)
        import pandas as pd
        for row in cls.rows:
            row['data_source'] = DATA_SOURCE
        pd.DataFrame(cls.rows).to_parquet(cls.path, index=False)

    def test_native_reward_rpc_score_placement_and_success_filter_keep_visual_tensors(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
        from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
        from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
        from verl.workers.rollout.replica import TokenOutput
        import verl.trainer.ppo.ray_trainer as trainer_module
        from ours.visual_evidence_dataset import VisualEvidenceDataset
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            for key in ('HARNESS_PATH', '_HARNESS_SYSTEM_PROMPT'):
                os.environ.pop(key, None)
            os.environ['_HARNESS_USER_PROMPT_TEMPLATE'] = 'Use the visible evidence. {question}'
            data_config = VisualDatasetNativeTests.config(self, Path(tmp))
            from verl.utils.dataset.rl_dataset import get_dataset_class
            data_config.custom_cls = {'path': 'pkg://ours.visual_evidence_dataset', 'name': 'VisualEvidenceDataset'}
            dataset_cls = get_dataset_class(data_config)
            self.assertIs(dataset_cls, VisualEvidenceDataset)
            dataset = dataset_cls(str(self.path), self.tokenizer, data_config, self.processor)
            indices = [0, 1, 1, 0]
            answers = [row['reward_model']['ground_truth'] for row in self.rows]
            mixed = [answers[0], answers[0], answers[1] + '.', 'Both A and B']
            reward_config = OmegaConf.create({'actor_rollout_ref': {'model': {'path': self.model}},
                'reward': {'custom_reward_function': {'path': 'pkg://ours.visual_evidence_reward', 'name': 'compute_score',
                    'reward_kwargs': {}}, 'reward_model': {'enable': False},
                    'reward_manager': {'source': 'register', 'name': 'naive'}}})

            async def make_batch(replies):
                reward_worker = RewardLoopWorker(reward_config)
                reward_requests = []
                async def reward_rpc(data):
                    reward_requests.append(data)
                    return await reward_worker.compute_score(data)
                worker = AgentLoopWorker.__new__(AgentLoopWorker)
                worker.processor = self.processor
                worker.tokenizer = self.tokenizer
                worker.rollout_config = OmegaConf.create({'prompt_length': 256, 'response_length': 16})
                worker.reward_loop_worker_handles = [SimpleNamespace(compute_score=SimpleNamespace(remote=reward_rpc))]
                outputs, requests, expected_masks = [], [], []
                for index, reply in zip(indices, replies, strict=True):
                    item = dataset[index]
                    item['extra_info'].update(score=100., acc=1., rollout_reward_scores={'accuracy': 1.})
                    response = self.tokenizer.encode(reply, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                    expected_masks.append(len(response))
                    class ScriptedServer:
                        async def generate(self, **kwargs):
                            requests.append(kwargs)
                            return TokenOutput(token_ids=response, log_probs=[-1.] * len(response), num_preempted=0,
                                extra_fields={'scripted_reply_fixture': True})
                    loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
                    loop.processor, loop.tokenizer = self.processor, self.tokenizer
                    loop.dataset_cls, loop.data_config = VisualEvidenceDataset, data_config
                    loop.apply_chat_template_kwargs = {'enable_thinking': False}
                    loop.loop, loop.server_manager = asyncio.get_running_loop(), ScriptedServer()
                    loop.response_length, loop.prompt_length = 16, 256
                    output = await loop.run({'temperature': 0.}, **item)
                    self.assertIsNone(output.reward_score)
                    outputs.append(await worker._agent_loop_postprocess(output, **item))
                batch = worker._postprocess(outputs)
                self.assertEqual(len(reward_requests), 4)
                self.assertTrue(all(set(request) == {'request_id', 'prompt_ids', 'sampling_params', 'image_data', 'video_data'}
                    for request in requests))
                self.assertTrue(all(len(request['image_data']) == 1 for request in requests))
                self.assertTrue(all(request['prompt_ids'] == requests[0]['prompt_ids'] for request in requests))
                self.assertEqual(batch.batch['response_mask'].sum(-1).tolist(), expected_masks)
                self.assertEqual(batch.batch['position_ids'].shape, (4, 4, 272))
                for mmi in batch.non_tensor_batch['multi_modal_inputs']:
                    self.assertEqual(mmi['images_seqlens'].tolist(), [480])
                    self.assertEqual(list(mmi['pixel_values'].shape), [480, 1536])
                return batch, expected_masks

            for name, replies, expected_scores, selected in [
                ('mixed', mixed, [1., 0., 1., 0.], [0, 2]),
                ('empty', ['No valid label'] * 4, [0., 0., 0., 0.], [])]:
                with self.subTest(scenario=name):
                    batch, mask_lengths = asyncio.run(make_batch(replies))
                    self.assertEqual(batch.batch['rm_scores'].sum(-1).tolist(), expected_scores)
                    for i, score in enumerate(expected_scores):
                        expected = torch.zeros(16)
                        expected[mask_lengths[i] - 1] = score
                        self.assertTrue(torch.equal(batch.batch['rm_scores'][i], expected))
                    trainer = trainer_module.RayPPOTrainer.__new__(trainer_module.RayPPOTrainer)
                    trainer.config = OmegaConf.create({'trainer': {'online_rsft': {'iterations': 0, 'sft_epochs': 1,
                        'score_threshold': 0.5}, 'save_freq': -1, 'test_freq': -1, 'rollout_data_dir': None},
                        'actor_rollout_ref': {'actor': {'ppo_mini_batch_size': 4, 'ppo_micro_batch_size_per_gpu': 1},
                            'rollout': {'temperature': 0., 'n': 1}}})
                    trainer.global_steps, trainer.total_training_steps = 0, 1
                    trainer.train_dataloader = [{'fixture_row': torch.arange(4).unsqueeze(1)}]
                    trainer.train_dataset, trainer.actor_rollout_wg = [], SimpleNamespace()
                    trainer.use_rm, trainer.use_critic = False, False
                    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
                    trainer._get_gen_batch = lambda incoming: incoming
                    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=lambda incoming: batch)
                    captured, synced, metrics = [], [], []
                    def actor_stub(accepted):
                        captured.append(accepted)
                        return DataProto(meta_info={'metrics': {}})
                    trainer._update_sft_actor = actor_stub
                    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None, update_weights=synced.append)
                    logger = SimpleNamespace(log=lambda data, step: metrics.append(data), finish=lambda: None)
                    trainer._fit_online_rsft(logger)
                    self.assertEqual(len(captured), int(bool(selected)))
                    if selected:
                        accepted = captured[0]
                        self.assertEqual(accepted.batch['fixture_row'].flatten().tolist(), selected)
                        self.assertTrue(torch.equal(accepted.batch['response_mask'], batch.batch['response_mask'][selected]))
                        for after, index in zip(accepted.non_tensor_batch['multi_modal_inputs'], selected, strict=True):
                            self.assertTrue(torch.equal(after['pixel_values'], batch.non_tensor_batch['multi_modal_inputs'][index]['pixel_values']))
                            self.assertTrue(torch.equal(after['image_grid_thw'], batch.non_tensor_batch['multi_modal_inputs'][index]['image_grid_thw']))
                    else:
                        self.assertEqual(metrics[0]['online_rsft/skipped_empty_sft'], 1.)
                    self.assertEqual(metrics[0]['online_rsft/accepted'], len(selected))
                    self.assertEqual(synced, [1])
                    print(json.dumps({'kind': 'native_visual_reward_and_success_selection_fixture', 'status': 'PASS',
                        'scenario': name, 'scripted_responses': 4, 'scores': expected_scores, 'selected_rows': selected,
                        'model_calls': 0, 'optimizer_steps_executed': 0, 'ray_transport_and_actor_are_stubs': True,
                        'scope': 'Native processor, SingleTurn, reward loader/manager/worker, score collation and RSFT selection.'}), flush=True)


if __name__ == '__main__':
    unittest.main()
