"""Native crop/tool/answer/mask integration with scripted model replies.

The real dataset, processor, native tool parser/state machine, PIL crop, reward
and DataProto collation run on CPU. No task model, actor or optimizer executes.
"""
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'qwen_vl_utils'))


@unittest.skipUnless(AVAILABLE, 'Requires native runtime and isolated visual overlay')
class VisualToolNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        VisualDatasetNativeTests.setUpClass.__func__(cls)

    def test_native_crop_final_answer_and_tool_masks_share_one_execution_path(self):
        from ours.visual_evidence_tool_loop import VisualEvidenceToolAgentLoop, VisualEvidenceZoomTool
        from ours.visual_evidence_dataset import VisualEvidenceDataset
        from ours.visual_evidence_reward import DATA_SOURCE
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap, _agent_loop_registry
        from verl.workers.rollout.replica import TokenOutput
        from omegaconf import OmegaConf
        import hydra
        import torch

        tool_path = str(Path('ours/visual_evidence_tools.yaml').resolve())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            for key in ('HARNESS_PATH', '_HARNESS_SYSTEM_PROMPT', '_HARNESS_USER_PROMPT_TEMPLATE'):
                os.environ.pop(key, None)
            data_config = VisualDatasetNativeTests.config(self, Path(tmp))
            data_config.tool_config_path = tool_path
            dataset = VisualEvidenceDataset(str(self.path), self.tokenizer, data_config, self.processor)
            config = OmegaConf.create({'actor_rollout_ref': {'model': {'path': self.model}, 'rollout': {
                'prompt_length': 4096, 'response_length': 1024, 'multi_turn': {
                    'max_user_turns': 3, 'max_assistant_turns': 3, 'max_assistant_tokens': 128,
                    'max_parallel_calls': 1, 'max_tool_response_length': 512,
                    'tool_response_truncate_side': 'right', 'tool_config_path': tool_path,
                    'format': 'qwen3_coder', 'interaction_config_path': None}}}})
            crop = '<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>[0, 0, 160, 160]</parameter>\n</function>\n</tool_call>'
            invalid_crop = crop.replace('[0, 0, 160, 160]', '[0, 0, 0, 0]')

            async def exercise(name, row_index, replies, expected_score, expected_images):
                item = dataset[row_index]
                item['data_source'] = DATA_SOURCE
                item['tools_kwargs'] = {'image_zoom_in_tool': {'create_kwargs': {'image': 'unused://metadata-image'}}}
                if name == 'no_image':
                    for message in item['raw_prompt']:
                        if isinstance(message['content'], list):
                            # A valid hidden image is a stronger leakage probe
                            # than a path that could fail to load on its own.
                            hidden_images = [part['image'] for part in message['content'] if part['type'] == 'image']
                            if hidden_images:
                                item['tools_kwargs']['image_zoom_in_tool']['create_kwargs']['image'] = hidden_images[0].copy()
                            message['content'] = [part for part in message['content'] if part['type'] != 'image']
                source_item = deepcopy(item)
                responses = [self.tokenizer.encode(reply, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                    for reply in replies]
                requests = []
                class ScriptedServer:
                    async def generate(inner, **kwargs):
                        index = len(requests)
                        requests.append(deepcopy(kwargs))
                        if index >= len(responses):
                            raise AssertionError('Unexpected extra model request')
                        return TokenOutput(token_ids=list(responses[index]), log_probs=[-1.] * len(responses[index]),
                            num_preempted=0, extra_fields={'scripted_response_fixture': True})
                case_config = deepcopy(config)
                if name == 'no_final_answer':
                    case_config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 1
                agent_configs = OmegaConf.load('ours/visual_evidence_agents.yaml')
                self.assertEqual(len(agent_configs), 1)
                agent_config = agent_configs[0]
                self.assertEqual(agent_config.name, 'visual_evidence_tool_agent')
                self.assertEqual(agent_config._target_, _agent_loop_registry[agent_config.name]['_target_'])
                loop = hydra.utils.instantiate(agent_config,
                    trainer_config=DictConfigWrap(case_config), server_manager=ScriptedServer(), tokenizer=self.tokenizer,
                    processor=self.processor, dataset_cls=VisualEvidenceDataset, data_config=DictConfigWrap(data_config))
                self.assertIsInstance(loop, VisualEvidenceToolAgentLoop)
                self.assertIsInstance(loop.tools['image_zoom_in_tool'], VisualEvidenceZoomTool)
                output = await loop.run({'temperature': 0., 'max_tokens': 64}, **item)
                self.assertEqual(len(requests), len(replies))
                self.assertEqual(output.reward_score, expected_score)
                self.assertEqual(output.extra_fields['reward_extra_info'], {'score': expected_score, 'acc': expected_score})
                self.assertEqual(output.extra_fields['visual_final_answer'], self.tokenizer.decode(responses[-1], skip_special_tokens=True))
                self.assertEqual(sum(output.response_mask), sum(map(len, responses)))
                self.assertEqual(output.response_ids[:len(responses[0])], responses[0])
                self.assertEqual(output.response_ids[-len(responses[-1]):], responses[-1])
                self.assertEqual(output.response_mask[-len(responses[-1]):], [1] * len(responses[-1]))
                if len(replies) > 1:
                    middle = output.response_mask[len(responses[0]):-len(responses[-1])]
                    self.assertTrue(middle)
                    self.assertEqual(sum(middle), 0)
                self.assertEqual(len(output.multi_modal_data.get('images') or []), expected_images)
                self.assertTrue(all(set(request) == {'request_id', 'prompt_ids', 'sampling_params', 'image_data', 'video_data'}
                    for request in requests))
                self.assertTrue(all(request['request_id'] == requests[0]['request_id'] for request in requests))
                if expected_images == 2:
                    self.assertEqual(len(requests[0]['image_data']), 1)
                    self.assertEqual(len(requests[1]['image_data']), 2)
                    initial, cropped = requests[1]['image_data']
                    self.assertEqual(cropped.tobytes(), initial.crop((0, 0, 160, 160)).tobytes())
                    self.assertEqual(cropped.size, (160, 160))
                self.assertEqual(loop.tools['image_zoom_in_tool']._instance_dict, {})
                self.assertEqual(item['tools_kwargs'], source_item['tools_kwargs'])
                worker = AgentLoopWorker.__new__(AgentLoopWorker)
                worker.processor, worker.tokenizer = self.processor, self.tokenizer
                worker.rollout_config = config.actor_rollout_ref.rollout
                worker.reward_loop_worker_handles = None
                internal = await worker._agent_loop_postprocess(output, **item)
                batch = worker._postprocess([internal])
                self.assertEqual(batch.batch['rm_scores'].sum().item(), expected_score)
                self.assertEqual(batch.batch['response_mask'].sum().item(), sum(map(len, responses)))
                mmi = batch.non_tensor_batch['multi_modal_inputs'][0]
                self.assertEqual(batch.batch['position_ids'].shape[1], 4)
                image_tokens = batch.batch['input_ids'][0] == self.processor.image_token_id
                if expected_images:
                    self.assertEqual(len(mmi['image_grid_thw']), expected_images)
                    self.assertEqual(mmi['pixel_values'].shape[0], int(mmi['images_seqlens'].sum()))
                    self.assertEqual(int(image_tokens.sum()), int(mmi['image_grid_thw'].prod(-1).sum()) // self.processor.image_processor.merge_size**2)
                else:
                    self.assertNotIn('pixel_values', mmi)
                    self.assertEqual(int(image_tokens.sum()), 0)
                    self.assertTrue(all(not request['image_data'] for request in requests))
                reply_image_tokens = image_tokens[-1024:]
                self.assertEqual(int(batch.batch['response_mask'][0][reply_image_tokens].sum()), 0)
                if name == 'invalid_crop':
                    self.assertEqual(output.extra_fields['tool_rewards'], [-0.05])
                    self.assertEqual(output.reward_score, 1.)
                print(json.dumps({'kind': 'native_visual_tool_loop_fixture', 'scenario': name, 'status': 'PASS',
                    'scripted_requests': len(requests), 'tool_images': max(expected_images - 1, 0),
                    'initial_image_present': bool(expected_images),
                    'assistant_loss_tokens': sum(output.response_mask),
                    'masked_observation_tokens': len(output.response_mask) - sum(output.response_mask),
                    'binary_task_reward': output.reward_score, 'model_calls': 0, 'optimizer_steps': 0}), flush=True)

            answers = [row['reward_model']['ground_truth'] for row in self.rows]
            for scenario in [('crop_correct', 0, [crop, answers[0]], 1., 2),
                ('crop_wrong', 1, [crop, answers[0]], 0., 2),
                ('invalid_crop', 0, [invalid_crop, answers[0]], 1., 1),
                ('no_image', 0, [crop, answers[0]], 1., 0),
                ('no_final_answer', 0, [crop], 0., 1),
                ('direct_answer', 1, [answers[1]], 1., 1)]:
                with self.subTest(scenario=scenario[0]):
                    asyncio.run(exercise(*scenario))


if __name__ == '__main__':
    unittest.main()
