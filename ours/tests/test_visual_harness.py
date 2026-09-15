"""Real visual candidate dispatch with scripted replies; no model training."""
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
class VisualHarnessNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        VisualDatasetNativeTests.setUpClass.__func__(cls)

    def test_candidate_contract_and_prompt_subspace_reject_program_or_state_changes(self):
        from ours.visual_harness import BASE, load_visual_harness
        source = BASE.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidate.py'
            path.write_text(source.replace('USER_PROMPT = "{question}"', 'USER_PROMPT = "Inspect the image. {question}"'))
            candidate = load_visual_harness(path, prompt_reference=BASE)
            self.assertEqual(candidate.invoke('format_observation', question='Q'), 'Inspect the image. Q')
            with self.assertRaises(ValueError):
                candidate.invoke('parse_answer', text='A', ground_truth='A')
            for modification in [
                source.replace('return dict(arguments)', "return {'bbox_2d': [0, 0, 128, 128]}"),
                source + '\nimport os\n', source + '\nfrom re import _compiler\n',
                source + '\nSEEN = {}\n',
                source.replace('return dict(arguments)', 'global SEEN\n    SEEN = 1\n    return dict(arguments)')]:
                path.write_text(modification)
                with self.assertRaises(ValueError):
                    load_visual_harness(path, prompt_reference=BASE)
            with self.assertRaises(ValueError):
                candidate.invoke('parse_answer', text='A')
            for declaration in ('def parse_answer(text, seen=[]):',
                                'def parse_answer(text, *, seen={}):'):
                path.write_text(source.replace('def parse_answer(text):', declaration))
                with self.assertRaisesRegex(ValueError, 'defaults must be immutable'):
                    load_visual_harness(path)

    def test_default_parity_executable_hooks_and_native_continuation_masks(self):
        from ours.visual_harness import BASE
        from ours.visual_harness_dataset import VisualHarnessDataset
        from ours.visual_harness_loop import VisualHarnessAgentLoop
        from ours.visual_evidence_tool_loop import VisualEvidenceToolAgentLoop
        from ours.visual_evidence_dataset import VisualEvidenceDataset
        from ours.visual_evidence_reward import DATA_SOURCE
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.experimental.agent_loop.agent_loop import DictConfigWrap, AgentLoopWorker
        from verl.workers.rollout.replica import TokenOutput
        from omegaconf import OmegaConf
        import hydra
        import torch

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            for key in ('HARNESS_PATH', '_HARNESS_SYSTEM_PROMPT', '_HARNESS_USER_PROMPT_TEMPLATE', 'WHALE_VISUAL_HARNESS_PATH'):
                os.environ.pop(key, None)
            root = Path(tmp)
            data_config = VisualDatasetNativeTests.config(self, root)
            data_config.tool_config_path = str(Path('ours/visual_evidence_tools.yaml').resolve())
            data_config.custom_cls = {'path': 'pkg://ours.visual_harness_dataset', 'name': 'VisualHarnessDataset'}
            data_config.visual_harness_path = str(BASE.resolve())
            def make_dataset(cfg):
                return create_rl_dataset(str(self.path), cfg, self.tokenizer, self.processor, is_train=True)
            dataset = make_dataset(data_config)
            self.assertIsInstance(dataset, VisualHarnessDataset)
            self.assertEqual(dataset.dataframe._fingerprint, make_dataset(data_config).dataframe._fingerprint)
            legacy_config = deepcopy(data_config)
            legacy_config.custom_cls = {'path': 'pkg://ours.visual_evidence_dataset', 'name': 'VisualEvidenceDataset'}
            legacy = make_dataset(legacy_config)
            config = OmegaConf.create({'actor_rollout_ref': {'model': {'path': self.model}, 'rollout': {
                'prompt_length': 4096, 'response_length': 1024, 'multi_turn': {
                    'max_user_turns': 3, 'max_assistant_turns': 3, 'max_assistant_tokens': 128,
                    'max_parallel_calls': 1, 'max_tool_response_length': 512, 'tool_response_truncate_side': 'right',
                    'tool_config_path': data_config.tool_config_path, 'format': 'qwen3_coder', 'interaction_config_path': None}}}})
            crop = '<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>[0, 0, 160, 160]</parameter>\n</function>\n</tool_call>'
            truth = self.rows[0]['reward_model']['ground_truth']

            async def run(ds, cfg, replies, *, native_legacy=False, max_turns=3, wrong_identity=False):
                item = ds[0];item['data_source'] = DATA_SOURCE
                if wrong_identity:
                    item['visual_harness_sha256'] = '0' * 64
                responses = [self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id] for text in replies]
                requests = []
                class Server:
                    async def generate(inner, **kwargs):
                        index = len(requests);requests.append(deepcopy(kwargs))
                        if index >= len(responses):raise AssertionError('Unexpected extra request')
                        return TokenOutput(token_ids=list(responses[index]), log_probs=[-1.] * len(responses[index]),
                            num_preempted=0, extra_fields={'scripted_response_fixture': True})
                trial = deepcopy(config)
                trial.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = max_turns
                arguments = dict(trainer_config=DictConfigWrap(trial), server_manager=Server(), tokenizer=self.tokenizer,
                    processor=self.processor, dataset_cls=type(ds), data_config=DictConfigWrap(cfg))
                if native_legacy:
                    loop = VisualEvidenceToolAgentLoop(**arguments)
                else:
                    loop = hydra.utils.instantiate(OmegaConf.load('ours/visual_harness_agents.yaml')[0], **arguments)
                    self.assertIsInstance(loop, VisualHarnessAgentLoop)
                if wrong_identity:
                    with self.assertRaisesRegex(ValueError, 'identities differ'):
                        await loop.run({'temperature': 0., 'max_tokens': 64}, **item)
                    self.assertEqual(requests, [])
                    return
                output = await loop.run({'temperature': 0., 'max_tokens': 64}, **item)
                self.assertEqual(len(requests), len(replies))
                if not native_legacy:
                    self.assertEqual(output.extra_fields['visual_policy_calls'], len(requests))
                    self.assertEqual(output.extra_fields['visual_generated_tokens'], sum(map(len, responses)))
                self.assertEqual(sum(output.response_mask), sum(map(len, responses)))
                self.assertTrue(all(set(request)=={'request_id','prompt_ids','sampling_params','image_data','video_data'} for request in requests))
                worker = AgentLoopWorker.__new__(AgentLoopWorker)
                worker.processor, worker.tokenizer = self.processor, self.tokenizer
                worker.rollout_config = trial.actor_rollout_ref.rollout;worker.reward_loop_worker_handles = None
                internal = await worker._agent_loop_postprocess(output, **item)
                batch = worker._postprocess([internal])
                self.assertEqual(batch.batch['rm_scores'].sum().item(), output.reward_score)
                self.assertEqual(batch.batch['response_mask'].sum().item(), sum(map(len, responses)))
                return output, requests, batch

            old = asyncio.run(run(legacy, legacy_config, [crop, truth], native_legacy=True))
            default = asyncio.run(run(dataset, data_config, [crop, truth]))
            self.assertEqual(old[0].prompt_ids, default[0].prompt_ids)
            self.assertEqual(old[0].response_ids, default[0].response_ids)
            self.assertEqual(old[0].response_mask, default[0].response_mask)
            self.assertEqual(old[0].reward_score, default[0].reward_score)
            for left, right in zip(old[1], default[1], strict=True):
                self.assertEqual(left['prompt_ids'], right['prompt_ids'])
                self.assertEqual(left['sampling_params'], right['sampling_params'])
                self.assertEqual([im.tobytes() for im in left['image_data']], [im.tobytes() for im in right['image_data']])
            self.assertTrue(torch.equal(old[2].batch['position_ids'], default[2].batch['position_ids']))
            asyncio.run(run(dataset, data_config, [truth], wrong_identity=True))

            candidate_path = root / 'candidate.py'
            source = BASE.read_text().replace('USER_PROMPT = "{question}"', 'USER_PROMPT = "Inspect the image. {question}"')
            source = source.replace('return dict(arguments)', "return {'bbox_2d': [0, 0, 128, 128]}")
            source = source.replace('def format_feedback(text):\n    return text', 'def format_feedback(text):\n    return "CROP EVIDENCE: " + text')
            source = source.replace('def parse_answer(text):\n    return text', 'def parse_answer(text):\n    return text.removeprefix("Answer: ")')
            source = source.replace('def nudge(text, assistant_turns):\n    return None',
                'def nudge(text, assistant_turns):\n    return "Inspect the available image and commit A or B." if text == "unsure" else None')
            candidate_path.write_text(source)
            changed_config = deepcopy(data_config);changed_config.visual_harness_path = str(candidate_path)
            changed = make_dataset(changed_config)
            self.assertNotEqual(dataset.dataframe._fingerprint, changed.dataframe._fingerprint)
            output, requests, batch = asyncio.run(run(changed, changed_config, [crop, 'unsure', 'Answer: ' + truth]))
            self.assertEqual(output.reward_score, 1.)
            self.assertEqual(output.extra_fields['visual_final_answer'], 'Answer: ' + truth)
            self.assertEqual(output.extra_fields['visual_committed_answer'], truth)
            self.assertEqual(output.extra_fields['visual_harness_nudges'], 1)
            self.assertEqual(batch.non_tensor_batch['visual_harness_nudges'].tolist(), [1])
            self.assertNotEqual(output.prompt_ids, default[0].prompt_ids)
            self.assertEqual(requests[1]['image_data'][1].size, (128, 128))
            trace = batch.non_tensor_batch['visual_harness_tool_trace'][0]
            self.assertEqual(json.loads(trace[0]['requested_arguments'])['bbox_2d'], [0, 0, 160, 160])
            self.assertEqual(json.loads(trace[0]['executed_arguments'])['bbox_2d'], [0, 0, 128, 128])
            self.assertEqual(trace[0]['returned_images'][0]['size'], (128, 128))
            mmi = batch.non_tensor_batch['multi_modal_inputs'][0]
            image_tokens = batch.batch['input_ids'][0] == self.processor.image_token_id
            self.assertEqual(int(image_tokens.sum()), int(mmi['image_grid_thw'].prod(-1).sum()) // self.processor.image_processor.merge_size**2)
            self.assertEqual(int(batch.batch['response_mask'][0][image_tokens[-1024:]].sum()), 0)
            self.assertIn('CROP EVIDENCE:', self.tokenizer.decode(requests[1]['prompt_ids']))
            self.assertIn('commit A or B', self.tokenizer.decode(requests[2]['prompt_ids']))
            self.assertGreater(output.response_mask.count(0), 0)
            limited, limited_requests, limited_batch = asyncio.run(run(changed, changed_config, ['unsure'], max_turns=1))
            self.assertEqual(limited.reward_score, 0.)
            self.assertEqual(len(limited_requests), 1)
            from verl import DataProto
            merged = DataProto.concat([batch, limited_batch])
            self.assertEqual(merged.non_tensor_batch['visual_harness_nudges'].tolist(), [1, 0])
            self.assertEqual(merged.non_tensor_batch['visual_policy_calls'].tolist(), [3, 1])
            self.assertEqual([len(trace) for trace in merged.non_tensor_batch['visual_harness_tool_trace']], [1, 0])
            candidate_path.write_text(source + '\n# changed after dataset construction\n')
            with self.assertRaisesRegex(ValueError, 'changed after construction'):
                changed[0]
            print(json.dumps({'kind':'native_visual_harness_dispatch_fixture','status':'PASS',
                'default_requests_match_existing_native_loop':True,'executable_hooks_exercised':5,
                'candidate_model_requests_are_scripted':True,'bounded_continuation_checked':True,
                'actual_model_calls':0,'optimizer_steps':0,'test_task_rows_loaded':False}), flush=True)


if __name__ == '__main__':
    unittest.main()
