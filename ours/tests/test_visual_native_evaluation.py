"""Native manager/worker/visual-loop dispatch, with model and Ray RPC fixtures."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'qwen_vl_utils'))


@unittest.skipUnless(AVAILABLE, 'Requires native runtime and isolated visual overlay')
class VisualNativeEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        VisualDatasetNativeTests.setUpClass.__func__(cls)

    def test_native_pair_coverage_order_reward_and_partial_failure(self):
        from ours.evidence import EvaluationIdentity, fingerprint
        from ours.run_visual_smoke import load_inputs
        from ours.visual_harness import BASE
        from ours.visual_native_evaluation import evaluate_pairs, write_pair_parquet, verifier_identity
        from ours.tests.test_visual_evidence_dataset import VisualDatasetNativeTests
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.experimental.agent_loop import agent_loop as native
        from verl.workers.rollout.replica import TokenOutput
        from verl.utils.rollout_trace import RolloutTraceConfig
        from verl import DataProto
        from omegaconf import OmegaConf
        from qwen_vl_utils import process_vision_info

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            for name in ('HARNESS_PATH', '_HARNESS_SYSTEM_PROMPT', '_HARNESS_USER_PROMPT_TEMPLATE', 'WHALE_VISUAL_HARNESS_PATH'):
                os.environ.pop(name, None)
            root = Path(tmp)
            _, pairs, paths = load_inputs(Path('data/engineering-chart-pairs-v1/manifest.json'))
            pair = pairs[0]
            image_files = {}
            for sha in pair.images_sha256:
                (root / f'{sha}.png').write_bytes(paths[sha].read_bytes())
                image_files[sha] = f'{sha}.png'
            manifest = {'role': 'engineering', 'kind': 'native_evaluation_fixture', 'pairs': [asdict(pair)],
                'audit_data_sha256': fingerprint([asdict(pair)]), 'image_files': image_files}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            parquet = root / 'pairs.parquet'
            materialized = write_pair_parquet(manifest_path, parquet)
            self.assertEqual(materialized['examples'], 2)
            data_config = VisualDatasetNativeTests.config(self, root)
            data_config.tool_config_path = str(Path('ours/visual_evidence_tools.yaml').resolve())
            data_config.custom_cls = {'path': 'pkg://ours.visual_harness_dataset', 'name': 'VisualHarnessDataset'}
            data_config.visual_harness_path = str(BASE.resolve())
            dataset = create_rl_dataset(str(parquet), data_config, self.tokenizer, self.processor, is_train=False)
            config = OmegaConf.create({'data': data_config, 'actor_rollout_ref': {'model': {'path': self.model}, 'rollout': {
                'prompt_length': 4096, 'response_length': 1024, 'temperature': .7, 'top_p': .9, 'top_k': -1,
                'calculate_log_probs': True, 'val_kwargs': {'temperature': 0., 'top_p': .88, 'top_k': -1},
                'agent': {'default_agent_loop': 'visual_harness_agent'}, 'multi_turn': {
                    'max_user_turns': 3, 'max_assistant_turns': 3, 'max_assistant_tokens': 128,
                    'max_parallel_calls': 1, 'max_tool_response_length': 512, 'tool_response_truncate_side': 'right',
                    'tool_config_path': data_config.tool_config_path, 'format': 'qwen3_coder', 'interaction_config_path': None}}}})
            # Identity is explicitly synthetic: this CPU test loads no actor weights.
            identity = EvaluationIdentity('0' * 64, '1' * 64, manifest['audit_data_sha256'],
                '2' * 64, verifier_identity(), 'scripted-native-evaluation')
            visible = {}
            for side in (0, 1):
                images, _ = process_vision_info(dataset[side]['raw_prompt'], image_patch_size=self.processor.image_processor.patch_size)
                visible[hashlib.sha256(images[0].tobytes()).hexdigest()] = side
            self.assertEqual(len(visible), 2)
            crop = '<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>[0, 0, 160, 160]</parameter>\n</function>\n</tool_call>'
            requests, request_turns = [], {}
            class ScriptedServer:
                async def generate(inner, **kwargs):
                    requests.append(deepcopy(kwargs))
                    self.assertEqual(set(kwargs), {'request_id', 'prompt_ids', 'sampling_params', 'image_data', 'video_data'})
                    self.assertEqual(kwargs['sampling_params']['temperature'], 0.)
                    self.assertEqual(kwargs['sampling_params']['top_p'], .88)
                    side = visible[hashlib.sha256(kwargs['image_data'][0].tobytes()).hexdigest()]
                    turn = request_turns.get(kwargs['request_id'], 0)
                    request_turns[kwargs['request_id']] = turn + 1
                    reply = crop if side == 1 and turn == 0 else pair.answers[0]
                    tokens = self.tokenizer.encode(reply, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                    return TokenOutput(token_ids=tokens, log_probs=[-1.] * len(tokens), num_preempted=0)
            RolloutTraceConfig.init('visual-fixture', 'native-pair-evaluation', None)
            def manager(*, workers=2, reverse=False, fail_second=False):
                class LocalManager(native.AgentLoopManager):
                    async def generate_sequences(inner, batch):
                        inner.batches += 1
                        if fail_second and inner.batches == 2:
                            raise RuntimeError('Injected interruption before second batch')
                        result = await super().generate_sequences(batch)
                        return result[list(range(len(result) - 1, -1, -1))] if reverse else result
                instance = LocalManager.__new__(LocalManager)
                instance.batches, instance.agent_loop_workers = 0, []
                for _ in range(workers):
                    worker = native.AgentLoopWorker.__new__(native.AgentLoopWorker)
                    worker.config, worker.rollout_config = config, config.actor_rollout_ref.rollout
                    worker.dataset_cls, worker.server_manager = type(dataset), ScriptedServer()
                    worker.tokenizer, worker.processor = self.tokenizer, self.processor
                    worker.reward_loop_worker_handles = None
                    instance.agent_loop_workers.append(SimpleNamespace(generate_sequences=SimpleNamespace(remote=worker.generate_sequences)))
                return instance
            agent = OmegaConf.load('ours/visual_harness_agents.yaml')[0]
            with patch.dict(native._agent_loop_registry, {'visual_harness_agent': agent}):
                receipt, report = asyncio.run(evaluate_pairs(manager(reverse=True), dataset, manifest_path=manifest_path,
                    identity=identity, output=root / 'complete', batch_size=2))
                self.assertEqual(receipt.role, 'engineering')
                self.assertEqual(receipt.correctness, ((1, 0),))
                self.assertEqual(report['paired_accuracy'], 0.)
                self.assertEqual(report['marginal_accuracy'], .5)
                self.assertEqual(report['policy_calls'], len(requests))
                self.assertEqual(len(requests), 3)
                self.assertEqual([r['policy_calls'] for r in report['records']], [1, 2])
                self.assertEqual([r['native_turns'] for r in report['records']], [2, 4])
                restored = DataProto.load_from_disk(root / 'complete/batch-0000.pkl')
                self.assertEqual(restored.batch['rm_scores'].sum(-1).tolist(), [0., 1.])
                self.assertEqual(restored.non_tensor_batch['visual_policy_calls'].tolist(), [2, 1])
                self.assertGreater(report['records'][1]['assistant_and_observation_tokens'], report['records'][1]['generated_tokens'])

                # Corrupt actual archived outputs without new model generation.
                for corruption in ('reward', 'coverage'):
                    class RecordedManager:
                        agent_loop_workers = [None, None]
                        async def generate_sequences(inner, batch):
                            replay = DataProto.load_from_disk(root / 'complete/batch-0000.pkl')
                            if corruption == 'reward':
                                replay.batch['rm_scores'][0, 0] = 2.
                            else:
                                replay.non_tensor_batch['visual_sample_id'][0] = replay.non_tensor_batch['visual_sample_id'][1]
                            return replay
                    with self.assertRaisesRegex(ValueError, corruption):
                        asyncio.run(evaluate_pairs(RecordedManager(), dataset, manifest_path=manifest_path,
                            identity=identity, output=root / f'corrupt-{corruption}', batch_size=2))
                    self.assertTrue((root / f'corrupt-{corruption}/failure.json').exists())
                    self.assertFalse((root / f'corrupt-{corruption}/result.json').exists())
                self.assertEqual(len(requests), 3)

                # Lost pair coverage fails before invoking even the first worker.
                incomplete = deepcopy(dataset)
                incomplete.dataframe = dataset.dataframe.select([0])
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    asyncio.run(evaluate_pairs(manager(workers=1), incomplete, manifest_path=manifest_path,
                        identity=identity, output=root / 'filtered', batch_size=2))
                self.assertEqual(len(requests), 3)
                self.assertFalse((root / 'filtered').exists())
                with self.assertRaisesRegex(ValueError, 'evenly'):
                    asyncio.run(evaluate_pairs(manager(), dataset, manifest_path=manifest_path,
                        identity=identity, output=root / 'uneven', batch_size=1))
                self.assertEqual(len(requests), 3)

                with self.assertRaisesRegex(RuntimeError, 'Injected interruption'):
                    asyncio.run(evaluate_pairs(manager(workers=1, fail_second=True), dataset, manifest_path=manifest_path,
                        identity=identity, output=root / 'partial', batch_size=1))
                self.assertEqual(len(requests), 4)
                self.assertTrue((root / 'partial/batch-0000.pkl').exists())
                failure = json.loads((root / 'partial/failure.json').read_text())
                self.assertEqual(failure['completed_examples'], 1)
                self.assertNotIn('paired_accuracy', failure)
                self.assertFalse((root / 'partial/result.json').exists())
            print(json.dumps({'kind': 'native_visual_pair_evaluation_fixture', 'status': 'PASS',
                'actual_native_manager_worker_and_loop': True, 'ray_rpc_and_model_replies_are_fixtures': True,
                'scripted_requests': len(requests), 'reversed_output_mapping_verified': True,
                'partial_failure_retained_without_aggregate': True, 'actual_model_calls': 0,
                'optimizer_steps': 0, 'test_task_rows_loaded': False}), flush=True)


if __name__ == '__main__':
    unittest.main()
