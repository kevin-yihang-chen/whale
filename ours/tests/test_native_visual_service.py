"""Actual native config/dataset/worker construction; generation is a CPU fixture."""
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


@unittest.skipUnless(AVAILABLE, 'Requires native runtime and visual overlay')
class NativeVisualServiceTests(unittest.TestCase):
    def test_actual_native_constructors_and_forwarded_call_observation(self):
        from ours.native_visual_service import configuration, dataset_for, ROOT
        from ours.visual_native_evaluation import write_pair_parquet
        from ours.native_visual_agent_observation import ObservedVisualAgentWorker, ObservedVisualServerManager
        from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
        from verl.workers.rollout.replica import TokenOutput
        from verl.utils.config import omega_conf_to_dataclass
        from omegaconf import OmegaConf
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            for key in ('HARNESS_PATH', '_HARNESS_SYSTEM_PROMPT', '_HARNESS_USER_PROMPT_TEMPLATE', 'WHALE_VISUAL_HARNESS_PATH'):
                os.environ.pop(key, None)
            root = Path(tmp)
            (root/'requests').mkdir()
            manifest = ROOT/'data/engineering-chart-pairs-v1/manifest.json'
            model = ROOT/'data/models/qwen3.5-4b-common-bf16-v1'
            cfg = configuration(model, manifest, root)
            rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
            self.assertEqual(rollout.engine_kwargs['vllm']['seed'], 42)
            self.assertEqual(rollout.multi_turn.max_assistant_tokens, 1024)
            parquet = root/'pairs.parquet'
            self.assertEqual(write_pair_parquet(manifest, parquet)['examples'], 16)
            plan = {'model': {'path': str(model)}, 'config': OmegaConf.to_container(cfg, resolve=True), 'bounds': {'images': 16}}
            dataset = dataset_for(plan, parquet)
            self.assertEqual(len(dataset), 16)
            # Constructor actually resolves native model/tokenizer/processor and
            # agent YAML; no Ray transport or model tensors are initialized.
            worker = ObservedVisualAgentWorker(cfg, [], None)
            self.assertIsInstance(worker.server_manager, ObservedVisualServerManager)
            self.assertIsNotNone(worker.processor)
            image = Image.new('RGB', (32, 32), 'red')
            sampling = {'temperature': 0., 'max_tokens': 8, 'top_p': 1., 'top_k': -1}
            original_sampling = deepcopy(sampling)
            calls = []
            async def generated(server, request_id, **kwargs):
                calls.append((request_id, kwargs))
                return TokenOutput(token_ids=[1, 2], log_probs=[-.1, -.2], num_preempted=0)
            os.environ['VETO_NATIVE_EVALUATION_OUTPUT'] = str(root)
            with patch.object(AsyncLLMServerManager, 'generate', generated):
                result = asyncio.run(worker.server_manager.generate('fixture', prompt_ids=[3, 4],
                    sampling_params=sampling, image_data=[image]))
            self.assertEqual(result.token_ids, [1, 2])
            self.assertEqual(sampling, original_sampling)
            self.assertIs(calls[0][1]['image_data'][0], image)
            starts = list((root/'requests').glob('*.request.json'))
            self.assertEqual(len(starts), 1)
            request = json.loads(starts[0].read_text())
            self.assertEqual(request['sampling_params'], original_sampling)
            self.assertEqual(request['images'][0]['size'], [32, 32])
            self.assertEqual(len(list((root/'requests').glob('*.response.json'))), 1)


if __name__ == '__main__':
    unittest.main()
