"""Verify actual native tokenization removes image tensors in the blind control."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
from ours.native_visual_service import ROOT, configuration, dataset_for
from ours.visual_native_evaluation import write_pair_parquet


class BlindControlTest(unittest.TestCase):
    def test_native_blind_rows_keep_questions_and_labels_but_have_no_images(self):
        manifest = ROOT/'data/engineering-chart-pairs-v1/manifest.json'
        model = ROOT/'data/models/qwen3.5-4b-common-bf16-v1'
        image_id = json.loads((model/'config.json').read_text())['image_token_id']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root/'pairs.parquet'
            write_pair_parquet(manifest, parquet)
            cfg = configuration(model, manifest, root)
            cfg.data.tool_config_path = None
            cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path = None
            direct = {'model':{'path':str(model)}, 'config':OmegaConf.to_container(cfg,resolve=True),
                      'bounds':{'images':16}}
            visible = dataset_for(direct, parquet, cache_override=str(root/'visible-cache'))
            blind_plan = deepcopy(direct)
            blind_plan['config']['data']['custom_cls'] = {
                'path':'pkg://ours.visual_blind_control', 'name':'BlindVisualHarnessDataset'}
            blind = dataset_for(blind_plan, parquet, cache_override=str(root/'blind-cache'))
            self.assertEqual(list(visible.dataframe),list(blind.dataframe))
            def tokenize(dataset, row):
                images,videos = asyncio.run(dataset.process_vision_info(row['raw_prompt'],
                    image_patch_size=16, config=cfg.data))
                self.assertFalse(videos)
                prompt=dataset.processor.apply_chat_template(row['raw_prompt'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                return dataset.processor(text=[prompt], images=images, return_tensors='pt')
            blind_tokens=[]
            for i in range(16):
                a,b = visible[i],blind[i]
                visible_input,blind_input=tokenize(visible,a),tokenize(blind,b)
                self.assertTrue(bool((visible_input['input_ids']==image_id).any()))
                self.assertFalse(bool((blind_input['input_ids']==image_id).any()))
                self.assertNotIn('pixel_values',blind_input)
                blind_tokens.append(blind_input['input_ids'].tolist())
                self.assertEqual(a['reward_model'],b['reward_model'])
                for message in b['raw_prompt']:
                    if isinstance(message['content'],list):
                        self.assertTrue(all(p['type']=='text' for p in message['content']))
            # Each engineering pair has the same question on both sides. Their
            # blind model inputs must therefore be identical despite opposite labels.
            for i in range(0,16,2):
                self.assertEqual(blind_tokens[i],blind_tokens[i+1])


if __name__ == '__main__':
    unittest.main()
