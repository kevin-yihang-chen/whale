"""Actual native image/processor/tensor path, with explicitly synthetic replies."""
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE=all(importlib.util.find_spec(name) for name in ('torch','transformers','qwen_vl_utils'))


@unittest.skipUnless(AVAILABLE,'Requires native runtime plus isolated visual dependencies')
class VisualDatasetNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from verl.utils.tokenizer import hf_tokenizer,hf_processor
        import pandas as pd
        from ours.run_visual_smoke import load_inputs
        from ours.visual_task import SYSTEM_PROMPT
        manifest=Path('data/engineering-chart-pairs-v1/manifest.json')
        if not manifest.exists():raise unittest.SkipTest('Existing engineering image manifest required')
        _,pairs,paths=load_inputs(manifest)
        temporary=tempfile.TemporaryDirectory();cls.addClassCleanup(temporary.cleanup)
        cls.path=Path(temporary.name)/'two-images.parquet'
        cls.rows=[{'data_source':'visual_engineering_fixture',
            'prompt':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'<image>\n'+pairs[0].question}],
            'images':[{'bytes':paths[sha].read_bytes()}],
            'reward_model':{'style':'rule','ground_truth':pairs[0].answers[side]},
            'extra_info':{'index':side,'question':pairs[0].question}} for side,sha in enumerate(pairs[0].images_sha256)]
        pd.DataFrame(cls.rows).to_parquet(cls.path,index=False)
        cls.model='data/models/qwen3.5-4b-common-bf16-v1'
        cls.tokenizer=hf_tokenizer(cls.model,local_files_only=True);cls.processor=hf_processor(cls.model,local_files_only=True)
        assert cls.processor is not None

    def config(self,root,limit=4096):
        from omegaconf import OmegaConf
        return OmegaConf.create({'prompt_key':'prompt','image_key':'images','image_patch_size':self.processor.image_processor.patch_size,
            'max_prompt_length':limit,'filter_overlong_prompts':True,'filter_overlong_prompts_workers':1,
            'cache_dir':str(root/'cache'),'apply_chat_template_kwargs':{'enable_thinking':False}})

    def tensors(self,messages):
        from qwen_vl_utils import process_vision_info
        images,videos=process_vision_info(messages,image_patch_size=self.processor.image_processor.patch_size)
        text=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
        return self.processor(text=[text],images=images,return_tensors='pt'),images,text

    def test_prompt_override_retains_images_and_filter_uses_full_visual_length(self):
        import torch
        from datasets import disable_caching,enable_caching
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from ours.visual_evidence_dataset import VisualEvidenceDataset
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ):
            for key in ('HARNESS_PATH','_HARNESS_SYSTEM_PROMPT','_HARNESS_USER_PROMPT_TEMPLATE'):os.environ.pop(key,None)
            root=Path(tmp);config=self.config(root);pixel_sets=[];fingerprints=[]
            for active in (False,True):
                if active:os.environ['_HARNESS_USER_PROMPT_TEMPLATE']='Use the visible evidence. {question}'
                fixed=VisualEvidenceDataset(str(self.path),self.tokenizer,config,self.processor);self.assertEqual(len(fixed),2)
                fingerprints.append(fixed.dataframe._fingerprint)
                images=[]
                for i in range(2):
                    item=fixed[i];inputs,media,text=self.tensors(item['raw_prompt'])
                    self.assertEqual(len(media),1);self.assertNotIn('images',item)
                    image_tokens=int((inputs['input_ids']==self.processor.image_token_id).sum())
                    expected=int(inputs['image_grid_thw'].prod())//self.processor.image_processor.merge_size**2
                    self.assertEqual(image_tokens,expected);self.assertGreater(image_tokens,1)
                    images.append(inputs['pixel_values'])
                self.assertFalse(torch.equal(images[0],images[1]));pixel_sets.append(images)
            self.assertNotEqual(fingerprints[0],fingerprints[1])
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(*pixel_sets,strict=True)))
            full,_,_=self.tensors(fixed[0]['raw_prompt']);limit=full['input_ids'].shape[1]-1
            disable_caching()
            try:
                # With the original hook, both rows become text-only and slip
                # through the same limit. Fixed filtering keeps the real image cost.
                original=RLHFDataset(str(self.path),self.tokenizer,self.config(root,limit),self.processor)
                self.assertEqual(len(original),2)
                self.assertTrue(all(isinstance(original[i]['raw_prompt'][-1]['content'],str) for i in range(2)))
                filtered=VisualEvidenceDataset(str(self.path),self.tokenizer,self.config(root,limit),self.processor)
                self.assertEqual(len(filtered),0)
            finally:enable_caching()
            # The shared adapter also preserves the no-image control.
            import pandas as pd
            controls=deepcopy(self.rows)
            for row in controls:
                row['images']=[];row['prompt'][-1]['content']=row['prompt'][-1]['content'].replace('<image>\n','')
            control_path=root/'no-image.parquet';pd.DataFrame(controls).to_parquet(control_path,index=False)
            control=VisualEvidenceDataset(str(control_path),self.tokenizer,config,self.processor)
            self.assertEqual(len(control),2)
            encoded=[]
            for i in range(2):
                inputs,media,_=self.tensors(control[i]['raw_prompt'])
                self.assertFalse(media);self.assertNotIn('pixel_values',inputs)
                self.assertEqual(int((inputs['input_ids']==self.processor.image_token_id).sum()),0)
                encoded.append(inputs['input_ids'])
            self.assertTrue(torch.equal(*encoded))
            os.environ['_HARNESS_USER_PROMPT_TEMPLATE']='Changed after construction: {question}'
            with self.assertRaisesRegex(ValueError,'changed after dataset construction'):fixed[0]
            print(json.dumps({'kind':'native_visual_prompt_and_filter_fixture','status':'PASS','engineering_images':2,
                'native_text_hook_drops_image':True,'fixed_full_image_tokens':int((full['input_ids']==self.processor.image_token_id).sum()),
                'fixed_full_prompt_tokens':full['input_ids'].shape[1],'harness_cache_identities_differ':True,'new_model_calls':0}),flush=True)

    def test_native_single_turn_and_training_postprocess_keep_pixels_and_masks(self):
        import torch
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
        from verl.workers.rollout.replica import TokenOutput
        from ours.visual_evidence_dataset import VisualEvidenceDataset
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ):
            for key in ('HARNESS_PATH','_HARNESS_SYSTEM_PROMPT'):os.environ.pop(key,None)
            os.environ['_HARNESS_USER_PROMPT_TEMPLATE']='Use the visible evidence. {question}'
            config=self.config(Path(tmp));dataset=VisualEvidenceDataset(str(self.path),self.tokenizer,config,self.processor)
            response=self.tokenizer.encode('fixture',add_special_tokens=False)+[self.tokenizer.eos_token_id]
            requests=[]
            class FixtureServer:
                async def generate(self,**kwargs):
                    requests.append(kwargs)
                    return TokenOutput(token_ids=response,log_probs=[-1.]*len(response),num_preempted=0,
                        extra_fields={'synthetic_response_fixture':True})
            async def replay():
                outputs=[]
                for i in range(2):
                    item=dataset[i];loop=SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
                    loop.processor=self.processor;loop.tokenizer=self.tokenizer;loop.dataset_cls=VisualEvidenceDataset
                    loop.data_config=config;loop.apply_chat_template_kwargs={'enable_thinking':False}
                    loop.loop=asyncio.get_running_loop();loop.server_manager=FixtureServer();loop.response_length=8;loop.prompt_length=4096
                    output=await loop.run({'temperature':0.},**item)
                    expected,images,_=self.tensors(item['raw_prompt'])
                    self.assertEqual(output.prompt_ids,expected['input_ids'][0].tolist())
                    self.assertEqual(len(output.multi_modal_data['images']),1)
                    output.reward_score=0. # Synthetic value only to bypass a reward service.
                    worker=AgentLoopWorker.__new__(AgentLoopWorker);worker.processor=self.processor;worker.tokenizer=self.tokenizer
                    from omegaconf import OmegaConf
                    worker.rollout_config=OmegaConf.create({'prompt_length':len(output.prompt_ids)+3,'response_length':8})
                    worker.reward_loop_worker_handles=None
                    internal=await worker._agent_loop_postprocess(output,**item)
                    self.assertEqual(internal.response_mask.sum().item(),len(response))
                    self.assertEqual(internal.position_ids.shape[1],4)
                    self.assertTrue(torch.equal(internal.multi_modal_inputs['pixel_values'],expected['pixel_values']))
                    self.assertTrue(torch.equal(internal.multi_modal_inputs['image_grid_thw'],expected['image_grid_thw']))
                    batch=worker._postprocess([internal])
                    self.assertIn('multi_modal_inputs',batch.non_tensor_batch)
                    self.assertEqual(batch.batch['response_mask'][0,:len(response)].tolist(),[1]*len(response))
                    self.assertEqual(batch.batch['response_mask'][0,len(response):].sum().item(),0)
                    outputs.append({'image_tokens':int((internal.prompt_ids==self.processor.image_token_id).sum()),
                        'loss_mask_tokens':int(internal.response_mask.sum()),'position_shape':list(internal.position_ids.shape),
                        'pixel_shape':list(internal.multi_modal_inputs['pixel_values'].shape)})
                return outputs
            outputs=asyncio.run(replay())
            self.assertEqual(len(requests),2)
            self.assertEqual(requests[0]['prompt_ids'],requests[1]['prompt_ids'])
            self.assertEqual(set(requests[0]),{'request_id','prompt_ids','sampling_params','image_data','video_data'})
            self.assertNotEqual(requests[0]['image_data'][0].tobytes(),requests[1]['image_data'][0].tobytes())
            print(json.dumps({'kind':'native_visual_single_turn_and_tensor_fixture','status':'PASS','outputs':outputs,
                'reply_and_reward_are_synthetic':True,'model_weights_loaded':False,'new_model_calls':0,'new_gpu_jobs':0,
                'scope':'Native dataset, processor, loop request and masked tensor collation; no actor forward/backward or task accuracy.'}),flush=True)


if __name__=='__main__':unittest.main()
