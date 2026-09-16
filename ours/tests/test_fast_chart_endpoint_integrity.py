"""Reject stale receiver weights and changed external inputs before scoring."""
import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class EndpointIntegrityTests(unittest.TestCase):
    def test_endpoint_cannot_relabel_one_seed_as_another(self):
        from ours import fast_chart_endpoint_evaluation as endpoint
        with TemporaryDirectory() as folder:
            root=Path(folder);reference=root/'reference.json';reference.write_text('{"seed":42}')
            with patch.object(endpoint,'verified_calibration',return_value={'status':'COMPLETE_SHARED_CHART_CONFIGURATION'}):
                with self.assertRaisesRegex(ValueError,'another seed'):
                    endpoint.register(root/'registration.json',reference,root/'configuration.json',
                        condition='harness_only',seed=43)
            self.assertFalse((root/'registration.json').exists())

    def test_receiver_probe_finds_update_outside_original_coordinates(self):
        import torch
        from safetensors.torch import save_file
        from ours.fast_chart_training_worker import CompactChartModelWorker

        class Receiver(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.embedding=torch.nn.Embedding.from_pretrained(weights)

            def embed_input_ids(self, ids):
                return self.embedding(ids)

        with TemporaryDirectory() as folder:
            root=Path(folder);base=root/'base';base.mkdir();(root/'state').mkdir()
            actor=root/'checkpoints/global_step_4/actor';actor.mkdir(parents=True)
            initial=torch.zeros(512,4,dtype=torch.bfloat16)
            updated=initial.float();updated[300,2]=1
            save_file({'model.language_model.embed_tokens.weight':initial},str(base/'model.safetensors'))
            torch.save({'model.language_model.embed_tokens.weight':updated},actor/'model_world_size_1_rank_0.pt')
            path=root/'plan.json';path.write_text(json.dumps({'output':str(root),'stage':1,'model':{'path':str(base)},
                'worker_coordinates':[{'token_id':0,'column':0,'expected':0.}]}))
            stale=Receiver(initial);fresh=Receiver(updated.to(torch.bfloat16))
            worker=SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda:stale))
            with patch.dict(os.environ,VETO_COMPACT_TRAINING_PLAN=str(path)):
                with self.assertRaisesRegex(ValueError,'receiver differs'):
                    CompactChartModelWorker.record_compact_weights(worker,4)
                worker.model_runner.get_model=lambda:fresh
                self.assertEqual(CompactChartModelWorker.record_compact_weights(worker,4),'PASS_COMPACT_RECEIVER')
                # The same real receiver check must work with an indexed base.
                (base/'model.safetensors').rename(base/'model-00001-of-00002.safetensors')
                save_file({'unrelated':torch.zeros(1)},str(base/'model-00002-of-00002.safetensors'))
                (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
                    'model.language_model.embed_tokens.weight':'model-00001-of-00002.safetensors',
                    'unrelated':'model-00002-of-00002.safetensors'}}))
                worker.model_runner.get_model=lambda:stale
                with self.assertRaisesRegex(ValueError,'receiver differs'):
                    CompactChartModelWorker.record_compact_weights(worker,4)
                worker.model_runner.get_model=lambda:fresh
                with patch('ours.fast_chart_training_worker.os.getpid',return_value=os.getpid()+1):
                    self.assertEqual(CompactChartModelWorker.record_compact_weights(worker,4),'PASS_COMPACT_RECEIVER')
            record=json.loads(next((root/'state').glob('receiver-*.json')).read_text())
            self.assertTrue(record['coordinates_distinguish_initial_model'])
            self.assertEqual(record['actual']['values'],[1.])

    def test_filtered_or_modified_external_dataset_never_reaches_model(self):
        from ours.fast_chart_endpoint_evaluation import external
        from ours.chart_answer_protocol import SharedAnswerHarness
        harness=object.__new__(SharedAnswerHarness)
        dataset=SimpleNamespace(visual_harness=harness,dataframe=[])
        expected={str(i):{'question':'original','answer':'1','source_image':'x.png'} for i in range(512)}
        with self.assertRaisesRegex(ValueError,'filtering or duplication'):
            asyncio.run(external(None,dataset,{},None,expected))
        dataset.dataframe=[{'visual_sample_id':str(i),'prompt':[]} for i in range(512)]
        with self.assertRaisesRegex(ValueError,'pixels, question or answer'):
            asyncio.run(external(None,dataset,{'manifest':'/tmp/unused-chartqa-manifest.json'},None,expected))

    def test_batch32_external_pass_preserves_all512_answers_and_artifacts(self):
        import numpy as np
        import torch
        from verl import DataProto
        from ours.fast_chart_endpoint_evaluation import external
        from ours.chart_answer_protocol import SharedAnswerHarness
        from ours.chartqa_open_answer import PROTOCOL
        from ours.visual_task import SYSTEM_PROMPT

        class Dataset:
            visual_harness=object.__new__(SharedAnswerHarness)
            def __len__(self):return len(self.dataframe)
            def __getitem__(self,i):return self.dataframe[i]

        class Manager:
            def __init__(self):self.seen=[]
            async def generate_sequences(self,batch):
                ids=batch.non_tensor_batch['visual_sample_id'].tolist();self.seen.extend(ids);size=len(ids)
                self_outer.assertEqual(size,32)
                rows={'visual_sample_id':ids,'visual_final_answer':['1']*size,'visual_committed_answer':['1']*size,
                    'visual_harness_sha256':['fixed']*size,'visual_policy_calls':[1]*size,
                    'visual_generated_tokens':[1]*size,'__num_turns__':[1]*size,'visual_harness_tool_trace':[[] for _ in ids]}
                columns={}
                for key,values in rows.items():
                    column=np.empty(size,dtype=object);column[:]=values;columns[key]=column
                columns.update(rm_scores=torch.ones(size,1),response_mask=torch.ones(size,1,dtype=torch.long),
                               attention_mask=torch.ones(size,1,dtype=torch.long))
                return DataProto.from_single_dict(columns)

        self_outer=self
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'png').mkdir();(root/'png/image.png').write_bytes(b'fixture-image-bytes')
            expected={str(i):{'question':'Read the chart value.','answer':'1','source_image':'image.png',
                'split':'human' if i<256 else 'augmented'} for i in range(512)}
            dataset=Dataset();dataset.dataframe=[{'visual_sample_id':ident,'data_source':PROTOCOL,
                'prompt':[{'role':'system','content':SYSTEM_PROMPT},
                    {'role':'user','content':'<image>\n'+row['question']}],
                'images':[{'bytes':b'fixture-image-bytes'}],'reward_model':{'style':'rule','ground_truth':'1'}}
                for ident,row in expected.items()]
            manager=Manager()
            with patch.object(SharedAnswerHarness,'unchanged',return_value=None,create=True):
                result=asyncio.run(external(manager,dataset,{'manifest':str(root/'plan.json'),
                    'batch_size':32,'harness_sha256':'fixed'},root/'evaluation',expected))
            self.assertEqual(set(manager.seen),set(expected));self.assertEqual(len(manager.seen),512)
            self.assertEqual(result['scores']['human']['examples'],256)
            self.assertEqual(result['scores']['augmented']['examples'],256)
            self.assertEqual(result['policy_calls'],512)
            self.assertEqual(len(result['batch_artifact_sha256']),16)
            self.assertFalse(result['scores']['official_full_benchmark_score'])


if __name__=='__main__':unittest.main()
