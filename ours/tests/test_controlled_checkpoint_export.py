"""Controlled export preserves inference semantics and zero-delta outcomes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'),
                     'Requires the native tensor and Transformers runtime')
class ControlledExportTests(unittest.TestCase):
    def test_source_generation_asset_absence_is_restored_and_decode_drift_is_rejected(self):
        from transformers import AutoConfig,GenerationConfig
        from ours.controlled_checkpoint_export import restore_generation_asset
        from ours.probe_updated_vllm import inference_contract
        from ours.visual_task import file_sha256
        base=Path('data/models/qwen3.5-4b-common-bf16-v1')
        if not base.exists():self.skipTest('Actual common initialization assets are absent')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'target';target.mkdir()
            for path in base.iterdir():
                if path.suffix in {'.json','.txt','.jinja'}:
                    (target/path.name).write_bytes(path.read_bytes())
            config=GenerationConfig.from_model_config(AutoConfig.from_pretrained(base,local_files_only=True))
            config.save_pretrained(target)
            generated=file_sha256(target/'generation_config.json')
            result=restore_generation_asset(base,target,root/'original-generated.json')
            self.assertEqual(result['sha256'],generated)
            self.assertFalse((target/'generation_config.json').exists())
            self.assertTrue(inference_contract(base,target)['generation_config_equal'])
            config.temperature=.375;config.do_sample=True;config.save_pretrained(target)
            with self.assertRaisesRegex(ValueError,'Unreviewed generated inference configuration'):
                restore_generation_asset(base,target,root/'must-not-exist.json')
            self.assertFalse((root/'must-not-exist.json').exists())
            self.assertTrue((target/'generation_config.json').exists())

    def test_completed_export_reports_unchanged_and_sub_bf16_updates_without_requiring_gains(self):
        import os
        import torch
        from safetensors.torch import save_file
        from ours.controlled_checkpoint_export import run
        from ours.local_completion import checkpoint_manifest
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        for delta in (0.,1e-7):
            with self.subTest(delta=delta),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'base';native=root/'native';target=root/'exported'
                base.mkdir();native.mkdir();(root/'results').mkdir()
                incoming=torch.tensor([1.,1.],dtype=torch.bfloat16)
                updated=incoming.float()+delta
                save_file({'embedding':incoming},base/'model.safetensors')
                write_json(base/'config.json',{'model_type':'qwen3_5'})
                torch.save({'embedding':updated,'head':updated.clone()},native/'model_world_size_1_rank_0.pt')
                write_json(native/'fsdp_config.json',{'world_size':1})
                digest=file_sha256(native/'model_world_size_1_rank_0.pt')
                plan={'source_job_id':'fixture','native_checkpoint_sha256':digest,'base':checkpoint_manifest(base),
                    'native':str(native),'target':str(target),'transition_report':str(root/'transition.json'),
                    'verified_aliases':{'head':'embedding'},'active_tensors':1,'step':2,
                    'optimizer_steps_through_checkpoint':int(delta>0),'training_deviation':'fixture',
                    'limitations':['Synthetic serialized CPU tensors; zero inference or optimizer calls.']}
                plan_path=root/'plan.json';write_json(plan_path,plan)
                def serializer(argv,**kwargs):
                    target.mkdir();save_file({'embedding':updated.bfloat16()},target/'model.safetensors')
                    write_json(target/'config.json',{'model_type':'qwen3_5'})
                    write_json(Path(argv[argv.index('--report')+1]),{'fixture_serialization':True})
                old=Path.cwd()
                try:
                    os.chdir(root)
                    with patch.dict(os.environ,SLURM_JOB_ID='fixture',CUDA_VISIBLE_DEVICES=''), \
                         patch('ours.controlled_checkpoint_export.subprocess.run',serializer), \
                         patch('ours.controlled_checkpoint_export.inference_contract',return_value={'fixture_only':True}):
                        run(plan_path,plan)
                finally:os.chdir(old)
                report=json.loads((root/'transition.json').read_text())
                self.assertEqual(report['status'],'PASS')
                self.assertEqual(report['native_fp32']['status'],'CHANGED' if delta else 'UNCHANGED')
                self.assertEqual(report['exported_bf16']['status'],'UNCHANGED')
                self.assertTrue(report['export_exact_native_bf16_cast'])
                self.assertEqual(file_sha256(native/'model_world_size_1_rank_0.pt'),digest)


if __name__=='__main__':unittest.main()
