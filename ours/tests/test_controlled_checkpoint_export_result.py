"""Frozen seed42 exporter terminal closure with explicit synthetic provenance."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),'Requires native tensor runtime')
class ControlledExportClosureTests(unittest.TestCase):
    def test_original_export_receipts_close_without_losing_deviation_or_empty_updates(self):
        import torch
        from safetensors.torch import save_file
        from ours import controlled_checkpoint_export as original
        from ours.controlled_checkpoint_export_result import close
        from ours.local_completion import checkpoint_manifest
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        source=Path.cwd()
        for delta in (0.,1e-7):
            with self.subTest(delta=delta),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'base';directory=root/'global_step_2';native=directory/'actor';target=root/'exported'
                base.mkdir();native.mkdir(parents=True);(root/'results').mkdir()
                (root/'ours').symlink_to(source/'ours',target_is_directory=True)
                (root/'upstream').symlink_to(source/'upstream',target_is_directory=True)
                incoming=torch.tensor([1.,1.],dtype=torch.bfloat16);updated=incoming.float()+delta
                save_file({'embedding':incoming},base/'model.safetensors');write_json(base/'config.json',{'model_type':'qwen3_5'})
                torch.save({'embedding':updated,'head':updated.clone()},native/'model_world_size_1_rank_0.pt')
                write_json(native/'fsdp_config.json',{'world_size':1});torch.save({'fixture_only':True},directory/'data.pt')
                digest=file_sha256(native/'model_world_size_1_rank_0.pt');manifest=checkpoint_manifest(base)
                checkpoint={'step':2,'directory':str(directory),'artifact_sha256':{'actor/model_world_size_1_rank_0.pt':digest}}
                training={'condition':'weight_only','seed':42,'initial_manifest':manifest}
                deviation={'fixture_only':True,'reason':'Preserved schedule-preflight deviation'}
                trained={'seed':42,'job_id':'synthetic','optimizer_steps':int(delta>0),'deviation':deviation}
                training_path=root/'training.json';result_path=root/'trained.json';write_json(training_path,training);write_json(result_path,trained)
                bound={n:file_sha256(source/n) for n in original.SOURCES}
                bound[str(directory/'data.pt')]=file_sha256(directory/'data.pt')
                plan={'kind':'controlled_weight_only_export_plan','seed':42,'step':2,'new_model_calls':0,
                    'training_plan':str(training_path),'training_result':str(result_path),'source_job_id':'synthetic',
                    'native_checkpoint_sha256':digest,'base':manifest,'native':str(native),'target':str(target),
                    'transition_report':str(root/'transition.json'),'verified_aliases':{'head':'embedding'},'active_tensors':1,
                    'optimizer_steps_through_checkpoint':int(delta>0),'training_deviation':deviation,
                    'original_preflight_batch_ids_match':False,'source_sha256':bound,'versions':{},
                    'resources':{'time_limit_seconds':1200},
                    'limitations':['Synthetic training/Slurm and serializer; actual complete tensor and inference-asset comparisons.']}
                path=root/'plan.json';write_json(path,plan)
                def serializer(argv,**kwargs):
                    target.mkdir();save_file({'embedding':updated.bfloat16()},target/'model.safetensors')
                    write_json(target/'config.json',{'model_type':'qwen3_5'})
                    write_json(Path(argv[argv.index('--report')+1]),{'fixture_serialization':True})
                try:
                    os.chdir(root)
                    with patch.dict(os.environ,SLURM_JOB_ID='999993',CUDA_VISIBLE_DEVICES=''), \
                         patch('ours.controlled_checkpoint_export.subprocess.run',serializer), \
                         patch('ours.controlled_checkpoint_export.inference_contract',return_value={'fixture_only':True}):
                        original.run(path,plan)
                    slurm=root/'slurm.txt';slurm.write_text(f'JobId=999993 JobState=COMPLETED ExitCode=0:0 RunTime=00:00:30 AllocTRES=cpu=4,gres/gpu=1,gres/gpu:rtx_4090=1 Command={source}/ours/run_controlled_checkpoint_export.sh\n')
                    inputs=(training,trained,{'exported':manifest},checkpoint,{str(directory/'data.pt'):bound[str(directory/'data.pt')]})
                    with patch('ours.controlled_checkpoint_export.completed_inputs',return_value=inputs):
                        result=close(path,slurm)
                        self.assertEqual(result['status'],'COMPLETED_CANONICAL_EXPORT');self.assertEqual(result['seed'],42)
                        self.assertEqual(result['training_deviation'],deviation);self.assertFalse(result['original_preflight_batch_ids_match'])
                        self.assertEqual(result['optimizer_steps_through_checkpoint'],int(delta>0))
                        self.assertEqual(close(Path('plan.json'),Path('slurm.txt')),result)
                        original_slurm=slurm.read_text();slurm.write_text(original_slurm.replace('COMPLETED','FAILED'))
                        with self.assertRaisesRegex(ValueError,'did not complete'):close(path,slurm)
                        slurm.write_text(original_slurm)
                        proof=root/'results/controlled-checkpoint-export-999993/start.json';receipt=json.loads(proof.read_text())
                        receipt['source_job_id']='wrong';write_json(proof,receipt)
                        with self.assertRaisesRegex(ValueError,'execution receipts'):close(path,slurm)
                        receipt['source_job_id']='synthetic';write_json(proof,receipt)
                        state=directory/'data.pt';old=state.read_bytes();state.write_bytes(old+b'changed')
                        with self.assertRaisesRegex(ValueError,'frozen export input'):close(path,slurm)
                finally:os.chdir(source)
        print(json.dumps({'kind':'controlled_export_terminal_cpu_fixture','status':'PASS','deviation_retained':True,
            'empty_and_fp32_only_update_cases':True,'training_and_serializer_provenance_are_synthetic':True,
            'new_model_calls':0,'new_gpu_jobs':0}),flush=True)


if __name__=='__main__':unittest.main()
