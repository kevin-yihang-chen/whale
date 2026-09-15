"""Shared actual native small-VLM export fixture; launch identities are synthetic."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


def exercise_export(self,*,export_module,cases,condition,wrapper):
    from ours.training_bootstrap import prepare_worker
    prepare_worker()
    import torch
    from transformers import AutoModelForImageTextToText,PreTrainedTokenizerFast,Qwen3_5Config
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from importlib import import_module
    exporter=import_module(export_module)
    run,close,artifact_kind=exporter.run,exporter.close,exporter.artifact_kind
    from ours.local_completion import checkpoint_manifest
    from ours.native_search import write_json
    from ours.visual_task import file_sha256
    source_root=Path.cwd()
    config=Qwen3_5Config(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        layer_types=['full_attention'],max_position_embeddings=128,tie_word_embeddings=True,eos_token_id=2,
        rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,mrope_section=[1,1,2])),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,
            num_position_embeddings=16,patch_size=2,temporal_patch_size=1,spatial_merge_size=2),
        tie_word_embeddings=True,image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
    config.architectures=['Qwen3_5ForConditionalGeneration'];torch.manual_seed(31)
    model=AutoModelForImageTextToText.from_config(config,dtype=torch.bfloat16)
    with torch.no_grad():model.get_input_embeddings().weight[0,0]=1.
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f't{i}':i for i in range(64)},unk_token='t0')),
        unk_token='t0',bos_token='t1',eos_token='t2',pad_token='t3')
    tokenizer.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
    aliases={'lm_head.weight':'model.language_model.embed_tokens.weight'}
    for step,delta,seed in cases:
        with self.subTest(step=step,delta=delta),tempfile.TemporaryDirectory(prefix='whale-joint-export-fixture-') as tmp:
            root=Path(tmp);base=root/'base';checkpoint=root/f'global_step_{step}';native=checkpoint/'actor';target=root/'exported'
            (native/'huggingface').mkdir(parents=True);(root/'results').mkdir()
            model.save_pretrained(base,save_original_format=False);tokenizer.save_pretrained(base)
            (base/'generation_config.json').unlink(missing_ok=True)
            state={n:t.detach().float().clone() for n,t in model.state_dict().items()}
            state['model.language_model.embed_tokens.weight'][0,0]+=delta
            state['lm_head.weight']=state['model.language_model.embed_tokens.weight'].clone()
            torch.save(state,native/'model_world_size_1_rank_0.pt');write_json(native/'fsdp_config.json',{'world_size':1})
            model.config.save_pretrained(native/'huggingface')
            torch.save({'fixture_only':True},checkpoint/'data.pt');torch.save({'fixture_only':True},native/'extra_state_world_size_1_rank_0.pt')
            artifacts={n:file_sha256(checkpoint/n) for n in ('data.pt','actor/model_world_size_1_rank_0.pt',
                'actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','actor/huggingface/config.json')}
            plan={'source_job_id':'synthetic','condition':condition,'seed':seed,'native_checkpoint_sha256':artifacts['actor/model_world_size_1_rank_0.pt'],
                'base':checkpoint_manifest(base),'native':str(native),'target':str(target),'transition_report':str(root/'transition.json'),
                'verified_aliases':aliases,'active_tensors':len(state)-1,'step':step,'optimizer_steps_through_checkpoint':int(delta>0),
                'resume_checkpoint':{'step':step,'directory':str(checkpoint),'artifact_sha256':artifacts},
                'resources':{'time_limit_seconds':1200},
                'limitations':['Synthetic CPU model and Slurm provenance; actual original native canonical merger subprocess.']}
            if condition=='weight_only':
                training_path=root/'training.json';result_path=root/'completed.json'
                write_json(training_path,{'seed':seed,'condition':condition,'initial_manifest':plan['base']})
                write_json(result_path,{'seed':seed,'condition':condition,'kind':'controlled_weight_only_continuation_result',
                    'status':'COMPLETED_NATIVE_CONTINUATION','job_id':plan['source_job_id'],
                    'original_preflight_batch_ids_match':True,'deviation':None,
                    'checkpoints':[{'step':1},plan['resume_checkpoint']],'optimizer_steps':plan['optimizer_steps_through_checkpoint']})
                plan.update(training_plan=str(training_path),training_result=str(result_path))
            plan_path=root/'plan.json';write_json(plan_path,plan)
            env={'SLURM_JOB_ID':'999992','CUDA_VISIBLE_DEVICES':'','PYTHONPATH':':'.join(str(source_root/p) for p in ('ours/compat','upstream/WHALE/domains/chess_puzzles','.'))}
            try:
                os.chdir(root)
                with patch.dict(os.environ,env):run(plan_path,plan)
                report=json.loads((root/'transition.json').read_text())
                self.assertEqual(report['kind'],artifact_kind(step,'checkpoint_transition'))
                self.assertEqual(report['native_fp32']['status'],'CHANGED' if delta else 'UNCHANGED')
                self.assertEqual(report['exported_bf16']['status'],'UNCHANGED')
                self.assertTrue(report['export_exact_native_bf16_cast'])
                self.assertTrue(report['inference_contract']['generation_config_equal'])
                self.assertEqual(artifacts,{n:file_sha256(checkpoint/n) for n in artifacts})
                slurm=root/'slurm.txt';slurm.write_text(f'JobId=999992 JobState=COMPLETED ExitCode=0:0 RunTime=00:00:30 AllocTRES=cpu=4,gres/gpu=1,gres/gpu:rtx_4090=1 Command={root}/{wrapper}\n')
                with patch('ours.joint_checkpoint_export.check',return_value=plan):
                    result=close(plan_path,slurm)
                    self.assertEqual(result['status'],'COMPLETED_CANONICAL_EXPORT')
                    self.assertEqual(result['kind'],artifact_kind(step,'export_result'))
                    self.assertEqual(result['step'],step)
                    self.assertEqual(result['slurm_terminal_path'],str(slurm.resolve()))
                    self.assertEqual(close(Path('plan.json'),Path('slurm.txt')),result)
                    self.assertAlmostEqual(result['allocation']['gpu_hours'],30/3600)
                    raw=slurm.read_text();slurm.write_text(raw.replace('COMPLETED','RUNNING'))
                    with self.assertRaisesRegex(ValueError,'did not complete'):close(plan_path,slurm)
                    slurm.write_text(raw);report['native_fp32']['tensor_count']-=1;write_json(root/'transition.json',report)
                    with self.assertRaisesRegex(ValueError,'active-parameter'):close(plan_path,slurm)
            finally:os.chdir(source_root)
    print(json.dumps({'kind':'canonical_native_export_fixture','export_module':export_module,'seed_cases':[c[2] for c in cases],'status':'PASS','actual_native_merger_subprocess':True,
        'complete_small_vlm_graph_checked':True,'zero_and_sub_bf16_changes_checked':True,'native_phases_checked':sorted({c[0] for c in cases}),
        'model_and_slurm_provenance_are_synthetic':True,'new_model_calls':0,'new_gpu_jobs':0}),flush=True)
