"""Completion fixtures use archived metrics/state, never reclassify that trial."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ours.joint_training_result import phase_metrics


class PhaseMetricsTests(unittest.TestCase):
    def test_multiple_minibatches_empty_update_and_incomplete_metrics(self):
        first='step:1 - online_rsft/accepted:9 - online_rsft/total:64 - online_rsft/sft_epochs:1 - training/global_step:1 - actor/sft_updates:2 - actor/sft_token_count:99 - actor/sft_loss:0.3 - actor/grad_norm:0 - timing_s/update_actor:5 - timing_s/save_checkpoint:1 - timing_s/update_weights:1'
        empty='step:1 - online_rsft/accepted:0 - online_rsft/total:64 - online_rsft/sft_epochs:1 - training/global_step:1 - online_rsft/skipped_empty_sft:1 - timing_s/save_checkpoint:1 - timing_s/update_weights:1'
        audit={'accepted':9,'accepted_loss_tokens':99}
        self.assertEqual(phase_metrics(first,audit)['optimizer_steps'],2)
        self.assertEqual(phase_metrics(empty,{'accepted':0,'accepted_loss_tokens':0})['optimizer_steps'],0)
        for log in ('',first+'\n'+first,first.replace('step:1','step:2'),first.replace('sft_updates:2','sft_updates:1'),
                    first.replace('actor/grad_norm:0','actor/grad_norm:nan'),first.replace('save_checkpoint:1','save_checkpoint:-1')):
            with self.assertRaises(ValueError):phase_metrics(log,audit)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('omegaconf'), 'Requires native state runtime')
class NativeStateCompletionTests(unittest.TestCase):
    def test_actual_saved_sampler_and_scheduler_with_unchanged_source(self):
        import torch
        from ours.joint_training_result import saved_state
        from ours.visual_task import file_sha256
        plan=json.loads(Path('results/controlled-whale-seed42-phase1-plan-20260910-v1.json').read_text())
        source=Path('data/native-rsft/controlled-weight-only-seed42-222801/global_step_1')
        if not source.exists():self.skipTest('Archived native state required')
        hashes={name:file_sha256(source/name) for name in ('data.pt','actor/extra_state_world_size_1_rank_0.pt')}
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'global_step_1';(directory/'actor/huggingface').mkdir(parents=True)
            for name in ('data.pt','actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','actor/huggingface/config.json'):
                shutil.copyfile(source/name,directory/name)
            (directory/'actor/model_world_size_1_rank_0.pt').write_bytes(b'synthetic model; not parameter evidence')
            self.assertEqual(saved_state(directory,plan,1)['sampler'],plan['native_dataset_order']['checkpoint_sampler_snapshots'][0])
            self.assertEqual(saved_state(directory,plan,2)['scheduler_last_epoch'],1)
            with self.assertRaisesRegex(ValueError,'scheduler'):saved_state(directory,plan,0)
            bad=deepcopy(plan);bad['native_dataset_order']['checkpoint_sampler_snapshots'][0]['sampler_yielded']=16
            with self.assertRaisesRegex(ValueError,'sampler'):saved_state(directory,bad,1)
            (directory/'actor/optim_world_size_1_rank_0.pt').write_bytes(b'forbidden')
            with self.assertRaisesRegex(ValueError,'optimizer-state'):saved_state(directory,plan,1)
        self.assertEqual(hashes,{name:file_sha256(source/name) for name in hashes})

    def test_completed_phase_binds_terminal_log_config_and_state_without_claiming_a_joint_run(self):
        import torch
        from omegaconf import OmegaConf
        from ours.joint_training_result import complete
        from ours.joint_training_phase1 import run_name
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        source_root=Path.cwd();source=source_root/'data/native-rsft/controlled-weight-only-seed42-222801/global_step_1'
        if not source.exists():self.skipTest('Archived native state required')
        old_log=(source_root/'results/controlled-pilot-222801.log').read_text()
        archived=json.loads((source_root/'results/controlled-pilot-schedule-deviation-audit-222801.json').read_text())['native_configuration_audit']['batches'][0]['trajectory_checks']
        line=next(line for line in old_log.splitlines() if re.search(r'\bstep:1 - online_rsft/accepted:',line))
        for condition in ('whale','whale_fst'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);job='999991'
                original_path=source_root/('results/controlled-whale-seed42-phase1-plan-20260910-v1.json' if condition=='whale' else 'results/controlled-whale-fst-seed42-phase1-plan-20260910-v1.json')
                plan=json.loads(original_path.read_text());name=run_name(plan,job)
                plan['initialization_report']=str((source_root/plan['initialization_report']).resolve())
                plan['source_sha256']={str((source_root/n).resolve()):digest for n,digest in plan['source_sha256'].items()}
                native=root/'native'/name;directory=native/'global_step_1';(directory/'actor/huggingface').mkdir(parents=True)
                for n in ('data.pt','actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','actor/huggingface/config.json'):shutil.copyfile(source/n,directory/n)
                (directory/'actor/model_world_size_1_rank_0.pt').write_bytes(b'synthetic serialized model fixture')
                (native/'latest_checkpointed_iteration.txt').write_text('1\n')
                raw=Path(plan['resolved_config']).read_text();cfg=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
                cfg.trainer.default_local_dir=str(native).replace(name,plan['cpu_name'])
                frozen=root/'frozen.yaml';frozen.write_text(OmegaConf.to_yaml(cfg,resolve=True));plan['resolved_config']=str(frozen)
                plan_path=root/'plan.json';write_json(plan_path,plan)
                start=root/f'results/controlled-joint-phase1-{job}';start.mkdir(parents=True)
                (start/'effective-config.yaml').write_text(frozen.read_text().replace(plan['cpu_name'],name))
                write_json(start/'start.json',{'entrypoint_source_sha256':file_sha256(source_root/'ours/joint_training_phase1.py'),
                    'plan_sha256':file_sha256(plan_path),'run_name':name})
                (root/'ours').symlink_to(source_root/'ours',target_is_directory=True)
                (root/'upstream').symlink_to(source_root/'upstream',target_is_directory=True)
                log=root/'training.log';log.write_text(line+'\n')
                slurm=root/'slurm.txt';slurm.write_text(f'JobId={job} JobState=COMPLETED ExitCode=0:0 RunTime=00:30:00 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2 StdOut={log} Command={source_root}/ours/run_joint_training_phase1.sh\n')
                audit={'kind':'controlled_joint_phase1_batch_audit','status':'PASS','phase':1,'condition':condition,
                    'seed':42,'job_id':job,'plan_sha256':file_sha256(plan_path),'trajectories':64,'new_model_calls':0,
                    'puzzle_ids':plan['native_dataset_order']['planned_batch_ids'][0],'accepted':3,'accepted_loss_tokens':22989,
                    'trajectory_checks':archived,
                    'audit_source_sha256':{n:file_sha256(source_root/n) for n in ('ours/audit_joint_training_phase1.py','ours/joint_training_phase1.py',
                        'ours/audit_alternation_training.py','ours/audit_native_training_batch.py','ours/native_training_trace.py')},'artifact_sha256':{f'results/controlled-joint-phase1-{job}/start.json':file_sha256(start/'start.json')},
                    'request_accounting':{'fixture_only':True,'observed_requests_fully_accounted':True,
                        'recorded_starts':sum(r['calls'] for r in archived),'completed':sum(r['calls'] for r in archived),
                        'errors':0,'pending_or_interrupted':0,'partial_tail_records':0,'completed_without_usage':0,'known_completion_tokens':0}}
                audit_path=root/'audit.json';write_json(audit_path,audit)
                try:
                    os.chdir(root)
                    with patch('ours.joint_training_result.check',return_value=plan):
                        report=complete(plan_path,audit_path,slurm,log)
                        self.assertEqual(report['status'],'COMPLETED_NATIVE_PHASE');self.assertEqual(report['condition'],condition)
                        self.assertEqual(report['optimizer_steps'],1);self.assertEqual(report['batches'][0]['loss_tokens'],22989)
                        self.assertFalse(report['saved_state']['gpu_restore_executed_by_finalizer'])
                        bad_audit=deepcopy(audit);bad_audit['trajectory_checks']=bad_audit['trajectory_checks'][:-1]
                        write_json(audit_path,bad_audit)
                        with self.assertRaisesRegex(ValueError,'trajectory audit coverage'):complete(plan_path,audit_path,slurm,log)
                        write_json(audit_path,audit)
                        from ours.joint_checkpoint_export import completed_inputs,prepare,check as check_export
                        result_path=root/'completed.json';write_json(result_path,report)
                        inputs=completed_inputs(plan_path,result_path)
                        self.assertEqual(inputs[1],report)
                        export_plan=root/'export-plan.json';prepare(export_plan,plan_path,result_path)
                        # This fixture checks provenance/config freezing only; its tiny model bytes
                        # are deliberately not passed off as a valid exported 4B checkpoint.
                        with patch('ours.joint_checkpoint_export.checkpoint_manifest',return_value=plan['initial_manifest']):
                            checked=check_export(export_plan)
                            self.assertEqual(checked['condition'],condition)
                            self.assertEqual(checked['resume_checkpoint'],report['checkpoints'][0])
                            with self.assertRaisesRegex(ValueError,'Preserve existing joint export plan'):
                                prepare(export_plan,plan_path,result_path)
                            selected=directory/'actor/model_world_size_1_rank_0.pt';original_bytes=selected.read_bytes()
                            selected.write_bytes(original_bytes+b'tampered')
                            with self.assertRaisesRegex(ValueError,'Changed export input'):check_export(export_plan)
                            selected.write_bytes(original_bytes)
                        changed=slurm.read_text().replace('COMPLETED','FAILED');slurm.write_text(changed)
                        with self.assertRaises(ValueError):complete(plan_path,audit_path,slurm,log)
                        slurm.write_text(changed.replace('FAILED','COMPLETED'))
                        log.write_text(line+'\n'+line+'\n')
                        with self.assertRaisesRegex(ValueError,'summaries'):complete(plan_path,audit_path,slurm,log)
                        log.write_text(line+'\n');(native/'global_step_2').mkdir()
                        with self.assertRaisesRegex(ValueError,'checkpoints'):complete(plan_path,audit_path,slurm,log)
                finally:os.chdir(source_root)
        print(json.dumps({'kind':'joint_completion_fixture','status':'PASS','archived_first_batch_metrics_used':True,
            'archived_sampler_extra_state_used':True,'provenance_and_model_are_synthetic':True,
            'new_training_runs':0,'new_model_calls':0}),flush=True)


if __name__=='__main__':unittest.main()
