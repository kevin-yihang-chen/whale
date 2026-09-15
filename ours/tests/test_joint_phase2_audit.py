"""Replay archived real replies through a synthetic second-phase provenance view.

Only copied fixture provenance is remapped. This is not a joint-condition run,
and the historical weight-only source archive is never modified or reclassified.
"""
import asyncio
from copy import deepcopy
import gzip
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native runtime and the archived second weight-only batch')
class JointPhaseAuditTests(unittest.TestCase):
    def test_all64_archived_second_batch_replies_and_native_restore_receipts(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from omegaconf import OmegaConf
        from ours.audit_joint_training_phase2 import audit,runtime_configuration
        from ours.joint_training_phase2 import run_name
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        source_root=Path.cwd()
        source=source_root/'data/native-rsft/controlled-weight-only-seed42-222801/audit'
        plan_path=source_root/'results/controlled-whale-seed42-phase1-plan-20260910-v1.json'
        if not source.exists() or not plan_path.exists():self.skipTest('Archived batch and prepared second-phase plan are required')
        base_plan=json.loads(plan_path.read_text())
        old_audit=json.loads((source_root/'results/controlled-pilot-schedule-deviation-audit-222801.json').read_text())
        batch_audit=old_audit['native_configuration_audit']['batches'][1]
        trace_ids={key for row in batch_audit['trajectory_checks'] for key in row['request_trace_ids']}
        original_digest=file_sha256(source/'batch-2.jsonl.gz')
        with tempfile.TemporaryDirectory(prefix='whale-joint-audit-fixture-') as tmp:
            root=Path(tmp);plan=deepcopy(base_plan);job='999901'
            from ours.tests.joint_phase2_fixture import incoming_checkpoint,restore_receipts
            plan.update(kind='controlled_joint_phase2_plan',phase=2,
                resume_checkpoint=incoming_checkpoint(root,source.parent/'global_step_1'),harness_sha256=file_sha256(Path(plan['harness'])))
            search=root/'synthetic-search-plan.json';write_json(search,{'joint_handoff':{'optimizer_steps':1}})
            plan['incoming_search']={'search_plan':str(search),'fixture_only':True}
            plan['source_sha256']['ours/native_resume_observation.py']=file_sha256(source_root/'ours/native_resume_observation.py')
            plan['initialization_report']=str((source_root/plan['initialization_report']).resolve())
            plan['dataset']=str((source_root/plan['dataset']).resolve())
            directory=root/'native'/run_name(plan,job)/'audit';directory.mkdir(parents=True)
            raw=Path(plan['resolved_config']).read_text();config=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
            config.trainer.default_local_dir=str(directory.parent)
            config.trainer.total_training_steps=2;config.trainer.resume_mode='resume_path'
            config.trainer.resume_from_path=plan['resume_checkpoint']['directory'];config.trainer.del_local_ckpt_after_load=False
            config_path=root/'fixture-config.yaml';config_path.write_text(OmegaConf.to_yaml(config,resolve=True))
            plan['resolved_config']=str(config_path)
            fixture_plan=root/'fixture-plan.json';write_json(fixture_plan,plan)
            _,context=runtime_configuration(plan,job)
            with gzip.open(source/'batch-2.jsonl.gz','rt') as incoming,gzip.open(directory/'batch-2.jsonl.gz','wt') as out:
                header=json.loads(next(incoming));header['context']=context
                out.write(json.dumps(header)+'\n')
                for line in incoming:out.write(line)
            receipt=json.loads((source/'batch-2.receipt.json').read_text())
            receipt['sha256']=file_sha256(directory/'batch-2.jsonl.gz')
            write_json(directory/'batch-2.receipt.json',receipt)
            copied=[]
            for path in sorted(source.glob('requests-*.jsonl')):
                events=[]
                for line in path.read_text().splitlines():
                    event=json.loads(line)
                    if event['trace_id'] in trace_ids:
                        event['context']=context;events.append(event)
                if events:
                    (directory/path.name).write_text(''.join(json.dumps(e)+'\n' for e in events));copied.extend(events)
            start=root/f'results/controlled-joint-phase2-{job}';start.mkdir(parents=True)
            write_json(start/'start.json',{'status':'PASS','plan_sha256':file_sha256(fixture_plan),
                'condition':'whale','seed':42,'phase':2,'run_name':run_name(plan,job),
                'entrypoint_source_sha256':file_sha256(source_root/'ours/joint_training_phase2.py'),
                'resume_checkpoint':plan['resume_checkpoint'],'harness_sha256':plan['harness_sha256']})
            (start/'effective-config.yaml').write_text(config_path.read_text().replace(plan['cpu_name'],run_name(plan,job)))
            actor,loader=restore_receipts(plan,fixture_plan,job,start)
            (root/'ours').symlink_to(source_root/'ours',target_is_directory=True)
            (root/'upstream').symlink_to(source_root/'upstream',target_is_directory=True)
            try:
                os.chdir(root)
                with patch('ours.audit_joint_training_phase2.check',return_value=plan):
                    report=asyncio.run(audit(fixture_plan,directory))
                    self.assertEqual(report['trajectories'],64);self.assertEqual(report['accepted'],batch_audit['accepted'])
                    self.assertEqual(report['accepted_loss_tokens'],batch_audit['accepted_loss_tokens'])
                    self.assertEqual(report['request_accounting']['recorded_starts'],len(trace_ids))
                    self.assertEqual(report['new_model_calls'],0)
                    self.assertEqual(report['restoration']['status'],'PASS_RESTORED_NATIVE_STATE_RECEIPTS')
                    from ours.joint_resume_receipts import CHECKPOINT_FILES
                    from ours.joint_training_phase2_result import complete
                    output=directory.parent/'global_step_2';(output/'actor/huggingface').mkdir(parents=True)
                    for n in CHECKPOINT_FILES:
                        if n!='actor/model_world_size_1_rank_0.pt':shutil.copyfile(source.parent/'global_step_2'/n,output/n)
                    shutil.copyfile(Path(plan['resume_checkpoint']['directory'])/'actor/model_world_size_1_rank_0.pt',
                                    output/'actor/model_world_size_1_rank_0.pt')
                    (directory.parent/'latest_checkpointed_iteration.txt').write_text('2\n')
                    audit_path=root/'completed-audit.json';write_json(audit_path,report)
                    old_log=(source_root/'results/controlled-pilot-222801.log').read_text()
                    line=next(line for line in old_log.splitlines() if re.search(r'\bstep:2 - online_rsft/accepted:',line))
                    log=root/'training.log';log.write_text(line+'\n')
                    slurm=root/'slurm.txt';slurm.write_text(f'JobId={job} JobState=COMPLETED ExitCode=0:0 RunTime=00:30:00 AllocTRES=cpu=16,mem=160G,gres/gpu=2,gres/gpu:h800=2 StdOut={log} Command={source_root}/ours/run_joint_training_phase2.sh\n')
                    with patch('ours.joint_training_phase2_result.check',return_value=plan):
                        done=complete(fixture_plan,audit_path,slurm,log)
                        self.assertEqual(done['status'],'COMPLETED_NATIVE_PHASE')
                        self.assertEqual(done['optimizer_steps'],0);self.assertEqual(done['total_joint_optimizer_steps'],1)
                        self.assertEqual(done['checkpoints'][0]['step'],2)
                        self.assertEqual(done['saved_state']['sampler']['sampler_yielded'],16)
                        self.assertEqual(done['saved_state']['incoming_scheduler_last_epoch'],1)
                        self.assertEqual(done['saved_state']['scheduler_last_epoch'],1)
                        from ours.joint_checkpoint_export import completed_inputs,prepare
                        completed_path=root/'completed-phase2.json';write_json(completed_path,done)
                        self.assertEqual(completed_inputs(fixture_plan,completed_path)[1],done)
                        export_path=root/'phase2-export-plan.json';export=prepare(export_path,fixture_plan,completed_path)
                        self.assertEqual(export['kind'],'controlled_joint_phase2_export_plan');self.assertEqual(export['step'],2)
                        self.assertEqual(export['optimizer_steps_through_checkpoint'],1)
                        self.assertEqual(export['resume_checkpoint'],done['checkpoints'][0])
                        self.assertEqual(Path(export['target']).name,'hf-step-2-canonical')

                        bad=deepcopy(report);bad['trajectory_checks']=bad['trajectory_checks'][:-1];write_json(audit_path,bad)
                        with self.assertRaisesRegex(ValueError,'trajectory audit coverage'):complete(fixture_plan,audit_path,slurm,log)
                        write_json(audit_path,report)
                        log.write_text(line+'\n'+line+'\n')
                        with self.assertRaisesRegex(ValueError,'phase summaries'):complete(fixture_plan,audit_path,slurm,log)
                        log.write_text(line+'\n')
                        (directory.parent/'global_step_1').mkdir()
                        with self.assertRaisesRegex(ValueError,'checkpoints'):complete(fixture_plan,audit_path,slurm,log)
                        (directory.parent/'global_step_1').rmdir()
                        raw_slurm=slurm.read_text();slurm.write_text(raw_slurm.replace('COMPLETED','FAILED'))
                        with self.assertRaises(ValueError):complete(fixture_plan,audit_path,slurm,log)
                        slurm.write_text(raw_slurm)

                    (directory/'batch-1.jsonl.gz').write_bytes(b'extra batch forbidden')
                    with self.assertRaisesRegex(ValueError,'Missing or extra sampled batch'):
                        asyncio.run(audit(fixture_plan,directory))
                    (directory/'batch-1.jsonl.gz').unlink()
                    changed=deepcopy(actor);changed['device']='cpu';write_json(start/'resume-actor-rank-0.json',changed)
                    with self.assertRaisesRegex(ValueError,'GPU actor restoration'):asyncio.run(audit(fixture_plan,directory))
            finally:os.chdir(source_root)
            self.assertEqual(file_sha256(source/'batch-2.jsonl.gz'),original_digest)
            print(json.dumps({'kind':'joint_phase2_archived_reply_fixture','status':'PASS','archived_rows':64,
                'archived_calls':len(trace_ids),'archived_accepted':report['accepted'],
                'archived_loss_tokens':report['accepted_loss_tokens'],'new_model_calls':0,'new_gpu_jobs':0,
                'provenance_and_gpu_receipts_are_synthetic':True,'actual_gpu_restore_executed':False,'terminal_provenance_is_synthetic':True,
                'actual_empty_batch_metrics_checked':True,'phase2_export_dispatch_and_plan_checked':True,'phase_optimizer_steps':done['optimizer_steps'],
                'combined_fixture_optimizer_steps':done['total_joint_optimizer_steps'],'source_weight_only_archive_unchanged':True}),flush=True)


if __name__=='__main__':unittest.main()
