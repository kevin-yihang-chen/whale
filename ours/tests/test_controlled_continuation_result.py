"""Explicit CPU terminal fixtures; archived seed42 data is never relabelled."""
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

from ours.controlled_continuation_result import batch_metrics


class ContinuationMetricsTests(unittest.TestCase):
    def test_actual_two_batch_metrics_keep_empty_second_batch_and_reject_extra_summaries(self):
        root=Path('results');source=root/'controlled-pilot-222801.log'
        if not source.exists():self.skipTest('Archived metrics absent')
        batches=json.loads((root/'controlled-pilot-schedule-deviation-audit-222801.json').read_text())['native_configuration_audit']['batches']
        text=source.read_text();self.assertEqual([b['optimizer_steps'] for b in batch_metrics(text,batches)],[1,0])
        first=next(s for s in text.splitlines() if re.search(r'\bstep:1 - online_rsft/accepted:',s))
        for bad in (text+'\n'+first,text.replace('step:2 - online_rsft/accepted:','step:3 - online_rsft/accepted:')):
            with self.assertRaisesRegex(ValueError,'summaries'):batch_metrics(bad,batches)
        changed=text.replace('actor/grad_norm:','actor/grad_norm:nan ')
        with self.assertRaises(ValueError):batch_metrics(changed,batches)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('omegaconf'),'Requires native saved-state runtime')
class ContinuationCompletionTests(unittest.TestCase):
    def test_both_seed_identities_complete_from_explicit_archived_cpu_fixtures(self):
        from omegaconf import OmegaConf
        from ours.controlled_continuation_result import complete,AUDIT_SOURCES
        from ours.controlled_pilot import CPU_NAME
        from ours.controlled_pilot_continuation import CONTRACT
        from ours.controlled_continuation_export import prepare,check as check_export
        from ours.joint_resume_receipts import CHECKPOINT_FILES
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        actual=Path.cwd();source=actual/'data/native-rsft/controlled-weight-only-seed42-222801'
        if not source.exists():self.skipTest('Archived native states absent')
        old_audit=json.loads((actual/'results/controlled-pilot-schedule-deviation-audit-222801.json').read_text())['native_configuration_audit']
        old_log=(actual/'results/controlled-pilot-222801.log').read_text()
        old_order=json.loads((actual/'results/controlled-whale-seed42-phase1-plan-20260910-v1.json').read_text())['native_dataset_order']
        source_hashes={str(source/f'global_step_{step}'/n):file_sha256(source/f'global_step_{step}'/n)
            for step in (1,2) for n in CHECKPOINT_FILES if 'model_world' not in n}
        for seed in (43,44):
            with self.subTest(seed=seed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);job=f'9999{seed}';name=f'controlled-weight-only-seed{seed}-{job}'
                plan=json.loads((actual/f'results/controlled-weight-only-seed{seed}-plan-20260910-v1.json').read_text())
                # This substitutes a declared fixture schedule, not a valid new trial.
                plan['native_dataset_order']=deepcopy(old_order)
                plan['initialization_report']=str((actual/plan['initialization_report']).resolve())
                plan['source_sha256']={str((actual/n).resolve()):h for n,h in plan['source_sha256'].items()}
                native=root/'native'/name
                for step in (1,2):
                    directory=native/f'global_step_{step}';(directory/'actor/huggingface').mkdir(parents=True)
                    for n in CHECKPOINT_FILES:
                        if 'model_world' in n:(directory/n).write_bytes(b'synthetic small model for provenance fixture only')
                        else:shutil.copyfile(source/f'global_step_{step}'/n,directory/n)
                (native/'latest_checkpointed_iteration.txt').write_text('2\n')
                raw=Path(plan['resolved_config']).read_text();cfg=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
                cfg.trainer.default_local_dir=str(native).replace(name,CPU_NAME)
                frozen=root/'frozen.yaml';frozen.write_text(OmegaConf.to_yaml(cfg,resolve=True));plan['resolved_config']=str(frozen)
                plan_path=root/'plan.json';write_json(plan_path,plan)
                start=root/f'results/controlled-pilot-{job}';start.mkdir(parents=True)
                (start/'effective-config.yaml').write_text(frozen.read_text().replace(CPU_NAME,name))
                write_json(start/'start.json',{'status':'PASS','plan_sha256':file_sha256(plan_path),'run_name':name,'seed':seed,
                    'native_batch_steps':2,'new_model_calls':0,'preflight_order_contract':CONTRACT,
                    'entrypoint_source_sha256':file_sha256(actual/'ours/controlled_pilot_continuation.py')})
                (root/'ours').symlink_to(actual/'ours',target_is_directory=True)
                (root/'upstream').symlink_to(actual/'upstream',target_is_directory=True)
                log=root/'training.log';log.write_text(old_log)
                slurm=root/'slurm.txt';slurm.write_text(f'JobId={job} JobState=COMPLETED ExitCode=0:0 RunTime=01:08:18 AllocTRES=cpu=16,gres/gpu=2,gres/gpu:h800=2 StdOut={log} Command={actual}/ours/run_controlled_pilot_continuation.sh\n')
                audit=deepcopy(old_audit);audit.update(kind='controlled_pilot_batch_audit',status='PASS',job_id=job,seed=seed,
                    condition='weight_only',plan_sha256=file_sha256(plan_path),
                    audit_source_sha256={n:file_sha256(actual/n) for n in AUDIT_SOURCES},
                    artifact_sha256={f'results/controlled-pilot-{job}/start.json':file_sha256(start/'start.json')},
                    fixture_note='Historical trajectory summaries and saved data states; synthetic identity/model/Slurm, no new training.')
                audit_path=root/'audit.json';write_json(audit_path,audit)
                try:
                    os.chdir(root)
                    with patch('ours.controlled_continuation_result.check',return_value=plan):
                        result=complete(plan_path,audit_path,slurm,log)
                        self.assertEqual(result['status'],'COMPLETED_NATIVE_CONTINUATION');self.assertEqual(result['seed'],seed)
                        self.assertEqual(result['optimizer_steps'],1);self.assertEqual([b['optimizer_steps'] for b in result['batches']],[1,0])
                        self.assertEqual([s['scheduler_last_epoch'] for s in result['saved_states']],[1,1])
                        self.assertTrue(result['original_preflight_batch_ids_match']);self.assertIsNone(result['deviation'])
                        result_path=root/'completed.json';write_json(result_path,result)
                        export_path=root/'export.json';prepare(export_path,plan_path,result_path)
                        with patch('ours.joint_checkpoint_export.checkpoint_manifest',return_value=plan['initial_manifest']):
                            checked=check_export(export_path)
                            self.assertEqual(checked['kind'],'controlled_weight_only_continuation_export_plan')
                            self.assertEqual(checked['condition'],'weight_only');self.assertEqual(checked['seed'],seed)
                            self.assertEqual(checked['step'],2);self.assertEqual(checked['optimizer_steps_through_checkpoint'],1)
                            changed=deepcopy(checked);changed['seed']=42;write_json(export_path,changed)
                            with self.assertRaisesRegex(ValueError,'lineage'):check_export(export_path)
                            write_json(export_path,checked)
                        bad=deepcopy(audit);bad['batches'][1]['trajectory_checks'].pop();write_json(audit_path,bad)
                        with self.assertRaisesRegex(ValueError,'coverage'):complete(plan_path,audit_path,slurm,log)
                        write_json(audit_path,audit)
                        extra=native/'global_step_2/actor/optim_world_size_1_rank_0.pt';extra.write_bytes(b'forbidden')
                        with self.assertRaisesRegex(ValueError,'optimizer-state'):complete(plan_path,audit_path,slurm,log)
                        extra.unlink()
                        text=slurm.read_text();slurm.write_text(text.replace('COMPLETED','FAILED'))
                        with self.assertRaises(ValueError):complete(plan_path,audit_path,slurm,log)
                finally:os.chdir(actual)
        self.assertEqual(source_hashes,{p:file_sha256(Path(p)) for p in source_hashes})
        print(json.dumps({'kind':'continuation_completion_cpu_fixture','status':'PASS','seed_identities_checked':[43,44],
            'historical_two_batch_summaries_and_data_states':True,'model_launch_and_slurm_identity_are_synthetic':True,
            'actual_archives_unchanged':True,'new_model_calls':0,'new_training_runs':0}),flush=True)


if __name__=='__main__':unittest.main()
