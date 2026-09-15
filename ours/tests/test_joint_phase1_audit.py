"""Replay archived real replies through a synthetic first-phase provenance view.

Only copied fixture provenance is remapped. This is not a joint-condition run,
and the historical weight-only source archive is never modified or reclassified.
"""
import asyncio
from copy import deepcopy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires native runtime and the archived first weight-only batch')
class JointPhaseAuditTests(unittest.TestCase):
    def test_all64_archived_replies_replay_and_extra_batch_is_rejected(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        from omegaconf import OmegaConf
        from ours.audit_joint_training_phase1 import audit,runtime_configuration
        from ours.joint_training_phase1 import run_name
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        source_root=Path.cwd()
        source=source_root/'data/native-rsft/controlled-weight-only-seed42-222801/audit'
        plan_path=source_root/'results/controlled-whale-seed42-phase1-plan-20260910-v1.json'
        if not source.exists() or not plan_path.exists():self.skipTest('Archived batch and prepared first-phase plan are required')
        base_plan=json.loads(plan_path.read_text())
        old_audit=json.loads((source_root/'results/controlled-pilot-schedule-deviation-audit-222801.json').read_text())
        batch_audit=old_audit['native_configuration_audit']['batches'][0]
        trace_ids={key for row in batch_audit['trajectory_checks'] for key in row['request_trace_ids']}
        original_digest=file_sha256(source/'batch-1.jsonl.gz')
        with tempfile.TemporaryDirectory(prefix='whale-joint-audit-fixture-') as tmp:
            root=Path(tmp);plan=deepcopy(base_plan);job='999901'
            plan['dataset']=str((source_root/plan['dataset']).resolve())
            directory=root/'native'/run_name(plan,job)/'audit';directory.mkdir(parents=True)
            raw=Path(plan['resolved_config']).read_text();config=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
            config.trainer.default_local_dir=str(directory.parent)
            config_path=root/'fixture-config.yaml';config_path.write_text(OmegaConf.to_yaml(config,resolve=True))
            plan['resolved_config']=str(config_path)
            fixture_plan=root/'fixture-plan.json';write_json(fixture_plan,plan)
            _,context=runtime_configuration(plan,job)
            with gzip.open(source/'batch-1.jsonl.gz','rt') as incoming,gzip.open(directory/'batch-1.jsonl.gz','wt') as out:
                header=json.loads(next(incoming));header['context']=context
                out.write(json.dumps(header)+'\n')
                for line in incoming:out.write(line)
            receipt=json.loads((source/'batch-1.receipt.json').read_text())
            receipt['sha256']=file_sha256(directory/'batch-1.jsonl.gz')
            write_json(directory/'batch-1.receipt.json',receipt)
            copied=[]
            for path in sorted(source.glob('requests-*.jsonl')):
                events=[]
                for line in path.read_text().splitlines():
                    event=json.loads(line)
                    if event['trace_id'] in trace_ids:
                        event['context']=context;events.append(event)
                if events:
                    (directory/path.name).write_text(''.join(json.dumps(e)+'\n' for e in events));copied.extend(events)
            start=root/f'results/controlled-joint-phase1-{job}';start.mkdir(parents=True)
            write_json(start/'start.json',{'status':'PASS','plan_sha256':file_sha256(fixture_plan),
                'condition':'whale','seed':42,'phase':1,'run_name':run_name(plan,job)})
            (root/'ours').symlink_to(source_root/'ours',target_is_directory=True)
            try:
                os.chdir(root)
                with patch('ours.audit_joint_training_phase1.check',return_value=plan):
                    report=asyncio.run(audit(fixture_plan,directory))
                    self.assertEqual(report['trajectories'],64);self.assertEqual(report['accepted'],batch_audit['accepted'])
                    self.assertEqual(report['accepted_loss_tokens'],batch_audit['accepted_loss_tokens'])
                    self.assertEqual(report['request_accounting']['recorded_starts'],len(trace_ids))
                    self.assertEqual(report['new_model_calls'],0)
                    (directory/'batch-2.jsonl.gz').write_bytes(b'extra batch forbidden')
                    with self.assertRaisesRegex(ValueError,'Missing or extra sampled batch'):
                        asyncio.run(audit(fixture_plan,directory))
            finally:os.chdir(source_root)
            self.assertEqual(file_sha256(source/'batch-1.jsonl.gz'),original_digest)
            print(json.dumps({'kind':'joint_phase1_archived_reply_fixture','status':'PASS','archived_rows':64,
                'archived_calls':len(trace_ids),'archived_accepted':report['accepted'],
                'archived_loss_tokens':report['accepted_loss_tokens'],'new_model_calls':0,'new_gpu_jobs':0,
                'provenance_is_synthetic':True,'source_weight_only_archive_unchanged':True}),flush=True)


if __name__=='__main__':unittest.main()
