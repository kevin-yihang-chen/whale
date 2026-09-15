"""Actual canonical native exporter through the continuous weight-only adapter."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class ContinuationIdentityTests(unittest.TestCase):
    def test_seed42_deviation_and_joint_records_do_not_substitute_for_a_continuation(self):
        from ours.controlled_continuation_export import artifact_kind,completed_inputs,export_context
        from ours import joint_checkpoint_export as backend
        original=(backend.artifact_kind,backend.check,backend.SOURCES,backend.CONDITIONS,backend.EXPORT_WRAPPER)
        with tempfile.TemporaryDirectory() as tmp:
            result=Path(tmp)/'result.json'
            for kind in ('controlled_weight_only_training_result','controlled_joint_phase2_training_result'):
                result.write_text(json.dumps({'kind':kind}))
                with self.assertRaisesRegex(ValueError,'completed native continuation'):
                    completed_inputs(Path(tmp)/'missing.json',result)
        with self.assertRaisesRegex(ValueError,'step2'):artifact_kind(1,'export_plan')
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with export_context():
                self.assertEqual(backend.CONDITIONS,('weight_only',))
                self.assertEqual(backend.artifact_kind(2,'export_plan'),'controlled_weight_only_continuation_export_plan')
                raise RuntimeError('fixture')
        self.assertEqual(original,(backend.artifact_kind,backend.check,backend.SOURCES,backend.CONDITIONS,backend.EXPORT_WRAPPER))


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'),'Requires native model runtime')
class ContinuationExportTests(unittest.TestCase):
    def test_actual_native_export_keeps_both_seed_identities_and_zero_bf16_deltas(self):
        from ours.tests.canonical_export_fixture import exercise_export
        exercise_export(self,export_module='ours.controlled_continuation_export',
            cases=[(2,0.,43),(2,1e-7,44)],condition='weight_only',
            wrapper='ours/run_controlled_continuation_export.sh')


if __name__=='__main__':unittest.main()
