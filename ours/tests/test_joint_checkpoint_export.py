"""Actual native small-VLM serialization through the joint export entrypoint."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'), 'Requires native model runtime')
class JointExportTests(unittest.TestCase):
    def test_weight_only_result_cannot_be_relabelled_as_joint_export(self):
        from ours.joint_checkpoint_export import completed_inputs
        with tempfile.TemporaryDirectory() as tmp:
            result=Path(tmp)/'result.json'
            result.write_text(json.dumps({'kind':'controlled_weight_only_training_result'}))
            with self.assertRaisesRegex(ValueError,'independently completed joint'):
                completed_inputs(Path(tmp)/'missing-training.json',result)

    def test_actual_canonical_subprocess_full_tensor_comparison_and_terminal_closure(self):
        from ours.tests.canonical_export_fixture import exercise_export
        exercise_export(self,export_module='ours.joint_checkpoint_export',
            cases=[(1,0.,42),(1,1e-7,42),(2,0.,42),(2,1e-7,42)],condition='whale',
            wrapper='ours/run_joint_checkpoint_export.sh')


if __name__=='__main__':unittest.main()
