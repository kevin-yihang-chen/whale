"""Scheduler completion may precede NFS result visibility; never rerun the GPU job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ours.fast_chart_calibration_controller import completed_artifact, start_receipt


class ControllerVisibilityTests(unittest.TestCase):
    def test_delayed_and_partial_result_becomes_readable_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=root/'result.json'
            with patch('ours.fast_chart_calibration_controller.time.time',return_value=100):
                self.assertIsNone(completed_artifact(root,result))
            marker=next(root.glob('artifact-visibility-wait-*.json'))
            first=marker.read_bytes()
            result.write_text('{')
            with patch('ours.fast_chart_calibration_controller.time.time',return_value=120):
                self.assertIsNone(completed_artifact(root,result))
            self.assertEqual(marker.read_bytes(),first)
            payload={'status':'COMPLETE','plan_sha256':'actual-result-identity'}
            result.write_text(json.dumps(payload));original=result.read_bytes()
            self.assertEqual(completed_artifact(root,result),payload)
            self.assertEqual(result.read_bytes(),original)

    def test_missing_result_reaches_bounded_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=root/'absent.json'
            with patch('ours.fast_chart_calibration_controller.time.time',return_value=100):
                self.assertIsNone(completed_artifact(root,result))
            with patch('ours.fast_chart_calibration_controller.time.time',return_value=400):
                with self.assertRaisesRegex(RuntimeError,'bounded wait'):
                    completed_artifact(root,result)
            self.assertFalse(result.exists())

    def test_explicit_resume_preserves_start_failure_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h0=root/'h0.json';h0.write_text('{}')
            start=start_receipt(root,h0,None,False);original=start.read_bytes()
            failure=root/'controller-failure.json';failure.write_text('{"error":"visibility lag"}')
            failure_bytes=failure.read_bytes()
            receipt=start_receipt(root,h0,None,True)
            self.assertTrue(json.loads(receipt.read_text())['resume'])
            self.assertEqual(start.read_bytes(),original)
            self.assertEqual(failure.read_bytes(),failure_bytes)
            h0.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'scope'):
                start_receipt(root,h0,None,True)

    def test_resume_cannot_replace_complete_or_unfailed_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h0=root/'h0.json';h0.write_text('{}')
            start_receipt(root,h0,None,False)
            with self.assertRaises(ValueError):start_receipt(root,h0,None,True)
            (root/'controller-failure.json').write_text('{}')
            (root/'result.json').write_text('{"status":"COMPLETE"}')
            with self.assertRaises(ValueError):start_receipt(root,h0,None,True)


if __name__=='__main__':unittest.main()
