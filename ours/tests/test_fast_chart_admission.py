from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest

from ours.fast_chart_admission import require_parameter_update
from ours.fast_chart_forecast import allocated_gib,evaluation_projection,include_observed_model_loading


class AdmissionTests(unittest.TestCase):
    def test_large_endpoint_preparation_scales_without_double_counting_V(self):
        baseline=evaluation_projection(360,1,10)
        measured=evaluation_projection(360,1,10,input_preparation_seconds_for_512=60)
        self.assertEqual(baseline['one_V512_gpu_hours'],measured['one_V512_gpu_hours'])
        self.assertEqual(baseline['external_18_conditions_gpu_hours'],measured['external_18_conditions_gpu_hours'])
        self.assertAlmostEqual(measured['final_evaluation_phase_proxy_gpu_hours']-
            baseline['final_evaluation_phase_proxy_gpu_hours'],1.2)
        self.assertAlmostEqual(measured['independent_R_six_1024_pair_passes_gpu_hours']-
            baseline['independent_R_six_1024_pair_passes_gpu_hours'],.3)
        updated=include_observed_model_loading(measured,[400])
        self.assertEqual(updated['input_preparation_seconds_for_512'],60)
        self.assertAlmostEqual(updated['one_T2048_gpu_hours'],(400+2048+10+180)/3600)

    def test_deleted_weight_cannot_be_measured_as_a_small_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weight=root/'model.pt';metadata=root/'config.json'
            weight.write_bytes(b'original model payload');metadata.write_text('{}')
            required=('model.pt','config.json')
            self.assertGreater(allocated_gib(root,required_files=required),0)
            weight.unlink()
            with self.assertRaisesRegex(ValueError,'missing or empty'):
                allocated_gib(root,required_files=required)
            weight.touch()
            with self.assertRaisesRegex(ValueError,'missing or empty'):
                allocated_gib(root,required_files=required)

    def test_missing_directory_is_not_zero_and_hardlinks_are_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'existing real directory'):
                allocated_gib(root/'absent')
            first=root/'model.pt';first.write_bytes(b'payload')
            os.link(first,root/'identical.pt')
            self.assertEqual(allocated_gib(root),first.stat().st_blocks*512/1024**3)

    def test_parameter_cast_or_zero_gradient_cannot_pass_engineering_gate(self):
        execution={'status':'COMPLETE_COMPACT_NATIVE_STAGE','nonzero_gradient_events':2}
        changed={'status':'CHANGED','changed_elements':8}
        transition={'native_fp32':changed,'exported_bf16':changed,'export_exact_native_bf16_cast':True}
        require_parameter_update(execution,transition,stage=1)
        missing=deepcopy(transition);missing['exported_bf16']={'status':'UNCHANGED','changed_elements':0}
        with self.assertRaises(ValueError):require_parameter_update(execution,missing,stage=1)
        with self.assertRaises(ValueError):require_parameter_update({**execution,'nonzero_gradient_events':0},transition,stage=1)
        with self.assertRaises(KeyError):require_parameter_update(execution,transition,stage=2)

    def test_endpoint_cost_includes_all_seeds_controls_and_common_h0(self):
        result=evaluation_projection(360,1)
        self.assertAlmostEqual(result['final_evaluation_phase_proxy_gpu_hours'],
            (24*(360+2048)+18*(360+512))/3600)
        self.assertAlmostEqual(result['independent_R_six_1024_pair_passes_gpu_hours'],6*(360+2048)/3600)

    def test_updated_model_loading_is_charged_to_every_final_and_recheck_pass(self):
        baseline=evaluation_projection(250,.8,10)
        baseline['measured_batch_size']=32
        updated=include_observed_model_loading(baseline,[240,300,280])
        self.assertEqual(updated['measured_batch_size'],32)
        self.assertEqual(updated['seconds_per_image'],.8)
        self.assertEqual(updated['overhead_seconds'],10)
        self.assertAlmostEqual(updated['final_evaluation_phase_proxy_gpu_hours']-
            baseline['final_evaluation_phase_proxy_gpu_hours'],42*50/3600)
        self.assertAlmostEqual(updated['independent_R_six_1024_pair_passes_gpu_hours']-
            baseline['independent_R_six_1024_pair_passes_gpu_hours'],6*50/3600)
        self.assertEqual(include_observed_model_loading(baseline,[200]),baseline)


if __name__=='__main__':unittest.main()
