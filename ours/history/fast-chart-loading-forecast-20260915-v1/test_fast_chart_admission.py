from copy import deepcopy
import unittest

from ours.fast_chart_admission import require_parameter_update
from ours.fast_chart_forecast import evaluation_projection


class AdmissionTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
