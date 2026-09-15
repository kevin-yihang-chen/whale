from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ours import fast_chart_inference_policy as policy


class InferencePolicyTests(unittest.TestCase):
    def test_every_method_receives_same_schedule_without_changing_model_or_limits(self):
        setting={'batch_size':32,'agent_workers':2,'reference_model_assets':{'processor':'same'},
                 'verifier_sha256':'shared'}
        for method in ('weight_only','harness_only','whale','veto','marginal_gate','counterfactual_augmentation'):
            plan={'condition':method,'seed':43,'model':{'assets':{'processor':'same'},'weights_sha256':method},
                  'verifier_sha256':'shared','batch_size':8,'bounds':{'images':2048,'maximum_generation_calls':6144},
                  'config':{'actor_rollout_ref':{'rollout':{'max_num_seqs':8,'agent':{'num_workers':2},
                    'temperature':0.,'max_assistant_tokens':1024}},'data':{'harness':method}}}
            before=deepcopy(plan)
            policy.bind(plan,setting)
            self.assertEqual(plan['batch_size'],32)
            self.assertEqual(plan['config']['actor_rollout_ref']['rollout']['max_num_seqs'],32)
            plan['batch_size']=8;plan['config']['actor_rollout_ref']['rollout']['max_num_seqs']=8
            self.assertEqual(plan,before)
        plan['verifier_sha256']='different'
        with self.assertRaisesRegex(ValueError,'shared scoring'):
            policy.bind(plan,setting)

    def test_incomplete_measurement_cannot_create_policy(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);result=root/'result.json';suite=root/'plan.json'
            result.write_text('{"status":"INCOMPLETE"}');suite.write_text('{}')
            with self.assertRaisesRegex(ValueError,'complete matching'):
                policy.verify_measurement(result,suite)

    def test_policy_selection_is_closed_after_independent_submission(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);allocation=root/'allocations/endpoint';allocation.mkdir(parents=True)
            plan=root/'plan.json';plan.write_text('{"kind":"compact_registered_endpoint_evaluation"}')
            (allocation/'submission.json').write_text(json.dumps({'plan':str(plan)}))
            with patch.object(policy,'OUTPUT',root),patch.object(policy,'POLICY',root/'policy.json'):
                with self.assertRaisesRegex(ValueError,'after independent'):
                    policy.freeze(root/'unused-result.json',root/'unused-suite.json')
                self.assertFalse((root/'policy.json').exists())


if __name__=='__main__':unittest.main()
