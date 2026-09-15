from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

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

    def test_different_asset_layout_requires_verified_semantics_and_keeps_raw_identity(self):
        plan={'model':{'path':'export','assets':{'processor_config.json':'new'}},'verifier_sha256':'fixed',
              'config':{'actor_rollout_ref':{'rollout':{'agent':{}}}}}
        setting={'reference_model_path':'base','reference_model_assets':{'preprocessor_config.json':'old'},
                 'verifier_sha256':'fixed','batch_size':16,'agent_workers':2}
        original=deepcopy(plan['model']);proof={'processor_sha256':'equal-loaded-config'}
        with patch.object(policy,'processor_contract',return_value=proof) as check:
            policy.bind(plan,setting)
            check.assert_called_once_with(original,setting)
        self.assertEqual(plan['model'],original)
        self.assertEqual(plan['processor_equivalence'],proof)
        with patch.object(policy,'processor_contract',side_effect=ValueError('different pixels')):
            with self.assertRaisesRegex(ValueError,'different pixels'):policy.bind(plan,setting)

    def test_processor_comparison_rejects_pixels_template_and_bpe_changes(self):
        class Component:
            def __init__(self,value):self.value=value
            def to_dict(self):return self.value
        def processor():
            p=Component({'class':'shared'})
            p.image_processor=Component({'size':128,'normalize':True})
            p.video_processor=Component({'fps':2})
            p.chat_template='same'
            p.tokenizer=SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:'{"merges":["a b"]}'))
            return p
        self.assertIn('tokenizer_backend_sha256',policy.compare_processors(processor(),processor()))
        for mutation in ('pixels','template','merges'):
            a,b=processor(),processor()
            if mutation=='pixels':b.image_processor.value['size']=256
            elif mutation=='template':b.chat_template='changed'
            else:b.tokenizer.backend_tokenizer.to_str=lambda:'{"merges":["b a"]}'
            with self.assertRaises(ValueError):policy.compare_processors(a,b)

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
