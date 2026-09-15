from collections import Counter
from copy import deepcopy
import json
import unittest

from ours.fast_chart_backbone_audit import validate_records


class FullBackboneAuditTests(unittest.TestCase):
    def fixture(self):
        expected=Counter();records=[]
        for i in range(3):
            pixels={'actor_pixels':{'sha':str(i)},'consumed_pixels':{'sha':'cast'+str(i)},'grid':{'shape':[1,3]}}
            expected[json.dumps(pixels,sort_keys=True)]+=1
            records.append({'kind':'visual_backbone_input','event':str(i),'job_id':'7',
                'grad_enabled':True,'exact_after_native_dtype_cast':True,**pixels})
            records.append({'kind':'visual_feature_gradient','event':str(i),'job_id':'7',
                'finite':True,'l1':1.,'nonzero_elements':4})
        return expected,records

    def test_complete_images_and_gradients_are_paired(self):
        expected,records=self.fixture();result=validate_records(expected,records,'7')
        self.assertEqual(result['accepted_examples'],3)
        self.assertEqual(result['nonzero_gradient_events'],3)

    def test_missing_wrong_and_duplicate_image_events_fail(self):
        expected,records=self.fixture()
        for mutation in ('missing-gradient','changed-image','duplicate','wrong-job','no-native-cast'):
            changed=deepcopy(records)
            if mutation=='missing-gradient':changed.pop()
            if mutation=='changed-image':changed[0]['actor_pixels']['sha']='wrong'
            if mutation=='duplicate':changed.extend(changed[:2])
            if mutation=='wrong-job':changed[1]['job_id']='8'
            if mutation=='no-native-cast':changed[0]['exact_after_native_dtype_cast']=False
            with self.assertRaises(ValueError):validate_records(expected,changed,'7')

    def test_zero_success_branch_is_reportable(self):
        result=validate_records(Counter(),[],'7')
        self.assertEqual(result['accepted_examples'],0)
        self.assertEqual(result['paired_gradient_events'],0)


if __name__=='__main__':unittest.main()
