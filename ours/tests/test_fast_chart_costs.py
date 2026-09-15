import unittest
from copy import deepcopy
from fractions import Fraction

from ours.fast_chart_costs import attribute,totals


class SharedCostTests(unittest.TestCase):
    def fixture(self):
        resources=[{'id':'prefix','unit':'gpu_hours','settled':'3','held':'0','status':'COMPLETED'},
            {'id':'failure','unit':'gpu_hours','settled':'1','held':'0','status':'FAILED'},
            {'id':'pending-api','unit':'cny','settled':'0','held':'2','status':'UNRESOLVED_PROVIDER_CALL'}]
        report={'resources':resources,'contains_unsettled_resources':True}
        manifest={'kind':'compact_shared_cost_attribution','owners':['whale','veto'],
            'scope':'Fixture study; equal shared prefix, recorded failed branch, unresolved shared API.',
            'assignments':[{'resource_id':'prefix','beneficiaries':['whale','veto'],'reason':'Shared real first stage'},
                {'resource_id':'failure','beneficiaries':['whale'],'reason':'Failed execution is still charged'},
                {'resource_id':'pending-api','beneficiaries':['whale','veto'],'reason':'Shared proposer; charge unknown'}]}
        return report,manifest

    def test_shared_prefix_conserves_total_and_failures_remain_charged(self):
        report,manifest=self.fixture();result=attribute(report,manifest)
        whale=result['attributed']['whale']['totals'];veto=result['attributed']['veto']['totals']
        self.assertEqual(Fraction(whale['gpu_hours']['settled']['exact_fraction']),Fraction(5,2))
        self.assertEqual(Fraction(veto['gpu_hours']['settled']['exact_fraction']),Fraction(3,2))
        self.assertEqual(whale['cny']['unsettled_reserved']['exact_fraction'],'1')
        self.assertEqual(veto['cny']['settled']['exact_fraction'],'0')
        self.assertIn('UNSETTLED',result['status'])

    def test_missing_duplicate_and_empty_resource_assignments_are_rejected(self):
        report,manifest=self.fixture()
        for mutation in ('missing','duplicate','empty','repeat-owner'):
            changed=deepcopy(manifest)
            if mutation=='missing':changed['assignments'].pop()
            if mutation=='duplicate':changed['assignments'].append(changed['assignments'][0])
            if mutation=='empty':changed['assignments'][0]['beneficiaries']=[]
            if mutation=='repeat-owner':changed['assignments'][0]['beneficiaries']=['whale','whale']
            with self.assertRaises(ValueError):attribute(report,changed)


if __name__=='__main__':unittest.main()
