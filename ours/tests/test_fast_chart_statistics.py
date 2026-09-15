import unittest
from ours.chartqa_open_answer import score_response,relaxed_correct
from ours.fast_chart_statistics import matched_source_interval,candidate_verdict


class StatisticsTests(unittest.TestCase):
    def test_external_open_answers_numeric_boundary_and_text(self):
        self.assertTrue(relaxed_correct('105','100'))
        self.assertFalse(relaxed_correct('105.01','100'))
        self.assertTrue(relaxed_correct('0.5','50%'))
        self.assertTrue(relaxed_correct('0.0','0'))
        self.assertFalse(relaxed_correct('0.000001','0'))
        self.assertTrue(score_response('Reasoning\nFinal answer: United States','United States')['correct'])
        self.assertFalse(relaxed_correct('United State','United States'))

    def test_group_bootstrap_and_complete_seed_coverage(self):
        before={s:{str(i):{'source_id':str(i//2),'paired':0.} for i in range(8)} for s in (42,43,44)}
        after={s:{str(i):{'source_id':str(i//2),'paired':1.} for i in range(8)} for s in (42,43,44)}
        r=matched_source_interval(before,after,replicates=100)
        self.assertEqual(r['sources'],4)
        self.assertEqual(r['source_cluster_95_percentile_interval'],[1.,1.])
        del after[44]['0']
        with self.assertRaises(ValueError):matched_source_interval(before,after,replicates=100)

    def test_no_positive_claim_from_zero_interval_or_equivalent_selection(self):
        pair={'per_seed_difference':{42:.02,43:.02,44:0},'mean_difference':.04/3,'source_cluster_95_percentile_interval':[0,.03]}
        ordinary={'mean_difference':0}
        baseline={'mean_difference':.01}
        self.assertEqual(candidate_verdict(pair,ordinary,baseline,distinct_choices=1)['status'],'INSUFFICIENT_METHOD_EVIDENCE')
        pair['source_cluster_95_percentile_interval'][0]=.001
        self.assertEqual(candidate_verdict(pair,ordinary,baseline,distinct_choices=0)['status'],'INSUFFICIENT_METHOD_EVIDENCE')


if __name__=='__main__':unittest.main()
