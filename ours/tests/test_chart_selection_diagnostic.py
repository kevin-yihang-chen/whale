"""Subset selection and source-matched inference use synthetic records only."""
from copy import deepcopy
import unittest

from ours.chart_selection_diagnostic import contrast, r_subset
from ours.chart_selection_protocol import selection_modes


class SelectionDiagnosticTests(unittest.TestCase):
    def test_fixed_subset_ignores_answers_and_rejects_test_or_repeated_sources(self):
        m={'role':'V','partition':'R','pairs':[{'source_id':str(i).zfill(4),'pair_id':str(i),'answers':['A','B']}
                                             for i in reversed(range(1024))]}
        chosen=r_subset(m)
        self.assertEqual([p['source_id'] for p in chosen],[str(i).zfill(4) for i in range(512)])
        m['pairs'].reverse()
        self.assertEqual(r_subset(m),chosen)
        bad=deepcopy(m); bad['partition']='T'
        with self.assertRaises(ValueError): r_subset(bad)
        bad=deepcopy(m); bad['pairs'][0]['source_id']=bad['pairs'][1]['source_id']
        with self.assertRaises(ValueError): r_subset(bad)

    def test_unchanged_decisions_are_not_positive_evidence(self):
        rows={str(i):{'source_id':str(i//2),'paired':1,'marginal':1.} for i in range(8)}
        result=contrast(rows,rows)
        self.assertEqual(result['source_cluster_95_percentile_interval'],[0.,0.])
        self.assertEqual(result['sources'],4)
        self.assertEqual(result['status'],'INSUFFICIENT_SELECTION_EVIDENCE')
        worse={key:dict(row,paired=0,marginal=.5) for key,row in rows.items()}
        self.assertEqual(contrast(rows,worse)['status'],'SUPPORTS_BOUNDED_CONTINUATION')
        self.assertEqual(contrast(worse,rows)['status'],'INSUFFICIENT_SELECTION_EVIDENCE')
        with self.assertRaises(ValueError): contrast(rows,{})
        bad=deepcopy(rows); bad['0']['source_id']='changed'
        with self.assertRaises(ValueError): contrast(rows,bad)

    def test_old_protocol_not_silently_reinterpreted(self):
        self.assertEqual(dict(selection_modes({}))['veto'],'paired')
        self.assertEqual(dict(selection_modes({'selection_protocol':'evidence_v2'}))['veto'],'paired_rank')
        with self.assertRaises(ValueError): selection_modes({'selection_protocol':'unknown'})


if __name__=='__main__': unittest.main()
