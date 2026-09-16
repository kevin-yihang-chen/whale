"""Synthetic H outcomes exercise delivery, grading and private-data exclusion."""
from copy import deepcopy
import unittest

from ours.chart_answer_protocol import encode_truth
from ours.chart_proposal_feedback import summarize, proposal_prompt
from ours.evidence import fingerprint


def fixture():
    manifest = {'role':'H', 'partition':'H', 'pairs':[], 'examples':[], 'task_by_source':{}}
    records = []
    for i in range(128):
        ident = str(i).zfill(3); task = ('comparison','read_value','difference')[i % 3]
        truth = encode_truth('binary' if task == 'comparison' else 'numeric', 'A' if task == 'comparison' else '10')
        answer = 'A' if task == 'comparison' else '10'
        raw = 'unreadable' if i % 4 == 0 else answer
        manifest['examples'].append({'sample_id':ident, 'source_id':ident, 'question':'Synthetic question', 'answer':truth})
        manifest['task_by_source'][ident] = task
        records.append({'sample_id':fingerprint({'role':'H','sample_id':ident}), 'raw_answer':raw, 'committed_answer':raw,
                        'correct':int(raw == answer), 'generated_tokens':5})
    return manifest, {'records':records}


class FeedbackTests(unittest.TestCase):
    def test_counts_delivery_and_no_inferred_causes_or_private_fields(self):
        m, r = fixture(); m['C_labels'] = 'PRIVATE_SENTINEL'; r['private_audit'] = 'PRIVATE_SENTINEL'
        summary = summarize(m,r)
        self.assertEqual(sum(t['correct'] for t in summary['counts_by_task'].values()),96)
        self.assertEqual(sum(t['unparseable'] for t in summary['counts_by_task'].values()),32)
        self.assertEqual(summary['generated_tokens'],640)
        self.assertEqual([len(v) for v in summary['failure_examples_by_task'].values()],[2,2,2])
        prompt = proposal_prompt('Shared contract',summary)
        self.assertIn('BEGIN_OBSERVED_H_FEEDBACK',prompt)
        self.assertIn('Synthetic question',prompt)
        self.assertNotIn('PRIVATE_SENTINEL',prompt)
        r['records'].reverse(); m['examples'].reverse()
        self.assertEqual(summarize(m,r),summary)

    def test_rejects_private_roles_missing_duplicate_and_bad_reward(self):
        for change in ('role','pairs','missing','duplicate','reward','parse','task'):
            with self.subTest(change=change):
                m,r = fixture()
                if change == 'role': m['partition'] = 'C'
                elif change == 'pairs': m['pairs'] = [{}]
                elif change == 'missing': r['records'].pop()
                elif change == 'duplicate': r['records'][-1] = deepcopy(r['records'][0])
                elif change == 'reward': r['records'][0]['correct'] = 1
                elif change == 'parse': r['records'][0]['committed_answer'] = 'A'
                else: m['task_by_source']['000'] = 'unknown'
                with self.assertRaises(ValueError): summarize(m,r)


if __name__ == '__main__': unittest.main()
