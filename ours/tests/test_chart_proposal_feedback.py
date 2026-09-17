"""Synthetic H outcomes exercise delivery, grading and private-data exclusion."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ours.chart_answer_protocol import encode_truth
from ours.chart_proposal_feedback import summarize, summarize_pairs, proposal_prompt, pair_feedback_receipt
from ours.evidence import fingerprint
from ours.visual_task import file_sha256


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

    def test_pair_feedback_reports_observed_sides_without_private_audit_data(self):
        manifest={'role':'H','partition':'H-pair','examples':[],'pairs':[],
                  'base_side_by_pair':{},'task_by_source':{},'C_PRIVATE':'PRIVATE_SENTINEL'}
        records=[]
        for i in range(128):
            ident=str(i).zfill(3);task=('comparison','read_value','difference')[i%3];base=i%2
            answers=(encode_truth('binary','A'),encode_truth('binary','B'))
            manifest['pairs'].append({'pair_id':ident,'source_id':ident,'question':'Paired synthetic question',
                                      'answers':answers})
            manifest['base_side_by_pair'][ident]=base;manifest['task_by_source'][ident]=task
            pattern=((1,1),(1,0),(0,1),(0,0))[i%4]
            # pattern is ordered as base/counterfactual; records retain manifest side order.
            side_correct=[None,None];side_correct[base],side_correct[1-base]=pattern
            for side in (0,1):
                expected=('A','B')[side];raw=expected if side_correct[side] else ('B' if expected=='A' else 'A')
                records.append({'sample_id':fingerprint({'pair_id':ident,'side':side}),
                    'raw_answer':raw,'committed_answer':raw,'correct':side_correct[side],'generated_tokens':3})
        summary=summarize_pairs(manifest,{'records':records,'private':'PRIVATE_SENTINEL'})
        totals={key:sum(row[key] for row in summary['counts_by_task'].values())
                for key in ('both_correct','base_only','counterfactual_only','both_wrong')}
        self.assertEqual(totals,{'both_correct':32,'base_only':32,'counterfactual_only':32,'both_wrong':32})
        single,_=fixture();single_summary=summarize(single,fixture()[1])
        prompt=proposal_prompt('V3 contract',single_summary,summary)
        self.assertIn('BEGIN_OBSERVED_H_PAIR_FEEDBACK',prompt)
        self.assertIn('Paired synthetic question',prompt)
        self.assertNotIn('PRIVATE_SENTINEL',prompt)

    def test_pair_feedback_rejects_wrong_role_coverage_and_reward(self):
        manifest={'role':'H','partition':'H-pair','examples':[],'pairs':[],
                  'base_side_by_pair':{},'task_by_source':{}}
        records=[]
        for i in range(128):
            ident=str(i);answers=(encode_truth('binary','A'),encode_truth('binary','B'))
            manifest['pairs'].append({'pair_id':ident,'source_id':ident,'question':'q','answers':answers})
            manifest['base_side_by_pair'][ident]=0;manifest['task_by_source'][ident]='comparison'
            for side,answer in enumerate(('A','B')):
                records.append({'sample_id':fingerprint({'pair_id':ident,'side':side}),
                    'raw_answer':answer,'committed_answer':answer,'correct':1,'generated_tokens':1})
        for change in ('role','missing','reward','base'):
            with self.subTest(change=change):
                m,r=deepcopy(manifest),{'records':deepcopy(records)}
                if change=='role':m['partition']='C'
                elif change=='missing':r['records'].pop()
                elif change=='reward':r['records'][0]['correct']=0
                else:m['base_side_by_pair']['0']=2
                with self.assertRaises(ValueError):summarize_pairs(m,r)

    def test_pair_feedback_receipt_binds_real_nested_audit_identity(self):
        manifest={'role':'H','partition':'H-pair','examples':[],'pairs':[],
                  'base_side_by_pair':{},'task_by_source':{}}
        records=[]
        for i in range(128):
            ident=str(i);answers=(encode_truth('binary','A'),encode_truth('binary','B'))
            manifest['pairs'].append({'pair_id':ident,'source_id':ident,'question':'q','answers':answers})
            manifest['base_side_by_pair'][ident]=0;manifest['task_by_source'][ident]='comparison'
            for side,answer in enumerate(('A','B')):
                records.append({'sample_id':fingerprint({'pair_id':ident,'side':side}),
                    'raw_answer':answer,'committed_answer':answer,'correct':1,'generated_tokens':1})
        manifest['audit_data_sha256']=fingerprint(manifest['pairs'])
        harness_sha,weights_sha='a'*64,'b'*64
        result={'status':'COMPLETE_NATIVE_PAIR_EVALUATION','records':records,
            'audit':{'identity':{'weights_sha256':weights_sha,
                'audit_data_sha256':manifest['audit_data_sha256']},
                'harness_sha256':harness_sha,'role':'H',
                'pair_ids':[pair['pair_id'] for pair in manifest['pairs']],
                'correctness':[[1,1] for _ in manifest['pairs']]}}
        with TemporaryDirectory() as directory:
            manifest_path=Path(directory)/'manifest.json';result_path=Path(directory)/'result.json'
            manifest_path.write_text(json.dumps(manifest));result_path.write_text(json.dumps(result))
            receipt=pair_feedback_receipt(manifest_path,result_path,
                                           harness_sha256=harness_sha,weights_sha256=weights_sha)
            self.assertEqual(receipt['status'],'COMPLETE_H_PAIR_FEEDBACK')
            self.assertEqual(receipt['manifest_sha256'],file_sha256(manifest_path))
            bad=deepcopy(result);bad['audit']['correctness'][0]=[0,1]
            result_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'audit receipt'):
                pair_feedback_receipt(manifest_path,result_path,
                                      harness_sha256=harness_sha,weights_sha256=weights_sha)


if __name__ == '__main__': unittest.main()
