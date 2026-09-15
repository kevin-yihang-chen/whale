"""Synthetic native-loop fixtures, including sequential resume across three slots."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours import fast_chart_search as search
from ours.acceptance import CandidateMetrics
from ours.evidence import AuditReceipt,EvaluationIdentity,fingerprint
from ours.visual_task import file_sha256
from meta_harness import meta_harness_chess_puzzle as native


class CompactSearchTests(unittest.TestCase):
    def test_native_three_candidate_resume_and_distinct_gates(self):
        for score in (.9,1.):
            with self.subTest(score=score),TemporaryDirectory() as folder:
                root=Path(folder);run=root/'search';(run/'logs/claude_sessions').mkdir(parents=True)
                identity=EvaluationIdentity(*(fingerprint(x) for x in ('w','H','C','decode','v')),phase='fixture')
                archive={}
                # Same marginal accuracy for h0 and h1; only h0 keeps the pair.
                for name,a,correct in [('h0',.5,((1,1),(0,0))),('h1',score,((1,0),(0,1))),
                                       ('h2',.7,((1,1),(1,0))),('h3',.6,((1,1),(0,0)))]:
                    code=run/f'harnesses/{name}/harness.py';code.parent.mkdir(parents=True)
                    code.write_text((search.ROOT/'ours/visual_harnesses/canonical_answer_harness.py').read_text()+'\n# '+name)
                    cp=root/f'{name}-plan.json';ap=root/f'{name}-allocation.json';cp.write_text('{}');ap.write_text('{}')
                    candidate=CandidateMetrics(name,file_sha256(code),a,2.)
                    archive[str(cp)]={'candidate':candidate,'audit':AuditReceipt(identity,candidate.harness_sha256,'C',('p0','p1'),correct),
                        'plan':{'reference_plan':'fixture'},'result':{'identity':asdict(identity)}}
                plan={'identity':asdict(identity),'seed':42,'parent_plan_sha256':fingerprint('parent'),
                    'baseline':{'plan':str(root/'h0-plan.json'),'allocation':str(root/'h0-allocation.json')}}
                search.write_new(root/'native-config.json',{'task_profiles':{'H':{}},'models':[],'seeds':[42]})
                def publish(directory,item):
                    p=directory/f"logs/H/{item['candidate'].name}/qwen35-4b";p.mkdir(parents=True,exist_ok=True)
                    search.once(p/'val.json',{'success_rate':item['candidate'].accuracy,'avg_reward':item['candidate'].accuracy,'mean_turn_count':2.})
                with patch.object(search,'load',return_value=plan),patch.object(search,'certify',side_effect=lambda p,a:archive[str(p)]),\
                     patch.object(search,'public_feedback',publish),patch.object(search,'load_isolated_visual_harness'),\
                     patch.object(native.claude_wrapper,'parse_stream_events',return_value=SimpleNamespace(show=lambda:None)):
                    search.advance(root)
                    self.assertTrue((root/'proposal-request.json').exists())
                    session=run/'logs/claude_sessions/fixture';session.mkdir()
                    (session/'events.jsonl').write_text('{}\n');search.write_new(session/'meta.json',{'duration_seconds':0,'exit_code':0,'cwd':str(run)})
                    search.write_new(root/'proposal-ready.json',{'request_sha256':file_sha256(root/'proposal-request.json'),
                        'session':str(session),'session_sha256':search.tree_hashes(session)})
                    search.write_new(run/'pending_eval.json',{'candidates':[{'name':n} for n in search.SLOTS]})
                    for name in search.SLOTS:
                        search.advance(root)
                        self.assertTrue((root/f'evaluation-request-{name}.json').exists())
                        cp=root/f'{name}-plan.json';ap=root/f'{name}-allocation.json'
                        search.write_new(root/f'evaluation-{name}.json',{'plan':str(cp),'allocation':str(ap),
                            'plan_sha256':file_sha256(cp),'allocation_sha256':file_sha256(ap)})
                    search.advance(root)
                    self.assertEqual(search.read(root/'result.json')['selections'],{'whale':'h1','veto':'h2','marginal_gate':'h1'})
                    self.assertEqual(search.read(root/'selection-veto.json')['stop']['stage'],'early_stop' if score==1 else 'ordinary')
                    self.assertEqual(search.read(root/'restored-selection-veto.json')['accepted_harness'],'h2')


if __name__=='__main__':unittest.main()
