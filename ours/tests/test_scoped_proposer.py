"""Multi-slot integrity and real iteration propagation, without paid requests."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.glm_gateway import MODEL
from ours.native_search import tree_hashes
from ours.scoped_proposer import check_proposal, isolated_native_proposal, normalize_candidates


class ScopedProposerTests(unittest.TestCase):
    def test_scope_rejects_duplicate_conflicting_or_stale_metadata(self):
        payload = {'candidates': [{'slot': 'h4', 'parent': 'h2'}, {'name': 'h5'}, {'name': 'h6'}]}
        self.assertEqual([p['name'] for p in normalize_candidates(payload, ['h4','h5','h6'], {'h0','h1','h2'})['candidates']],
                         ['h4','h5','h6'])
        for entries in ([{'name':'h1'}], [{'name':'h4','slot':'h5'}], [{'name':'h4'},{'name':'h4'}],
                        [{'name':'h4','path':'../h0/harness.py'}], [{'name':'h4','parent':'h99'}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                normalize_candidates({'candidates':entries}, ['h4','h5','h6'], {'h0','h1','h2'})

    def test_three_candidates_cannot_modify_earlier_evidence_or_add_another_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'harnesses/h0').mkdir(parents=True)
            old=root/'harnesses/h0/harness.py'; old.write_text('baseline')
            before=tree_hashes(root)
            for name in ('h4','h5','h6'):
                (root/f'harnesses/{name}').mkdir()
                (root/f'harnesses/{name}/harness.py').write_text('candidate')
            check_proposal(root,before,['h4','h5','h6'],2)
            old.write_text('changed')
            with self.assertRaisesRegex(ValueError,'protected evidence'):
                check_proposal(root,before,['h4','h5','h6'],2)
            old.write_text('baseline')
            extra=root/'harnesses/h7'; extra.mkdir(); (extra/'harness.py').write_text('unallocated')
            with self.assertRaisesRegex(ValueError,'Unexpected proposer artifact'):
                check_proposal(root,before,['h4','h5','h6'],2)

    def test_actual_iteration_and_all_slots_reach_proposer_with_stale_metadata_removed(self):
        @contextmanager
        def fake_relay(*args,**kwargs):
            yield {'base_url':'http://127.0.0.1:1','token':'offline-fixture'}
        for write_metadata in (True,False):
            with self.subTest(write_metadata=write_metadata), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); source=root/'search'
                for name in ('h0','h1','h2','h3'):
                    (source/f'harnesses/{name}').mkdir(parents=True)
                    (source/f'harnesses/{name}/harness.py').write_text(f'protected {name}')
                old={'candidates':[{'name':'h1'}]}
                (source/'pending_eval.json').write_text(json.dumps(old))
                def propose(*,task_prompt,iteration,run_dir,proposer_model,proposer_effort,timeout_seconds,use_api_key):
                    self.assertEqual(iteration,2)
                    self.assertEqual(task_prompt,'allocated h4 h5 h6')
                    self.assertFalse((run_dir/'pending_eval.json').exists())
                    for name in ('h4','h5','h6'):
                        (run_dir/f'harnesses/{name}').mkdir()
                        (run_dir/f'harnesses/{name}/harness.py').write_text(f'candidate {name}')
                    if write_metadata:
                        (run_dir/'pending_eval.json').write_text(json.dumps({'candidates':[{'slot':name} for name in ('h4','h5','h6')]}))
                    (run_dir/'logs/claude_sessions').mkdir(parents=True)
                    return SimpleNamespace(exit_code=0,duration_seconds=0.)
                def recover(run_dir,names):
                    self.assertEqual(names,['h4','h5','h6'])
                    return [{'name':name} for name in names if (run_dir/f'harnesses/{name}/harness.py').exists()]
                native=SimpleNamespace(propose_claude=propose,recover_pending_candidates=recover,
                    claude_wrapper=SimpleNamespace(build_command=lambda:None,_EMPTY_PLUGIN_DIR=None))
                plan={'data_role':'mh_val','max_cli_turns':1,'max_api_requests':1}
                with patch('ours.scoped_proposer.relay',fake_relay):
                    isolated_native_proposal(native,plan,root/'proposal',SimpleNamespace(summary=lambda:{}),
                        task_prompt='allocated h4 h5 h6',iteration=2,next_names=['h4','h5','h6'],run_dir=source,
                        proposer_model=MODEL,proposer_effort='low',timeout_seconds=1,use_api_key=True)
                pending=json.loads((source/'pending_eval.json').read_text())
                self.assertEqual([p['name'] for p in pending['candidates']],['h4','h5','h6'])
                for name in ('h0','h1','h2','h3'):
                    self.assertEqual((source/f'harnesses/{name}/harness.py').read_text(),f'protected {name}')
                report=json.loads((root/'proposal/proposal-integrity.json').read_text())
                self.assertEqual(report['metadata_source'],'proposer' if write_metadata else 'native_recovery')
                self.assertFalse((source/'.cli-state').exists())


@unittest.skipUnless(importlib.util.find_spec('chess'), 'Requires native Chess runtime')
class NativeMultiCandidateTests(unittest.TestCase):
    def test_two_original_iterations_evaluate_all_six_allocated_candidates(self):
        from autoharness_chess_puzzle import runner
        from ours.controlled_conditions import native_search_context
        from ours.tests.test_prompt_subspace import BASELINE, replace_user, write_harness
        @contextmanager
        def fake_relay(*args,**kwargs):
            yield {'base_url':'http://127.0.0.1:1','token':'offline-fixture'}
        for condition in ('whale','whale_fst'):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                config=root/'config.json'
                config.write_text(json.dumps({'models':[{'model':'fixture'}],'seeds':[44],
                    'task_profiles':{'mh_val':{'dataset_path':'fixture-only','limit':4}}}))
                args=SimpleNamespace(run_name='search',config=str(config),iterations=2,proposals_per_iter=3,
                    proposer_model=MODEL,proposer_effort='low',propose_timeout=1,early_stop_success_rate=1.,
                    fresh=False,force=False,start_iteration=1,early_stop_min_iters=0,early_stop_patience=2,
                    eval_only=False,use_api_key=True,prompt_only=condition=='whale_fst')
                evaluated,allocations=[],[]
                with native_search_context(condition,root,BASELINE) as (native,benchmark,_):
                    def propose(**kw):
                        names=[f'h{n}' for n in range(3*(kw['iteration']-1)+1,3*kw['iteration']+1)]
                        for name in names:
                            write_harness(kw['run_dir'],name,replace_user(BASELINE.read_text(),repr(name+' {observation}')))
                        (kw['run_dir']/'pending_eval.json').write_text(json.dumps({'candidates':[{'name':n} for n in names]}))
                        (kw['run_dir']/'logs/claude_sessions').mkdir(parents=True,exist_ok=True)
                        return SimpleNamespace(exit_code=0,duration_seconds=0.,show=lambda:None)
                    def boundary(**kw):
                        allocations.append(list(kw['next_names']))
                        return isolated_native_proposal(native,{'data_role':'mh_val','max_api_requests':1,'max_cli_turns':1},
                            root/f"proposal-{kw['iteration']}",SimpleNamespace(summary=lambda:{}),**kw)
                    def evaluate(**kw):
                        name=Path(kw['harness_path']).parent.name
                        evaluated.append(name)
                        number=int(name[1:]); successes=0 if not number else (number-1)%3+1
                        rows=[{'example_id':'redacted','reward':float(i<successes),'turn_count':1,
                               'metrics':{'assistant_tokens_used':1},'prompt':[],'harness_trace':[]} for i in range(4)]
                        summary=runner.summarize_outputs(rows,{'seed':44,'fixture_only':True})
                        out=Path(kw['output_dir']);out.mkdir(parents=True)
                        (out/'val.json').write_text(json.dumps(summary))
                        return summary
                    with patch('ours.scoped_proposer.relay',fake_relay),patch.object(native,'propose_claude',propose), \
                         patch.object(native,'propose_claude_with_retries',boundary),patch.object(benchmark,'evaluate_harness',evaluate):
                        native.run_evolve(args)
                    self.assertEqual(allocations,[['h1','h2','h3'],['h4','h5','h6']])
                    self.assertEqual(evaluated,['h0','h1','h2','h3','h4','h5','h6'])
                    self.assertEqual(native.get_accepted_harness(root/'search'),'h3')
                    print(json.dumps({'kind':'native_multi_candidate_fixture','status':'PASS',
                        'condition':condition,'allocated':allocations,'evaluated':evaluated,
                        'task_scores_are_synthetic':True,'api_calls':0}),flush=True)


if __name__=='__main__':
    unittest.main()
