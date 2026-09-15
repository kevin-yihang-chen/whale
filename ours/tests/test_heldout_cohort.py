"""Final lineage routing and unopened-test preparation; simulated parents are explicit."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ours.heldout_cohort import CONDITIONS,SEEDS,certify_trial,seal,check
from ours.native_search import write_json
from ours.visual_task import file_sha256


class CohortCoverageTests(unittest.TestCase):
    def test_every_trial_is_declared_and_incomplete_entries_never_become_evaluable(self):
        entries=[{'condition':c,'seed':s,'status':'INCOMPLETE','reason':'Explicit CPU fixture; not a trial status report.',
                  'evidence_paths':['ours/controlled_pilot_protocol.md']} for c in CONDITIONS for s in SEEDS]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input.json';dest=root/'cohort.json'
            for bad in (entries[:-1],entries+[entries[0]],entries[:-1]+[entries[0]]):
                write_json(source,bad)
                with self.assertRaisesRegex(ValueError,'exactly once'):seal(source,dest)
                self.assertFalse(dest.exists())
            write_json(source,entries);result=seal(source,dest)
            self.assertEqual(len(result['trials']),12);self.assertFalse(any(t['evaluable'] for t in result['trials']))
            self.assertFalse(result['test_tasks_loaded']);self.assertEqual(check(dest),result)
            with self.assertRaisesRegex(ValueError,'Unfinished trials'):check(dest,condition='harness_only',seed=42)
            with self.assertRaisesRegex(ValueError,'Preserve an existing'):seal(source,dest)
            entries[0]['reason']='changed';write_json(source,entries)
            with self.assertRaisesRegex(ValueError,'cohort identity'):check(dest)
        bad={'condition':'weight_only','seed':42,'status':'INCOMPLETE','reason':'fixture','evidence_paths':['ours/README.md'],'score':0.}
        with self.assertRaisesRegex(ValueError,'without a model or score'):certify_trial(bad)

    def test_completed_harness_only_uses_native_selection_and_common_untrained_weights(self):
        common=json.loads(Path('results/canonical-initialization-20260910.json').read_text())['exported']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);harness=root/'search/harnesses/h1/harness.py';harness.parent.mkdir(parents=True)
            harness.write_text('fixture = True\n');result_path=root/'result.json';write_json(result_path,{'fixture_only':True})
            source=root/'plan.json';plan={'kind':'controlled_staged_mh_plan','condition':'harness_only','seed':42,
                'target_manifest':common,'phase_root':str(root),'source_sha256':{}}
            write_json(source,plan)
            proof={'condition':'harness_only','seed':42,'harness':str(harness),'harness_sha256':file_sha256(harness),
                'accepted':'h1','completed_rounds':5,'native_perfect_stop':False,'artifact_sha256':{},'source_sha256':{},
                'search_sha256':{'harnesses/h1/harness.py':file_sha256(harness)},'result_path':str(result_path),
                'result_sha256':file_sha256(result_path)}
            entry={'condition':'harness_only','seed':42,'status':'COMPLETE','search_plan':str(source)}
            with patch('ours.heldout_cohort.verify_completed_search',return_value=proof), \
                 patch('ours.heldout_cohort.checkpoint_manifest',return_value=common):
                certified=certify_trial(entry);self.assertTrue(certified['evaluable'])
                self.assertEqual(certified['harness_sha256'],file_sha256(harness));self.assertEqual(certified['model'],common)
                self.assertEqual(certified['provenance']['optimizer_steps'],0)
                for key,value in (('seed',43),('kind','controlled_joint_staged_mh_plan'),('target_manifest',{'not_common':True})):
                    changed=deepcopy(plan);changed[key]=value;write_json(source,changed)
                    with self.assertRaisesRegex(ValueError,'fixed-weight final lineage'):certify_trial(entry)
                write_json(source,plan)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'Requires native finalizer imports')
class TrainedLineageTests(unittest.TestCase):
    def test_all_trained_condition_routes_preserve_phase_seed_harness_and_deviation(self):
        from ours.staged_search import BASELINE
        common=json.loads(Path('results/canonical-initialization-20260910.json').read_text())['exported']
        for condition,seed,module,kind in [('weight_only',42,'ours.controlled_checkpoint_export_result','old'),
            ('weight_only',43,'ours.controlled_continuation_export','continuation'),
            ('weight_only',44,'ours.controlled_continuation_export','continuation'),
            ('whale',42,'ours.joint_checkpoint_export','joint'),('whale_fst',44,'ours.joint_checkpoint_export','joint')]:
            with self.subTest(condition=condition,seed=seed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);native={'step':2,'directory':str(root/'native'),'artifact_sha256':{}}
                training={'condition':condition,'seed':seed,'initial_manifest':common,'harness':str(BASELINE.resolve()),
                    'harness_sha256':file_sha256(BASELINE)}
                trained={'condition':condition,'seed':seed,'job_id':'synthetic','original_preflight_batch_ids_match':kind!='old',
                    'deviation':{'fixture_only':True} if kind=='old' else None,
                    'kind':'controlled_joint_phase2_training_result' if kind=='joint' else 'synthetic-continuation',
                    'phase':2,'harness_sha256':file_sha256(BASELINE)}
                training_path=root/'training.json';trained_path=root/'trained.json';write_json(training_path,training);write_json(trained_path,trained)
                export_path=root/'export.json';result_path=root/'export-result.json'
                export={'training_plan':str(training_path),'training_result':str(trained_path),'base':common,'source_sha256':{}}
                result={'condition':condition,'seed':seed,'step':2,'resume_checkpoint':native,'exported':common,
                    'job_id':'synthetic-export','slurm_terminal_path':str(root/'synthetic-terminal.txt'),
                    'artifact_sha256':{},'optimizer_steps_through_checkpoint':0}
                write_json(export_path,export);write_json(result_path,result)
                inputs=(training,trained,{'exported':common},native,{})
                input_module='ours.controlled_checkpoint_export' if kind=='old' else module
                with patch(input_module+'.completed_inputs',return_value=inputs),patch(module+'.close',return_value=result), \
                     patch('ours.heldout_cohort.checkpoint_manifest',return_value=common):
                    entry={'condition':condition,'seed':seed,'status':'COMPLETE','export_plan':str(export_path),'export_result':str(result_path)}
                    certified=certify_trial(entry)
                    self.assertEqual(certified['condition'],condition);self.assertEqual(certified['seed'],seed)
                    self.assertEqual(certified['harness_sha256'],file_sha256(BASELINE));self.assertEqual(certified['provenance']['optimizer_steps'],0)
                    self.assertEqual(certified['provenance']['training_deviation'],trained['deviation'])
                    result['seed']=43 if seed!=43 else 44;write_json(result_path,result)
                    with self.assertRaisesRegex(ValueError,'Mixed final training'):certify_trial(entry)
                    result['seed']=seed;result['step']=1;write_json(result_path,result)
                    with self.assertRaisesRegex(ValueError,'Mixed final training'):certify_trial(entry)
        print(json.dumps({'kind':'final_trained_lineage_routing_fixture','status':'PASS','parent_finalizers_are_synthetic_stubs':True,
            'old_and_continuation_weight_only_and_both_joint_conditions_checked':True,'zero_updates_remain_valid':True,
            'test_tasks_loaded':False,'new_model_calls':0}),flush=True)

    def test_plan_preparation_keeps_test_reader_closed_and_freezes_allocation(self):
        from autoharness_chess_puzzle import runner
        from ours.controlled_heldout import prepare,check_plan
        from ours.staged_search import BASELINE
        common=json.loads(Path('results/canonical-initialization-20260910.json').read_text())['exported']
        old=json.loads(Path('results/controlled-harness-only-seed42-plan-20260910-v1.json').read_text())
        trial={'model':common,'initial_manifest':common,'harness':str(BASELINE.resolve()),'harness_sha256':file_sha256(BASELINE),
            'provenance':{'fixture_only':True,'optimizer_steps':0}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort=root/'cohort.json';write_json(cohort,{'fixture_only':True})
            with patch('ours.controlled_heldout.check_cohort',return_value=trial), \
                 patch('ours.controlled_heldout.select_probe_reference',return_value=(old['worker_probe_reference'],old['worker_probe_coordinates'])), \
                 patch('ours.probe_updated_vllm.inference_contract',return_value={'fixture_only':True}), \
                 patch.object(runner,'_read_examples',side_effect=AssertionError('Preparation must not load task examples')):
                for gpus in (1,2,4):
                    path=root/f'plan-{gpus}.json';plan=prepare(path,cohort,'harness_only',42,root/f'eval-{gpus}',gpus)
                    self.assertEqual(check_plan(path),plan);self.assertEqual(plan['resources']['gpus'],gpus)
                    self.assertEqual([len(s) for s in plan['shard_ids']],[16]*4)
                    changed=deepcopy(plan);changed['target_config']['temperature']=.5;write_json(path,changed)
                    with self.assertRaisesRegex(ValueError,'decoding'):check_plan(path)
                    write_json(path,plan);changed=deepcopy(plan);changed['resources']['gpus']=3;write_json(path,changed)
                    with self.assertRaisesRegex(ValueError,'allocation budget'):check_plan(path)


if __name__=='__main__':unittest.main()
