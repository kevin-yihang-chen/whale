"""Actual task-loader order and preservation of the parent's RNG state."""
import importlib.util
import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ours.controlled_pilot_continuation import prepare


class ContinuationScopeTests(unittest.TestCase):
    def test_completed_seed_and_existing_plans_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'
            with self.assertRaisesRegex(ValueError,'unstarted'):prepare(path,42)
            path.with_suffix('.yaml').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Preserve existing'):prepare(path,43)
            self.assertEqual(path.with_suffix('.yaml').read_text(),'preserve')


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'Requires the native data and tensor runtime')
class NativeTaskOrderTests(unittest.TestCase):
    def test_actual_three_seed_loader_matches_recorded_schedule_and_restores_rng(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from ours.controlled_data_order import native_data_order
        path=Path('results/controlled-weight-only-seed42-plan-20260910-v2.json')
        plan=json.loads(path.read_text());raw=Path(plan['resolved_config']).read_text()
        expected={42:[['0095W','003o0','00BQD','007fJ','008qL','004Ao','003md','004b0'],
                      ['004u0','00Ar2','005gP','003r5','005nD','00AB1','003UW','00B7G']],
                  43:[['007eS','009aD','009bR','00AFG','004WZ','009L0','004RF','006XF'],
                      ['003mh','00BNd','005yO','007bH','0030b','006OI','00BM8','007hv']],
                  44:[['006of','009IO','009Wc','008Sk','006RM','006NL','0030b','00A5m'],
                      ['007eS','003r5','002vV','009lk','007bH','0061g','00B2k','00B7G']]}
        for seed in (42,43,44):
            with self.subTest(seed=seed):
                config=OmegaConf.create(raw[raw.index('model_engine: dp\n'):]);config.data.seed=seed
                random.seed(194);np.random.seed(195);torch.manual_seed(196)
                before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
                result=native_data_order(OmegaConf.to_container(config,resolve=True))
                self.assertEqual(result['planned_batch_ids'],expected[seed])
                self.assertEqual(result['native_dataset_class'],'RLHFDataset')
                self.assertEqual(result['native_filtered_ids'],plan['native_dataset_order']['native_filtered_ids'])
                self.assertEqual(random.getstate(),before[0])
                now=np.random.get_state();self.assertEqual(now[0],before[1][0]);np.testing.assert_array_equal(now[1],before[1][1])
                self.assertEqual(now[2:],before[1][2:]);self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
                if seed==42:
                    checkpoint=Path('data/native-rsft/controlled-weight-only-seed42-222801')
                    for step in (1,2):
                        with gzip.open(checkpoint/f'audit/batch-{step}.jsonl.gz','rt') as stream:
                            next(stream);ids=[json.loads(line)['metadata']['extra_info']['puzzle_id'] for i,line in enumerate(stream) if i%8==0]
                        self.assertEqual(result['planned_batch_ids'][step-1],ids)
                print(json.dumps({'kind':'actual_native_task_loader_preflight','seed':seed,'status':'PASS',
                    'batch_ids':result['planned_batch_ids'],'parent_rng_restored':True,'new_model_calls':0}),flush=True)

    def test_independent_probe_disagreement_is_rejected(self):
        from ours.controlled_data_order import native_data_order
        from omegaconf import OmegaConf
        plan=json.loads(Path('results/controlled-weight-only-seed42-plan-20260910-v2.json').read_text())
        raw=Path(plan['resolved_config']).read_text();cfg=OmegaConf.create(raw[raw.index('model_engine: dp\n'):])
        with patch('ours.controlled_data_order.native_schedule',return_value={'batch_ids':[]}):
            with self.assertRaisesRegex(ValueError,'differs from index-only'):
                native_data_order(OmegaConf.to_container(cfg,resolve=True))


if __name__=='__main__':unittest.main()
