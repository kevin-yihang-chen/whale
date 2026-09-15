"""Completion cannot be inferred from partial, nonfinite or inconsistent logs."""
import unittest
from ours.controlled_training_result import batch_metrics,metric


class ControlledCompletionTests(unittest.TestCase):
    def test_native_numpy_scalars_and_nonfinite_values(self):
        self.assertEqual(metric('x:np.float64(1e-07)','x'),1e-7)
        for value in ('nan','inf','-inf'):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):metric('x:'+value,'x')
        with self.assertRaisesRegex(ValueError,'Missing'):metric('other:2','x')

    def test_two_batches_allow_empty_update_but_require_every_recorded_optimizer_step(self):
        first='step:1 - online_rsft/accepted:3 - online_rsft/total:64 - online_rsft/sft_epochs:1 - training/global_step:1 - actor/sft_updates:np.float64(1.0) - actor/sft_token_count:9 - actor/sft_loss:0.3 - actor/grad_norm:2 - timing_s/update_actor:5 - timing_s/save_checkpoint:1 - timing_s/update_weights:1'
        second='step:2 - online_rsft/accepted:0 - online_rsft/total:64 - online_rsft/sft_epochs:1 - training/global_step:2 - online_rsft/skipped_empty_sft:1 - timing_s/save_checkpoint:1 - timing_s/update_weights:1'
        batches=[{'step':1,'accepted':3,'accepted_loss_tokens':9},{'step':2,'accepted':0,'accepted_loss_tokens':0}]
        result=batch_metrics(first+'\n'+second,batches)
        self.assertEqual([r['optimizer_steps'] for r in result],[1,0])
        for log in (first,first+'\n'+first+'\n'+second,(first+'\n'+second).replace('accepted:3','accepted:4'),
                    (first+'\n'+second).replace('np.float64(1.0)','np.float64(2.0)')):
            with self.subTest(log=log),self.assertRaises(ValueError):batch_metrics(log,batches)


if __name__=='__main__':unittest.main()
