"""Second-phase native counters and saved-state validation on CPU fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ours.joint_training_phase2_result import phase_metrics


class PhaseTwoMetricsTests(unittest.TestCase):
    def test_second_global_step_counts_multiple_minibatches_and_empty_updates(self):
        positive='step:2 - online_rsft/accepted:9 - online_rsft/total:64 - online_rsft/sft_epochs:1 - training/global_step:2 - actor/sft_updates:2 - actor/sft_token_count:99 - actor/sft_loss:0.3 - actor/grad_norm:0 - timing_s/update_actor:5 - timing_s/save_checkpoint:1 - timing_s/update_weights:1'
        empty='step:2 - online_rsft/accepted:0 - online_rsft/total:64 - online_rsft/sft_epochs:1 - training/global_step:2 - online_rsft/skipped_empty_sft:1 - timing_s/save_checkpoint:1 - timing_s/update_weights:1'
        audit={'accepted':9,'accepted_loss_tokens':99}
        self.assertEqual(phase_metrics(positive,audit)['optimizer_steps'],2)
        self.assertEqual(phase_metrics(empty,{'accepted':0,'accepted_loss_tokens':0})['optimizer_steps'],0)
        for log in ('',positive+'\n'+positive,positive.replace('step:2','step:1'),positive.replace('sft_updates:2','sft_updates:1'),
                    positive.replace('training/global_step:2','training/global_step:1'),positive.replace('actor/grad_norm:0','actor/grad_norm:nan')):
            with self.assertRaises(ValueError):phase_metrics(log,audit)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires native saved-state runtime')
class PhaseTwoStateTests(unittest.TestCase):
    def test_incomplete_actor_rng_and_data_receipts_cannot_certify_restoration(self):
        from ours.joint_resume_receipts import verify_restore_receipts
        from ours.tests.joint_phase2_fixture import incoming_checkpoint,restore_receipts
        from ours.native_search import write_json
        from ours.visual_task import file_sha256
        source=Path('data/native-rsft/controlled-weight-only-seed42-222801/global_step_1')
        plan=json.loads(Path('results/controlled-whale-seed42-phase1-plan-20260910-v1.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan.update(kind='controlled_joint_phase2_plan',phase=2,
                resume_checkpoint=incoming_checkpoint(root,source),harness_sha256=file_sha256(Path(plan['harness'])))
            plan['source_sha256']['ours/native_resume_observation.py']=file_sha256(Path('ours/native_resume_observation.py'))
            path=root/'plan.json';write_json(path,plan);receipts=root/'receipts';job='999990'
            actor,loader=restore_receipts(plan,path,job,receipts)
            self.assertEqual(verify_restore_receipts(plan,path,job,receipts)['status'],'PASS_RESTORED_NATIVE_STATE_RECEIPTS')
            for name in ('tensor','rng','sampler','condition'):
                with self.subTest(name=name):
                    a,b=deepcopy(actor),deepcopy(loader)
                    if name=='tensor':a['tensors']=[];a['tensor_count_including_aliases']=0
                    elif name=='rng':a['rng_after_observation_sha256']='different'
                    elif name=='sampler':b['pending_state_digest']='different'
                    else:a['condition']='whale_fst'
                    write_json(receipts/'resume-actor-rank-0.json',a)
                    b['actor_observation_sha256']=file_sha256(receipts/'resume-actor-rank-0.json')
                    write_json(receipts/'resume-loader.json',b)
                    with self.assertRaises(ValueError):verify_restore_receipts(plan,path,job,receipts)

    def test_empty_archived_state_and_real_constant_lr_scheduler_advance(self):
        import torch
        from ours.joint_resume_receipts import CHECKPOINT_FILES
        from ours.joint_training_phase2_result import saved_state
        from ours.tests.joint_phase2_fixture import incoming_checkpoint
        from ours.visual_task import file_sha256
        plan=json.loads(Path('results/controlled-whale-seed42-phase1-plan-20260910-v1.json').read_text())
        source=Path('data/native-rsft/controlled-weight-only-seed42-222801')
        before={str(source/f'global_step_{i}'/n):file_sha256(source/f'global_step_{i}'/n) for i in (1,2)
                for n in ('data.pt','actor/extra_state_world_size_1_rank_0.pt')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan['resume_checkpoint']=incoming_checkpoint(root,source/'global_step_1')
            directory=root/'global_step_2';(directory/'actor/huggingface').mkdir(parents=True)
            for n in CHECKPOINT_FILES:
                if n!='actor/model_world_size_1_rank_0.pt':shutil.copyfile(source/'global_step_2'/n,directory/n)
            shutil.copyfile(root/'global_step_1/actor/model_world_size_1_rank_0.pt',directory/'actor/model_world_size_1_rank_0.pt')
            self.assertEqual(saved_state(directory,plan,0)['scheduler_last_epoch'],1)
            with self.assertRaisesRegex(ValueError,'scheduler'):saved_state(directory,plan,2)
            extra_path=directory/'actor/extra_state_world_size_1_rank_0.pt'
            extra=torch.load(extra_path,map_location='cpu',weights_only=False)
            parameter=torch.nn.Parameter(torch.zeros(1));optimizer=torch.optim.AdamW([parameter],lr=1e-7)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            scheduler.load_state_dict(extra['lr_scheduler'])
            # Two toy minibatches and the single native per-batch scheduler step.
            for _ in range(2):parameter.grad=torch.ones_like(parameter);optimizer.step()
            scheduler.step();extra['lr_scheduler']=scheduler.state_dict();torch.save(extra,extra_path)
            self.assertEqual(saved_state(directory,plan,2)['scheduler_last_epoch'],2)
            bad=deepcopy(plan);bad['native_dataset_order']['checkpoint_sampler_snapshots'][1]['sampler_yielded']=8
            with self.assertRaisesRegex(ValueError,'sampler'):saved_state(directory,bad,2)
            (directory/'actor/optim_world_size_1_rank_0.pt').write_bytes(b'forbidden')
            with self.assertRaisesRegex(ValueError,'optimizer-state'):saved_state(directory,plan,2)
        self.assertEqual(before,{p:file_sha256(Path(p)) for p in before})
        print(json.dumps({'kind':'joint_phase2_saved_state_cpu_fixture','status':'PASS','actual_native_state_unchanged':True,
            'empty_and_two_minibatch_scheduler_checked':True,'toy_optimizer_steps':2,'task_model_calls':0}),flush=True)
