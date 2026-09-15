"""Four-shard CPU replay using duplicated, relabelled MH fixtures, never test data."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ours.native_search import tree_hashes,write_json
from ours.visual_task import file_sha256


@unittest.skipUnless(importlib.util.find_spec('chess') and importlib.util.find_spec('transformers'),'Requires native archived-reply runtime')
class HeldoutExecutionFixtures(unittest.TestCase):
    def test_four_shards_allocation_waves_full_replay_and_terminal_closure(self):
        from autoharness_chess_puzzle import runner
        from ours.heldout_evaluation import run_evaluation
        from ours.audit_heldout_evaluation import audit_evaluation
        from ours.controlled_heldout import close
        source_plan=json.loads(Path('results/updated-mh-phase-plan-20260909-v2.json').read_text())
        source=Path(source_plan['phase_root'])/'evaluations/h1'
        if not source.exists():self.skipTest('Archived MH replies absent')
        request=json.loads((source/'request.json').read_text())
        native_examples={e.example_id:e for e in runner._read_examples(source_plan['dataset'],limit=32,seed=42)}
        original_hashes=tree_hashes(source)
        examples=[];mapping={};original={}
        for shard in range(4):
            directory=source/f'replica-{shard%2}'
            ids=json.loads((directory/'ordered-ids.json').read_text())
            rows=json.loads((directory/'native-outputs.json').read_text());original[shard]=dict(zip(ids,rows,strict=True))
            for i,key in enumerate(ids):
                new_id=f'fixture_{4*i+shard:03d}';mapping[new_id]=(shard,key)
                examples.append(replace(native_examples[key],example_id=new_id))
        random.Random(42).shuffle(examples)
        seed_receipt=json.loads(Path('data/controlled-harness-only-seed42-20260910/evaluations/h12/replica-0/process-randomization.json').read_text())
        results=[]
        for gpus in (1,2,4):
            with self.subTest(gpus=gpus),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);private=root/'evaluation';path=root/'plan.json'
                plan=deepcopy(source_plan);plan.update(kind='controlled_heldout_plan',data_role='test',condition='harness_only',
                    evaluation_root=str(private),examples=64,logical_shards=4,cohort_sha256='synthetic-cohort-fixture',
                    harness=request['harness_path'],harness_sha256=request['harness_sha256'],dataset='fixture://duplicated-mh-replies-only',
                    puzzle_ids=sorted(mapping),shard_ids=[sorted(mapping)[i::4] for i in range(4)],
                    resources={'gpus':gpus,'time_limit_seconds':9600//gpus,'cpus':8*gpus,'ram_gib':64*gpus},
                    target_manifest=json.loads(Path(source_plan['transition']).read_text())['exported'],
                    limitations=['Synthetic final plan/cohort/GPU/Slurm; each of32 old MH cases appears twice with fixture IDs.'])
                write_json(path,plan);spawned=[]
                class Child:
                    returncode=0
                    def __init__(self,argv,**kwargs):
                        shard=int(argv[-1]);spawned.append((shard,kwargs['env']['CUDA_VISIBLE_DEVICES'],kwargs['env']['WHALE_TRIAL_SEED']))
                        directory=private/f'replica-{shard}';shutil.copytree(source/f'replica-{shard%2}',directory)
                        ordered=[e.example_id for e in examples if e.example_id in plan['shard_ids'][shard]]
                        rows=[original[shard][mapping[key][1]] for key in ordered]
                        write_json(directory/'ordered-ids.json',ordered);write_json(directory/'native-outputs.json',rows)
                        proof=json.loads((directory/'worker-weights.json').read_text());proof['plan_sha256']=file_sha256(path)
                        write_json(directory/'worker-weights.json',proof);write_json(directory/'process-randomization.json',{**seed_receipt,'fixture_only':True})
                        events=[json.loads(line) for line in (directory/'request-events.jsonl').read_text().splitlines()]
                        for event in events:event['id']=f'fixture-shard-{shard}-'+event['id']
                        (directory/'request-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
                        metadata=json.loads((directory/'evaluation/val.json').read_text())['metadata']
                        metadata.update(dataset_path=plan['dataset'],seed=42,harness_path=plan['harness'],limit=16)
                        write_json(directory/'evaluation/val.json',runner.summarize_outputs(rows,metadata))
                        (directory/'result.json').unlink()
                        write_json(directory/'result.json',{'status':'COMPLETED','replica':shard,'data_role':'test',
                            'condition':'harness_only','seed':42,'plan_sha256':file_sha256(path),'harness_sha256':plan['harness_sha256'],
                            'gpu':'synthetic CPU fixture','artifact_sha256':tree_hashes(directory)})
                    def poll(self):return self.returncode
                    def wait(self,**kwargs):return self.returncode
                def read(path,*,limit,seed):
                    self.assertEqual((str(path),limit,seed),(plan['dataset'],64,42));return list(examples)
                with patch.dict(os.environ,SLURM_JOB_ID='999970',CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(gpus)),
                     CHESS_PUZZLE_DEFAULT_MAX_TURNS='9',CHESS_PUZZLE_MAX_TURNS_CAP='18',CHESS_PUZZLE_FORMAT_RETRIES='1',CHESS_PUZZLE_ILLEGAL_RETRIES='1'), \
                     patch.object(runner,'_read_examples',read),patch('ours.heldout_evaluation.subprocess.Popen',Child):
                    run_evaluation(SimpleNamespace(plan=path,evaluation=private),plan)
                    receipt=json.loads((private/'result.json').read_text())
                    self.assertEqual((receipt['solved'],receipt['new_model_calls'],receipt['generated_tokens']),(12,68,502176))
                    self.assertEqual(spawned,[(i,str(i%gpus),'42') for i in range(4)])
                    waves=json.loads((private/'waves.json').read_text());self.assertEqual(len(waves),4//gpus)
                    self.assertEqual([a['replica'] for wave in waves for a in wave['assignments']],[0,1,2,3])
                    slurm=root/'terminal.txt';slurm.write_text(f'JobId=999970 JobState=COMPLETED ExitCode=0:0 RunTime=00:01:00 AllocTRES=cpu={8*gpus},mem={64*gpus}G,gres/gpu={gpus},gres/gpu:h800={gpus} Command={Path.cwd()}/ours/run_controlled_heldout.sh\n')
                    with patch('ours.controlled_heldout.check_plan',return_value=plan):
                        closed=close(path,slurm);self.assertEqual(closed['status'],'COMPLETED_HELDOUT_EVALUATION')
                        self.assertEqual(closed['examples'],64);self.assertAlmostEqual(closed['allocation']['gpu_hours'],gpus/60)
                        results.append((closed['solved'],closed['calls'],closed['generated_tokens']))
                        old=slurm.read_text();slurm.write_text(old.replace('COMPLETED','RUNNING'))
                        with self.assertRaisesRegex(ValueError,'did not complete'):close(path,slurm)
                        slurm.write_text(old.replace(f'cpu={8*gpus}',f'cpu={4*gpus}'))
                        with self.assertRaisesRegex(ValueError,'CPU or RAM'):close(path,slurm)
                        slurm.write_text(old)
                        saved=receipt.copy();wrong=deepcopy(waves);wrong[0]['assignments'][0]['replica']=3
                        write_json(private/'waves.json',wrong)
                        modified=deepcopy(receipt);modified['artifact_sha256']['waves.json']=file_sha256(private/'waves.json')
                        write_json(private/'result.json',modified)
                        with self.assertRaisesRegex(ValueError,'logical shard waves'):close(path,slurm)
                        write_json(private/'waves.json',waves);write_json(private/'result.json',saved)
                    if gpus==4:
                        directory=private/'replica-0';p=directory/'process-randomization.json';bad=json.loads(p.read_text());bad['trial_seed']=43;write_json(p,bad)
                        r=json.loads((directory/'result.json').read_text());hashes=tree_hashes(directory);hashes.pop('result.json');r['artifact_sha256']=hashes;write_json(directory/'result.json',r)
                        with self.assertRaisesRegex(ValueError,'randomization'):audit_evaluation(plan,path,private)
                    with self.assertRaises(FileExistsError):run_evaluation(SimpleNamespace(plan=path,evaluation=private),plan)
        self.assertEqual(results,[results[0]]*3);self.assertEqual(tree_hashes(source),original_hashes)
        print(json.dumps({'kind':'four_shard_archived_mh_cpu_fixture','status':'PASS','unique_source_mh_puzzles':32,
            'fixture_ids':64,'each_source_puzzle_copied_twice':True,'replayed_calls_per_case':68,'physical_widths_checked':[1,2,4],
            'source_archive_unchanged':True,'cohort_and_slurm_provenance_are_synthetic':True,'test_tasks_loaded':False,
            'new_model_calls':0,'new_gpu_jobs':0}),flush=True)


class HeldoutFailureTests(unittest.TestCase):
    def test_failed_wave_cannot_launch_later_shards(self):
        from ours.heldout_evaluation import run_shards
        created=[]
        class Child:
            def __init__(self,argv,**kwargs):
                self.replica=int(argv[-1]);created.append(self.replica);self.returncode=int(self.replica==1)
            def poll(self):return self.returncode
            def wait(self,**kwargs):return self.returncode
        with tempfile.TemporaryDirectory() as tmp,patch('ours.heldout_evaluation.subprocess.Popen',Child):
            args=SimpleNamespace(plan=Path(tmp)/'plan.json',evaluation=Path(tmp))
            with self.assertRaisesRegex(ValueError,'shard failed'):run_shards(args,{'seed':42},['0','1'])
        self.assertEqual(created,[0,1])


if __name__=='__main__':unittest.main()
