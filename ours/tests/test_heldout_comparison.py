"""Synthetic score/cost fixtures for descriptive aggregation, with no task data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ours.heldout_comparison import CONDITIONS,SEEDS,build,export
from ours.native_search import write_json
from ours.visual_task import file_sha256


class HeldoutComparisonTests(unittest.TestCase):
    def test_full_and_missing_tables_and_failure_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort_path=root/'cohort.json';write_json(cohort_path,{'fixture':'synthetic twelve trial identities'})
            index_path=root/'index.json';trials=[];entries=[];closed={}
            for ci,condition in enumerate(CONDITIONS):
                for si,seed in enumerate(SEEDS):
                    model={'fixture':f'{condition}-{seed}'};harness='synthetic-harness-hash'
                    trials.append({'condition':condition,'seed':seed,'status':'COMPLETE','evaluable':True,'model':model,'harness_sha256':harness,'provenance':{'fixture':True}})
                    plan=root/f'{condition}-{seed}-plan.json';write_json(plan,{'fixture':True})
                    terminal=root/f'{condition}-{seed}-slurm.txt'
                    terminal.write_text(f'JobId={1000+ci*3+si} JobState=COMPLETED ExitCode=0:0 RunTime=00:01:00 AllocTRES=gres/gpu=2\n')
                    solved=si*16+ci
                    result={'plan':str(plan),'slurm_terminal_path':str(terminal),'condition':condition,'seed':seed,
                        'cohort_sha256':file_sha256(cohort_path),'target_manifest':model,'harness_sha256':harness,
                        'solved':solved,'success_rate':solved/64,'calls':64+si,'generated_tokens':64000}
                    path=root/f'{condition}-{seed}-result.json';write_json(path,result);closed[plan]=result
                    entries.append({'condition':condition,'seed':seed,'status':'COMPLETE','result':str(path)})
            failed=root/'failed-training.txt';failed.write_text('JobId=999 JobState=OUT_OF_MEMORY ExitCode=1:0 RunTime=01:00:00 AllocTRES=gres/gpu=2\n')
            registry=[{'condition':'whale','seed':44,'stage':'training','slurm_terminal':str(failed)}]
            write_json(index_path,{'evaluations':entries,'allocation_registry':registry})
            with patch('ours.heldout_comparison.check_cohort',return_value={'trials':trials}), \
                 patch('ours.heldout_comparison.close',side_effect=lambda p,s:deepcopy(closed[p])):
                report=build(cohort_path,index_path)
                self.assertEqual(report['status'],'COMPLETE_TWELVE_SCORES')
                stats=report['conditions'][0]['accuracy'];self.assertEqual(stats['mean'],.25);self.assertEqual(stats['sample_sd'],.25)
                delta=next(d for d in report['paired_seed_differences'] if d['treatment']=='whale' and d['control']=='weight_only')
                self.assertEqual(delta['accuracy_differences'],[3/64]*3);self.assertEqual(delta['sample_sd'],0)
                row=report['trials'][-1];self.assertEqual(row['known_gpu_hours_by_stage']['training'],2)
                self.assertIsNone(row['known_gpu_hours_by_stage']['search']);self.assertEqual(row['failed_terminal_allocations'],1)
                self.assertEqual(len(report['known_terminal_allocations']),13)
                export(report,root/'complete');self.assertIn('25.00 ± 25.00',(root/'complete/comparison.md').read_text())
                with self.assertRaisesRegex(ValueError,'Preserve'):export(report,root/'complete')
                entries[-1]={'condition':'whale','seed':44,'status':'INCOMPLETE','reason':'synthetic unfinished evaluation','evidence_paths':[str(failed)]}
                write_json(index_path,{'evaluations':entries,'allocation_registry':registry});partial=build(cohort_path,index_path)
                self.assertEqual(partial['status'],'INCOMPLETE_COMPARISON');self.assertIsNone(partial['trials'][-1]['solved'])
                self.assertIsNone(partial['conditions'][-1]['accuracy']['mean']);self.assertIsNone(partial['conditions'][-1]['accuracy']['sample_sd'])
                self.assertIsNone(partial['paired_seed_differences'][-1]['mean'])
                self.assertEqual(partial['trials'][-1]['known_gpu_hours_by_stage']['training'],2)
                export(partial,root/'partial');self.assertIn('missing | incomplete',(root/'partial/comparison.md').read_text())
                registry.append({**registry[0],'condition':'weight_only'})
                write_json(index_path,{'evaluations':entries,'allocation_registry':registry})
                with self.assertRaisesRegex(ValueError,'attribution'):build(cohort_path,index_path)
                registry.pop();entries[-1]['solved']=0
                write_json(index_path,{'evaluations':entries,'allocation_registry':registry})
                with self.assertRaisesRegex(ValueError,'no imputed score'):build(cohort_path,index_path)

    def test_cross_seed_or_unfinished_lineage_cannot_enter_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort_path=root/'cohort.json';write_json(cohort_path,{'fixture':True})
            evidence=root/'reason.txt';evidence.write_text('synthetic missing trial')
            trials=[{'condition':c,'seed':s,'status':'INCOMPLETE','evaluable':False} for c in CONDITIONS for s in SEEDS]
            entries=[{'condition':t['condition'],'seed':t['seed'],'status':'INCOMPLETE','reason':'not run','evidence_paths':[str(evidence)]} for t in trials]
            index=root/'index.json';write_json(index,{'evaluations':entries,'allocation_registry':[]})
            with patch('ours.heldout_comparison.check_cohort',return_value={'trials':trials}),patch('ours.heldout_comparison.close') as close:
                report=build(cohort_path,index);self.assertEqual(report['status'],'INCOMPLETE_COMPARISON');close.assert_not_called()
                entries[0]={'condition':'weight_only','seed':42,'status':'COMPLETE','result':'must-not-read.json'}
                write_json(index,{'evaluations':entries,'allocation_registry':[]})
                with self.assertRaisesRegex(ValueError,'Unfinished trial'):build(cohort_path,index)
                close.assert_not_called()
                entries[0]=entries[1]
                write_json(index,{'evaluations':entries,'allocation_registry':[]})
                with self.assertRaisesRegex(ValueError,'twelve'):build(cohort_path,index)


if __name__=='__main__':unittest.main()
