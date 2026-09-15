"""A transport repair cannot silently revise an experiment or reuse failed output."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ours.fast_chart_calibration_controller import verify_recovery
from ours.visual_task import file_sha256


class RecoveryTests(unittest.TestCase):
    def test_only_plan_and_output_paths_can_change(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);old_path=root/'old.json';new_path=root/'new.json'
            old={key:None for key in ('seed','rate','stage','harness_sha256','h0_calibration','parent_plan','selection',
                'resume_checkpoint','training_data','cpu_preflight','augmentation','model','worker_coordinates','bounds')}
            old.update(seed=42,rate=1e-7,stage=1,cpu_preflight={'next_four_batches':[['a','b']]},
                output=str(root/'old'),config={'output':str(root/'old/cache'),'plan':str(old_path),'batch':8})
            old_path.write_text(json.dumps(old))
            manifest=root/'recovery.json';manifest.write_text(json.dumps({'kind':'compact_native_transport_recovery',
                'retry_phase':'retry','failed_training_plan':str(old_path),
                'evidence_sha256':{str(old_path):file_sha256(old_path)}}))
            new=deepcopy(old);new['output']=str(root/'new')
            new['config'].update(output=str(root/'new/cache'),plan=str(new_path))
            new_path.write_text(json.dumps(new));verify_recovery(new_path,manifest)
            for mutation in ('rate','batch','order'):
                changed=deepcopy(new)
                if mutation=='rate':changed['rate']=1e-6
                elif mutation=='batch':changed['config']['batch']=4
                else:changed['cpu_preflight']['next_four_batches'][0].reverse()
                new_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):verify_recovery(new_path,manifest)


if __name__=='__main__':unittest.main()
