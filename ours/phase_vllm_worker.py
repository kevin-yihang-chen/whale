"""Verify the actual E4 checkpoint inside each MH serving worker after loading."""
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from .audit_native_training_batch import require
from .probe_updated_vllm import worker_embedding_samples
from .visual_task import file_sha256


class CheckpointVerifiedWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        require(not load_dummy_weights, 'A measured search requires real weights')
        super().load_model(load_dummy_weights=load_dummy_weights)
        plan_path = Path(os.environ['WHALE_MH_PLAN'])
        plan = json.loads(plan_path.read_text())
        coordinates = plan['worker_probe_coordinates']
        result = worker_embedding_samples(self.model_runner.get_model(), coordinates)
        require(result['model_class'] == 'Qwen3_5ForConditionalGeneration', 'Wrong target class')
        require(result['values'] == [p['updated'] for p in coordinates], 'MH worker has different weights')
        require(all(p['base'] != p['updated'] for p in coordinates), 'Nondistinguishing coordinates')
        with Path(os.environ['WHALE_MH_WORKER_RECEIPT']).open('x') as stream:
            json.dump({'status': 'PASS', 'plan_sha256': file_sha256(plan_path),
                       'actual_worker': result, 'changed_coordinates': coordinates}, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
