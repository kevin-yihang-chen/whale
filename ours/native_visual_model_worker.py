"""Observe loaded VLM parameters inside the real native vLLM worker.

File manifests certify source bytes; these coordinates independently check
actual service loading. Coordinate checks are not a full in-memory weight hash.
"""
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from .audit_native_training_batch import require
from .probe_updated_vllm import worker_embedding_samples
from .visual_task import file_sha256


class NativeVisualModelWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        require(not load_dummy_weights, 'Visual evaluation requires real model weights')
        super().load_model(load_dummy_weights=load_dummy_weights)
        path = Path(os.environ['VETO_NATIVE_EVALUATION_PLAN'])
        plan = json.loads(path.read_text())
        actual = worker_embedding_samples(self.model_runner.get_model(), plan['worker_coordinates'])
        require(actual['model_class'] == 'Qwen3_5ForConditionalGeneration' and
            actual['values'] == [p['expected'] for p in plan['worker_coordinates']], 'Different loaded visual model')
        require(Path(self.vllm_config.model_config.model).resolve() == Path(plan['model']['path']).resolve(),
            'Different actual model load path')
        directory = Path(os.environ['VETO_NATIVE_EVALUATION_OUTPUT'])/'workers'
        with (directory/f'model-{os.getpid()}.json').open('x') as stream:
            json.dump({'status': 'PASS_LOADED_COORDINATES', 'plan_sha256': file_sha256(path),
                'actual_worker': actual, 'model_path': self.vllm_config.model_config.model,
                'seed': self.vllm_config.model_config.seed,
                'tensor_parallel_size': self.vllm_config.parallel_config.tensor_parallel_size,
                'limitation': 'Coordinate checks plus complete source-file hashes; not a full in-memory parameter comparison.'}, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
