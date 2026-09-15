"""Observe native E4 weight transport inside the actual visual rollout worker."""
import json
import os
from pathlib import Path

from .native_visual_model_worker import NativeVisualModelWorker
from .probe_updated_vllm import worker_embedding_samples
from .visual_task import file_sha256


def observe_received_weights(model, plan, global_step):
    import torch
    assert global_step in (1, 2)
    directory = (Path(plan['resume_checkpoint']['directory']) if global_step == 1 else
                 Path(plan['output']) / 'checkpoints/global_step_2')
    path = directory / 'actor/model_world_size_1_rank_0.pt'
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    embedding = state['model.language_model.embed_tokens.weight']
    coordinates = plan['resume_coordinates']
    expected = [float(embedding[p['token_id'], p['column']].to(torch.bfloat16)) for p in coordinates]
    actual = worker_embedding_samples(model, coordinates)
    assert actual['values'] == expected and actual['embedding_dtype'] == 'torch.bfloat16'
    if global_step == 1:
        assert expected == [p['expected'] for p in coordinates]
        assert all(p['base'] != p['expected'] for p in coordinates)
    return {'status': 'PASS_NATIVE_RECEIVER_COORDINATES', 'global_step': global_step,
        'native_checkpoint_sha256': file_sha256(path), 'actual': actual, 'expected': expected,
        'model_calls': 0, 'optimizer_steps': 0,
        'limitation': 'Eight receiver coordinates, not a full in-memory weight comparison or performance result.'}


class VisualResumeModelWorker(NativeVisualModelWorker):
    def record_native_resumed_weights(self, global_step):
        path = Path(os.environ['VETO_VISUAL_RESUME_PLAN'])
        plan = json.loads(path.read_text())
        assert plan['kind'] == 'native_visual_rsft_resume'
        assert type(self.model_runner.get_model()).__name__ == 'Qwen3_5ForConditionalGeneration'
        report = observe_received_weights(self.model_runner.get_model(), plan, global_step)
        report.update(plan_sha256=file_sha256(path), job_id=os.environ['SLURM_JOB_ID'])
        output = Path(plan['output']) / 'resume' / f'receiver-step{global_step}-{os.getpid()}.json'
        with output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return report['status']
