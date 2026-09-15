"""E4 transport preparation: make idle allocator memory available before NCCL.

The native trainer already clears gradients after SFT. Its PyTorch allocator
can retain freed blocks while the NCCL engine requests separate CuPy buffers.
Release only idle blocks at this synchronized boundary; retain the original
FP32 tensors, optimizer state, bucket size and native transfer implementation.
"""
import json
import os
from pathlib import Path

from .visual_task import file_sha256


def release_idle_before_prepare(torch, cp):
    torch.cuda.synchronize()
    cp.cuda.Device().synchronize()
    pool = cp.get_default_memory_pool()
    before = {'torch_live_bytes':torch.cuda.memory_allocated(),
        'torch_reserved_bytes':torch.cuda.memory_reserved(),
        'cupy_live_bytes':pool.used_bytes(),'cupy_reserved_bytes':pool.total_bytes(),
        'device_free_bytes':cp.cuda.runtime.memGetInfo()[0]}
    torch.cuda.empty_cache()
    pool.free_all_blocks()
    after = {'torch_live_bytes':torch.cuda.memory_allocated(),
        'torch_reserved_bytes':torch.cuda.memory_reserved(),
        'cupy_live_bytes':pool.used_bytes(),'cupy_reserved_bytes':pool.total_bytes(),
        'device_free_bytes':cp.cuda.runtime.memGetInfo()[0]}
    if any(before[key]!=after[key] for key in ('torch_live_bytes','cupy_live_bytes')):
        raise ValueError('Live allocations changed at the idle transport boundary')
    return {'before':before,'after':after,
        'released_reserved_bytes':sum(before[k]-after[k] for k in ('torch_reserved_bytes','cupy_reserved_bytes'))}


def install_transport_preparation():
    import torch
    import cupy as cp
    from verl.checkpoint_engine import CheckpointEngineRegistry as registry
    original = registry.get('nccl')
    if getattr(original,'_compact_idle_transport_prepare',False):return original
    if not getattr(original,'_whale_release_idle_cupy',False):
        raise ValueError('Compact preparation requires the preserved native finalizer')

    class CompactPreparedNCCLCheckpointEngine(original):
        _compact_idle_transport_prepare = True

        def prepare(self):
            if self.is_master:
                if getattr(self,'send_buf',None) is not None or getattr(self,'recv_buf',None) is not None:
                    raise ValueError('Cannot prepare while prior transport buffers remain active')
                path=Path(os.environ['VETO_COMPACT_TRAINING_PLAN'])
                plan=json.loads(path.read_text())
                record={'event':'COMPACT_IDLE_MEMORY_RELEASE_BEFORE_NATIVE_PREPARE',
                    'plan_sha256':file_sha256(path),'pid':os.getpid(),'job_id':os.environ.get('SLURM_JOB_ID'),
                    'bucket_bytes':self.bucket_size,**release_idle_before_prepare(torch,cp)}
                directory=Path(plan['output'])/'state'
                with (directory/f'transport-memory-{os.getpid()}.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
            return super().prepare()

    registry.register('nccl')(CompactPreparedNCCLCheckpointEngine)
    return CompactPreparedNCCLCheckpointEngine
