"""Release idle CuPy transport allocations after native E4 weight synchronization.

WHALE's original finalize drops both buffers but only empties PyTorch's cache.
The subclass also releases unused CuPy blocks after device synchronization.
Live allocations, weight values, gradients and optimizer state are untouched.
"""
import json
import os

from .audit_native_training_batch import require


def install_transport_release():
    import cupy as cp
    from verl.checkpoint_engine import CheckpointEngineRegistry
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
    original = CheckpointEngineRegistry.get('nccl')
    if getattr(original, '_whale_release_idle_cupy', False):
        return original
    require(original is NCCLCheckpointEngine, 'Unexpected NCCL engine registration')

    class ReclaimedNCCLCheckpointEngine(original):
        _whale_release_idle_cupy = True

        def finalize(self):
            # The original manager awaits all send/receive RPCs before finalize.
            super().finalize()
            if self.is_master:
                cp.cuda.Device().synchronize()
                pool = cp.get_default_memory_pool()
                before = {'used': pool.used_bytes(), 'total': pool.total_bytes(), 'free': pool.free_bytes()}
                free_before, total = cp.cuda.runtime.memGetInfo()
                pool.free_all_blocks()
                after = {'used': pool.used_bytes(), 'total': pool.total_bytes(), 'free': pool.free_bytes()}
                free_after, _ = cp.cuda.runtime.memGetInfo()
                require(after['used'] == before['used'], 'Live CuPy allocation accounting changed')
                print(json.dumps({'kind': 'native_transport_idle_memory_release',
                    'job_id': os.environ.get('SLURM_JOB_ID'), 'pid': os.getpid(),
                    'pool_before_bytes': before, 'pool_after_bytes': after,
                    'released_pool_bytes': before['total'] - after['total'],
                    'device_free_before_bytes': free_before, 'device_free_after_bytes': free_after,
                    'device_total_bytes': total, 'bucket_bytes': self.bucket_size}), flush=True)

    CheckpointEngineRegistry.register('nccl')(ReclaimedNCCLCheckpointEngine)
    return ReclaimedNCCLCheckpointEngine


def gpu_probe():
    """Exercise real native buffer allocation/finalize; this is not an NCCL transfer."""
    import cupy as cp
    engine_class = install_transport_release()
    engine = engine_class.__new__(engine_class)
    engine.is_master, engine.rebuild_group = True, False
    engine.bucket_size, engine.ip, engine.listen_port = 32 << 20, 'fixture', 0
    sentinel = cp.arange(17, dtype=cp.int64)
    engine.prepare()
    before = cp.get_default_memory_pool().total_bytes()
    engine.finalize()
    released = before - cp.get_default_memory_pool().total_bytes()
    require(engine.send_buf is None and engine.recv_buf is None, 'Native buffers remain live')
    require(bool(cp.array_equal(sentinel, cp.arange(17, dtype=cp.int64))), 'Live sentinel changed')
    require(released >= 2 * engine.bucket_size, 'Native fixture buffers were not released')
    print(json.dumps({'kind': 'native_transport_gpu_fixture', 'status': 'PASS',
                      'released_bytes': released, 'real_nccl_transfer': False}), flush=True)


if __name__ == '__main__':
    gpu_probe()
