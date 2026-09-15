"""Native finalize ordering and live-allocation protection (CPU boundary fixtures)."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('cupy'), 'Requires native training runtime')
class NativeTransportMemoryTests(unittest.TestCase):
    def test_original_finalize_then_release_unused_pool_and_preserve_live_allocation(self):
        import cupy as cp
        import torch
        from verl.checkpoint_engine import CheckpointEngineRegistry as registry
        from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
        from ours.native_transport_memory import install_transport_release
        original = registry.get('nccl')
        registry.register('nccl')(NCCLCheckpointEngine)
        events, released = [], []
        engine = None
        def synchronize():
            self.assertIsNone(engine.send_buf)
            self.assertIsNone(engine.recv_buf)
            events.append('synchronize')
        def release():
            self.assertEqual(events, ['native_torch_empty_cache', 'synchronize'])
            released.append(True)
        pool = SimpleNamespace(used_bytes=lambda: 19, total_bytes=lambda: 19 if released else 219,
                               free_bytes=lambda: 0 if released else 200, free_all_blocks=release)
        try:
            cls = install_transport_release()
            self.assertIs(cls, install_transport_release())
            engine = cls.__new__(cls)
            engine.is_master, engine.rebuild_group, engine.bucket_size = True, False, 100
            engine.send_buf, engine.recv_buf = object(), object()
            with patch.object(torch.cuda, 'empty_cache', lambda: events.append('native_torch_empty_cache')), \
                 patch.object(cp.cuda, 'Device', return_value=SimpleNamespace(synchronize=synchronize)), \
                 patch.object(cp, 'get_default_memory_pool', return_value=pool), \
                 patch.object(cp.cuda.runtime, 'memGetInfo', side_effect=[(500, 1000), (700, 1000)]):
                engine.finalize()
            self.assertEqual(pool.used_bytes(), 19)
            self.assertEqual(released, [True])
        finally:
            registry.register('nccl')(original)

    def test_receiver_keeps_original_torch_finalization(self):
        import cupy as cp
        import torch
        from verl.checkpoint_engine import CheckpointEngineRegistry as registry
        from ours.native_transport_memory import install_transport_release
        original = registry.get('nccl')
        try:
            cls = install_transport_release()
            engine = cls.__new__(cls)
            engine.is_master, engine.rebuild_group = False, False
            engine.send_buf, engine.recv_buf = object(), object()
            with patch.object(torch.cuda, 'empty_cache') as empty, \
                 patch.object(cp, 'get_default_memory_pool') as pool:
                engine.finalize()
            empty.assert_called_once()
            pool.assert_not_called()
            self.assertIsNone(engine.send_buf)
        finally:
            registry.register('nccl')(original)


if __name__ == '__main__':
    unittest.main()
