"""Idle allocator release must precede native buffers and preserve live storage."""
from types import SimpleNamespace
import unittest
from ours.fast_chart_transport import release_idle_before_prepare


class CompactTransportTests(unittest.TestCase):
    def fixture(self, change_live=False):
        events=[];state={'torch_reserved':90,'cupy_reserved':20,'live':60}
        def torch_release():
            events.append('torch_release');state['torch_reserved']=60
            if change_live:state['live']=59
        def cupy_release():events.append('cupy_release');state['cupy_reserved']=5
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('torch_sync'),
            empty_cache=torch_release,memory_allocated=lambda:state['live'],memory_reserved=lambda:state['torch_reserved']))
        pool=SimpleNamespace(used_bytes=lambda:5,total_bytes=lambda:state['cupy_reserved'],free_all_blocks=cupy_release)
        cp=SimpleNamespace(get_default_memory_pool=lambda:pool,cuda=SimpleNamespace(
            Device=lambda:SimpleNamespace(synchronize=lambda:events.append('cupy_sync')),
            runtime=SimpleNamespace(memGetInfo=lambda:(150-state['torch_reserved']-state['cupy_reserved'],150))))
        return torch,cp,events

    def test_release_synchronizes_then_returns_idle_memory_without_touching_live_bytes(self):
        torch,cp,events=self.fixture()
        result=release_idle_before_prepare(torch,cp)
        self.assertEqual(events,['torch_sync','cupy_sync','torch_release','cupy_release'])
        self.assertEqual(result['released_reserved_bytes'],45)
        self.assertEqual(result['before']['torch_live_bytes'],result['after']['torch_live_bytes'])
        self.assertEqual(result['before']['cupy_live_bytes'],result['after']['cupy_live_bytes'])

    def test_live_allocation_change_is_rejected(self):
        torch,cp,_=self.fixture(change_live=True)
        with self.assertRaisesRegex(ValueError,'Live allocations changed'):
            release_idle_before_prepare(torch,cp)


if __name__=='__main__':unittest.main()
