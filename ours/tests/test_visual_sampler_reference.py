from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class SamplerReferenceTests(unittest.TestCase):
    def test_references_follow_actual_checkpoint_and_leave_global_rng_unchanged(self):
        import torch
        from torchdata.stateful_dataloader import StatefulDataLoader
        from torch.utils.data import RandomSampler
        from ours.visual_sampler_reference import restore_next_batch
        def loader(seed):
            data = list(range(32))
            sampler = RandomSampler(data, generator=torch.Generator().manual_seed(seed))
            return StatefulDataLoader(data, batch_size=8, sampler=sampler, num_workers=0)
        with TemporaryDirectory() as folder:
            observed = []
            for seed in (42, 43):
                running = loader(seed)
                iterator = iter(running)
                first = next(iterator)
                path = Path(folder) / f'{seed}.pt'
                torch.save(running.state_dict(), path)
                expected = next(iterator)
                rng = torch.get_rng_state().clone()
                # StatefulDataLoader may consume a base seed during iteration;
                # use a fork here so the probe does not alter caller randomness.
                with torch.random.fork_rng(devices=[]):
                    actual, receipt = restore_next_batch(loader(seed), path)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertTrue(torch.equal(actual, expected))
                self.assertFalse(set(first.tolist()) & set(actual.tolist()))
                observed.append(receipt['saved_state_sha256'])
            self.assertNotEqual(*observed)


if __name__ == '__main__':
    unittest.main()
