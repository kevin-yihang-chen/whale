"""Exercise canonical-to-native transition using small real serialized tensors."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native tensor runtime')
class AlternationTransitionTests(unittest.TestCase):
    def test_real_checkpoint_files_separate_cast_only_updates_and_reject_graph_drift(self):
        import torch
        from safetensors.torch import save_file
        from ours.verify_alternation_transition import compare_active_parameters
        with tempfile.TemporaryDirectory(prefix='whale-transition-fixture-') as root:
            base, native, exported = [Path(root) / name for name in ('base', 'native', 'exported')]
            for path in (base, native, exported):
                path.mkdir()
            (native / 'fsdp_config.json').write_text(json.dumps({'world_size': 1}))
            incoming = torch.tensor([1., 0.], dtype=torch.bfloat16)
            save_file({'embedding': incoming}, str(base / 'model.safetensors'))
            def record(value, *, wrong_head=False, extra=False):
                state = {'embedding': value, 'lm_head': value.clone()}
                if wrong_head:
                    state['lm_head'][0] += 1
                if extra:
                    state['unreviewed'] = torch.ones(2)
                torch.save(state, native / 'model_world_size_1_rank_0.pt')
                save_file({'embedding': value.bfloat16()}, str(exported / 'model.safetensors'))
            def compare():
                return compare_active_parameters(base, native, exported,
                    aliases={'lm_head': 'embedding'}, tensor_count=1)
            record(incoming.float())
            result = compare()
            self.assertEqual(result['native_fp32']['status'], 'UNCHANGED')
            self.assertEqual(result['exported_bf16']['status'], 'UNCHANGED')
            record(incoming.float() + torch.tensor([1e-7, 1e-7]))
            result = compare()
            self.assertEqual(result['native_fp32']['changed_elements'], 2)
            self.assertEqual(result['exported_bf16']['changed_elements'], 1)
            record(incoming.float(), wrong_head=True)
            with self.assertRaisesRegex(ValueError, 'alias values'):
                compare()
            record(incoming.float(), extra=True)
            with self.assertRaisesRegex(ValueError, 'missing or extra'):
                compare()
            record(incoming.float())
            save_file({'embedding': torch.zeros_like(incoming)}, str(exported / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'Export differs'):
                compare()


if __name__ == '__main__':
    unittest.main()
