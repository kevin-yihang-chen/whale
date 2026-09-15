"""Exercise native RPC encoding and actual BF16 reads without allocating a GPU."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('vllm'), 'Requires native vLLM runtime')
class WeightProbeWorkerTests(unittest.TestCase):
    def test_native_rpc_roundtrip_reads_model_not_expected_coordinate_values(self):
        import torch
        from vllm import envs
        from vllm.utils.import_utils import resolve_obj_by_qualname
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult
        from vllm.v1.worker.gpu_worker import Worker
        from ours.vllm_weight_probe_worker import WeightProbeWorkerExtension

        self.assertFalse(envs.VLLM_ALLOW_INSECURE_SERIALIZATION)
        self.assertIs(resolve_obj_by_qualname('ours.vllm_weight_probe_worker.WeightProbeWorkerExtension'),
                      WeightProbeWorkerExtension)
        self.assertFalse(hasattr(Worker, 'sample_updated_embeddings'))
        actual = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
        class LoadedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(actual)
            def embed_input_ids(self, ids):
                return self.embedding(ids)
        worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: LoadedModel()))
        coords = [{'token_id': 1, 'column': 0}, {'token_id': 0, 'column': 1}]
        request = ('sample_updated_embeddings', 60., (), {'coordinates': coords})
        decoded = MsgpackDecoder().decode(MsgpackEncoder().encode(request))
        method, _, args, kwargs = decoded
        result = getattr(WeightProbeWorkerExtension, method)(worker, *args, **kwargs)
        response = MsgpackDecoder(UtilityResult).decode(MsgpackEncoder().encode(UtilityResult([result])))
        self.assertEqual(response.result[0]['values'], [3., 2.])
        self.assertEqual(response.result[0]['embedding_dtype'], 'torch.bfloat16')
        # The complete observed result is also suitable for the JSON evidence log.
        self.assertEqual(json.loads(json.dumps(response.result)), response.result)


if __name__ == '__main__':
    unittest.main()
