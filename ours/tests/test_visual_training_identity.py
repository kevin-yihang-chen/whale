"""Exercise native reward-handle collation with a scripted visual output."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class VisualTrainingIdentityTest(unittest.TestCase):
    def test_identity_survives_native_training_metadata_filter_without_changing_tensors(self):
        from ours.training_bootstrap import prepare_worker
        prepare_worker()
        import numpy as np
        import torch
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, AgentLoopOutput, AgentLoopMetrics, _InternalAgentLoopOutput
        from ours.visual_harness_loop import VisualHarnessAgentLoop
        from ours.visual_training_identity import IdentifiedVisualHarnessAgentLoop
        original = AgentLoopOutput(prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 1],
            reward_score=1., metrics=AgentLoopMetrics(), extra_fields={'visual_committed_answer': 'A'})
        loop = IdentifiedVisualHarnessAgentLoop.__new__(IdentifiedVisualHarnessAgentLoop)
        with patch.object(VisualHarnessAgentLoop, 'run', new=AsyncMock(return_value=original)) as parent:
            output = asyncio.run(loop.run({'temperature': 1.}, visual_sample_id='synthetic-sample'))
        self.assertIs(output, original)
        parent.assert_awaited_once_with({'temperature': 1.}, visual_sample_id='synthetic-sample')
        internal = _InternalAgentLoopOutput(prompt_ids=torch.tensor([[1]]), response_ids=torch.tensor([[2, 3]]),
            input_ids=torch.tensor([[1, 2, 3]]), response_mask=torch.tensor([[1, 1]]),
            attention_mask=torch.tensor([[1, 1, 1]]), position_ids=torch.tensor([[0, 1, 2]]),
            reward_score=1., metrics=AgentLoopMetrics(), extra_fields=output.extra_fields,
            multi_modal_inputs={'pixel_values': torch.arange(12).reshape(4, 3)})
        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.reward_loop_worker_handles = [object()]  # Activate the real training-only metadata branch.
        batch = worker._postprocess([internal], input_non_tensor_batch={
            'visual_sample_id': np.asarray(['synthetic-sample'], dtype=object),
            'unrequested_metadata': np.asarray(['must-stay-absent'], dtype=object)})
        self.assertEqual(batch.non_tensor_batch['visual_sample_id'].tolist(), ['synthetic-sample'])
        self.assertNotIn('unrequested_metadata', batch.non_tensor_batch)
        self.assertTrue(torch.equal(batch.batch['response_mask'], internal.response_mask))
        self.assertTrue(torch.equal(batch.non_tensor_batch['multi_modal_inputs'][0]['pixel_values'],
                                    internal.multi_modal_inputs['pixel_values']))
        self.assertEqual(batch.batch['rm_scores'].sum().item(), 1.)


if __name__ == '__main__':
    unittest.main()
