"""Carry sample identity across the native reward-worker E4 return path.

The native worker omits input metadata when reward-worker handles are present.
Publish only the sample identifier as an output field; generation, pixels,
rewards, masks and original success filtering remain unchanged.
"""
from .visual_harness_loop import VisualHarnessAgentLoop
from verl.experimental.agent_loop.agent_loop import register


@register('visual_training_agent')
class IdentifiedVisualHarnessAgentLoop(VisualHarnessAgentLoop):
    async def run(self, sampling_params, **kwargs):
        sample_id = kwargs['visual_sample_id']
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError('A visual training trajectory needs its input sample identity')
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields['visual_sample_id'] = sample_id
        return output
