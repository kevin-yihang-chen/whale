"""Shared visual tool execution before VETO E1--E3 and original WHALE E4.

Reuse the pinned ToolAgentLoop and ImageZoomInTool. Crops use the exact initial
image seen by the model, with native bounding-box handling and feedback. The
final assistant span supplies r=v(answer, truth); native tool/image masks remain
zero. This is a shared domain adapter, not a new optimization objective or the
complete executable-harness search integration.
"""
from uuid import uuid4

from PIL import Image

from .training_bootstrap import prepare_worker
from .visual_evidence_reward import compute_score

prepare_worker()
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.image_zoom_in_tool import ImageZoomInTool
from verl.tools.schemas import ToolResponse


class VisualEvidenceZoomTool(ImageZoomInTool):
    """Native crop execution on model-visible pixels, without refetch/resizing.

    Native execute/release are retained. Their local PIL operation does not use
    the upstream constructor's Ray execution pool. Image sources are supplied by
    the loop, not by candidate paths, URLs, labels or dataset metadata.
    """
    def __init__(self, config, tool_schema):
        BaseTool.__init__(self, config, tool_schema)
        self._instance_dict = {}

    async def create(self, instance_id=None, **kwargs):
        image = kwargs.get('create_kwargs', {}).get('image')
        if not isinstance(image, Image.Image):
            raise ValueError('Visual crop requires the model-visible image')
        instance_id = instance_id or str(uuid4())
        self._instance_dict[instance_id] = {'image': image.copy(), 'response': '', 'reward': 0.}
        return instance_id, ToolResponse()


@register('visual_evidence_tool_agent')
class VisualEvidenceToolAgentLoop(ToolAgentLoop):
    """Keep native multi-turn execution and score the final assistant only.

    Training and future evaluation must invoke this same run method. All visual
    conditions share its tool, final-answer verifier and masks. Executable visual
    candidate dispatch and VETO selection are separate integration work.
    """
    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        if tool_call.name == 'image_zoom_in_tool':
            images = agent_data.image_data or []
            tools_kwargs = {**tools_kwargs, tool_call.name: {
                'create_kwargs': {'image': images[0] if images else None}}}
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def run(self, sampling_params, **kwargs):
        source = kwargs['data_source']
        truth = kwargs['reward_model']['ground_truth']
        compute_score(source, '', truth)  # Reject invalid task data before generation.
        output = await super().run(sampling_params, **kwargs)
        mask = output.response_mask
        if len(mask) != len(output.response_ids):
            raise ValueError('Native visual response tokens and masks differ')
        start = len(mask)
        while start and mask[start - 1] == 1:
            start -= 1
        final_ids = output.response_ids[start:] if start < len(mask) else []
        answer = self.tokenizer.decode(final_ids, skip_special_tokens=True)
        scores = compute_score(source, answer, truth)
        output.reward_score = scores['score']
        output.extra_fields.update(visual_final_answer=answer, reward_extra_info=scores)
        return output
