"""Shared Qwen coordinate convention for the visible-image crop tool.

Coordinate conversion: (x1,y1,x2,y2)_px = bbox_1000 * (W,H,W,H)/1000.
This is the E1/E4 task adapter, not a VETO acceptance mechanism. The convention
is documented by Qwen-Agent's image_zoom_in_qwen3vl.py, lines 143--147.
"""
import math

from .visual_evidence_tool_loop import VisualEvidenceZoomTool
from verl.tools.schemas import ToolResponse


def normalized_to_pixels(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('bbox_2d must contain four normalized coordinates')
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000 for x in box):
        raise ValueError('bbox_2d coordinates must be finite numbers between 0 and 1000')
    if not box[0] < box[2] or not box[1] < box[3]:
        raise ValueError('bbox_2d must have positive width and height')
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('A nonempty model-visible image is required')
    return [x * scale / 1000.0 for x, scale in zip(box, (width, height, width, height))]


class NormalizedEvidenceZoomTool(VisualEvidenceZoomTool):
    async def execute(self, instance_id, parameters, **kwargs):
        image = self._instance_dict[instance_id]['image']
        try:
            pixel_box = normalized_to_pixels(parameters.get('bbox_2d'), image.size)
        except ValueError as error:
            return ToolResponse(text=f'Error: {error}.'), -0.05, {'success': False}
        response, reward, details = await super().execute(
            instance_id, {**parameters, 'bbox_2d': pixel_box}, **kwargs)
        return response, reward, {**details, 'normalized_bbox': parameters['bbox_2d'],
                                   'pixel_bbox': pixel_box, 'initial_image_size': list(image.size)}
