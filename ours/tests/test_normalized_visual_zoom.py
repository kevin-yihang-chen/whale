import asyncio
import unittest

from PIL import Image
from ours.normalized_visual_zoom import NormalizedEvidenceZoomTool, normalized_to_pixels
from verl.tools.schemas import OpenAIFunctionToolSchema


class NormalizedZoomTest(unittest.TestCase):
    def test_nonsquare_scale_and_invalid_coordinates(self):
        self.assertEqual(normalized_to_pixels([500,100,750,900], (384,320)), [192.,32.,288.,288.])
        for box in ([0,0,0,4], [False,0,10,10], [0,0,1001,500], [0,0,float('nan'),500]):
            with self.assertRaises(ValueError):
                normalized_to_pixels(box, (384,320))

    def test_returned_pixels_match_region_in_initial_visible_image(self):
        schema = OpenAIFunctionToolSchema.model_validate({'type':'function', 'function':{
            'name':'image_zoom_in_tool', 'description':'normalized coordinates',
            'parameters':{'type':'object','properties':{},'required':[]}}})
        tool = NormalizedEvidenceZoomTool({}, schema)
        image = Image.new('RGB', (384,320))
        image.putdata([(x%256,y%256,(x+y)%256) for y in range(320) for x in range(384)])
        async def check():
            instance, _ = await tool.create(create_kwargs={'image':image})
            result, _, info = await tool.execute(instance, {'bbox_2d':[500,100,750,900]})
            self.assertTrue(info['success'])
            self.assertEqual(info['pixel_bbox'], [192.,32.,288.,288.])
            self.assertEqual(result.image[0].tobytes(), image.crop((192,32,288,288)).tobytes())
            await tool.release(instance)
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
