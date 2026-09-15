"""Source corruption, non-finite exclusions and style-group leakage checks."""
from copy import deepcopy
import io
import json
import unittest

from ours.visual_source_inventory import array_records, chart_schema, finite_tree, source_group


class SourceInventoryTests(unittest.TestCase):
    def test_chunks_do_not_change_source_record_identity(self):
        text = ' [ {"label":"a ] \\\" 中文", "x":[1,2]}, {"y":NaN} ] \n'
        expected = list(array_records(io.StringIO(text), chunk_size=512))
        self.assertEqual(list(array_records(io.StringIO(text), chunk_size=1))[0], expected[0])
        self.assertEqual(len(expected), 2)
        self.assertTrue(finite_tree(expected[0][1]))
        self.assertFalse(finite_tree(expected[1][1]))
        self.assertEqual(list(array_records(io.StringIO(text), chunk_size=1))[1][2], expected[1][2])

    def test_syntax_failure_is_not_a_rejected_scientific_sample(self):
        for raw in ('[{"x":1}', '[{},]', '[{}]garbage', '[{} {}]', '[0]', '{}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                list(array_records(io.StringIO(raw), chunk_size=1))
        with self.assertRaisesRegex(ValueError, 'bounded parser memory|Oversized'):
            list(array_records(io.StringIO('[{"x":"'+100*'a'+'"}]'), chunk_size=4, maximum_record_chars=24))

    def test_rendering_styles_and_series_order_share_a_group(self):
        row = {'general_figure_info': {'title': {'text': 'Series data'}},
            'models': [{'name': 'north', 'x': [0, 1], 'y': [10, 20], 'color': 'red'},
                       {'name': 'south', 'x': [0, 1], 'y': [20, 30], 'color': 'blue'}]}
        changed = deepcopy(row)
        changed['models'].reverse()
        changed['models'][0]['color'] = 'green'
        self.assertEqual(source_group(row), source_group(changed))
        changed['models'][0]['y'][0] += 1
        self.assertNotEqual(source_group(row), source_group(changed))

    def test_official_string_categories_align_with_numeric_tick_positions(self):
        row = {'type': 'vbar_categorical', 'models': [
            {'name': 'Female', 'x': ['1980', '1990', '2000'], 'y': [88.04653, 88.21484, 93.48526]}],
            'general_figure_info': {'title': {'text': 'Population'},
                'x_axis': {'major_labels': {'values': ['1980', '1990', '2000']},
                    'major_ticks': {'values': [0, 1, 2]}, 'label': {'text': 'Year'}},
                'y_axis': {'label': {'text': 'Percent'}}}}
        self.assertEqual(chart_schema(row)['categories'], ['1980', '1990', '2000'])
        changed = deepcopy(row)
        changed['models'][0]['x'][0] = '1970'
        with self.assertRaisesRegex(ValueError, 'differ_from_display'):
            chart_schema(changed)


if __name__ == '__main__':
    unittest.main()
