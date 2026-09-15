"""Numerical edits, group overlap and independent saved-pixel oracle checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from ours.plotqa_evidence_pairs import canonical_table, comparison, draw, exchange, pixel_values, truth


class PlotQAEvidenceTests(unittest.TestCase):
    def chart(self):
        return {'title': 'Annual values', 'series_names': ['North', 'South'],
            'categories': ['2020', '2021', '2022'], 'values': [[7,3,0],[2,11,4]],
            'x_label': 'Year', 'y_label': 'Count'}

    def test_two_cell_exchange_changes_truth_without_other_edits(self):
        chart = self.chart()
        cells = comparison(chart, canonical_table(chart))
        changed = exchange(chart['values'], cells)
        self.assertNotEqual(truth(chart['values'], cells), truth(changed, cells))
        differences = {(r,c) for r,row in enumerate(changed) for c,v in enumerate(row) if v != chart['values'][r][c]}
        self.assertEqual(differences, {tuple(cell) for cell in cells})

    def test_grouping_ignores_title_and_display_order_but_preserves_named_values(self):
        chart = self.chart()
        other = deepcopy(chart)
        other['title'] = 'Different rendering title'
        other['categories'].reverse()
        other['series_names'].reverse()
        other['values'] = [list(reversed(values)) for values in reversed(other['values'])]
        self.assertEqual(canonical_table(chart), canonical_table(other))
        other['values'][0][0] += 1
        self.assertNotEqual(canonical_table(chart), canonical_table(other))

    def test_saved_pixels_recover_original_and_exchanged_answers_including_zero_bar(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Requires the plotting environment')
        chart = self.chart()
        cells = [[0,0],[1,1]]
        with tempfile.TemporaryDirectory() as tmp:
            for side, values in enumerate((chart['values'], exchange(chart['values'], cells))):
                path = Path(tmp)/f'{side}.png'
                geometry = draw(chart, values, path)
                recovered = pixel_values(path, geometry)
                self.assertEqual(truth(values,cells), truth(recovered,cells))
                for row, decoded in zip(values,recovered):
                    for actual, estimate in zip(row,decoded):
                        self.assertLessEqual(abs(actual-estimate), 2*geometry['units_per_pixel'])


if __name__ == '__main__':
    unittest.main()
