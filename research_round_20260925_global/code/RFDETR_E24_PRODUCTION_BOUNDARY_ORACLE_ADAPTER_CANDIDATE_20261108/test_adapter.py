"""Synthetic production-support and archive-binding checks."""
import json
from types import SimpleNamespace
import unittest

import numpy as np

import adapter
from adapter import support_masks


def clipped_dilation(np_api, mask):
    height, width = mask.shape
    padded = np_api.pad(mask, 2, constant_values=False)
    return np_api.logical_or.reduce([padded[y:y+height, x:x+width]
                                     for y in range(5) for x in range(5)])


class ProductionBoundaryTests(unittest.TestCase):
    def test_predicted_band_does_not_look_at_gt_and_clips_image_border(self):
        gi = np.zeros((7, 7), dtype=bool)
        gi[3, 3] = True
        gf = gi.copy()
        gf[3, 4] = True
        bands = support_masks(np, gi, gf, gi, clipped_dilation)
        self.assertTrue(bands['predicted_band'][3, 4])
        self.assertFalse(bands['gt_band'][3, 4])
        swapped = support_masks(np, gi, gi, gi, clipped_dilation)
        self.assertTrue(np.array_equal(swapped['predicted_band'], bands['predicted_band']))
        full_prediction = np.ones_like(gi)
        full = support_masks(np, gi, gf, full_prediction, clipped_dilation)
        self.assertFalse(full['predicted_band'].any())


    def test_assigned_rank_and_archived_best_identity_are_rechecked_without_search(self):
        gi = np.zeros((7, 7), dtype=bool)
        gi[2:5, 2:5] = True
        gf = gi.copy()
        gf[2, 5] = True
        pred = np.zeros_like(gi)
        pred[2:4, 2:5] = True
        pred[1, 2] = True
        sizes = [7, 7]
        integer, floating = adapter.VARIANTS
        rank = 5
        q = 7
        raw = dict(image_id=9, original_hw=sizes,
                   raw_masks=[dict(size=sizes, counts='bad') for _ in range(100)],
                   slot0_native_ranks=list(range(101, 201)), retained=[dict(native_rank=rank, query_id=q)])
        raw['raw_masks'][q] = dict(size=sizes, counts='selected')
        raw['slot0_native_ranks'][q] = rank
        anchor = dict(image_id=9, GT_annotation_id=17, assigned_native_rank=rank, assigned_query_id=q,
                      by_GT={integer: dict(raw=6/10, retained=6/10, assigned=6/10),
                             floating: dict(raw=6/11, retained=6/11, assigned=6/11)})
        decoder_calls = []

        def decode(rle):
            if isinstance(rle, dict):
                decoder_calls.append(rle['counts'])
                self.assertEqual(rle['counts'], b'selected')
                return pred
            return rle

        def reader(mask):
            annotation = dict(image_id=9, category_id=1, area=1024, iscrowd=0)
            return SimpleNamespace(anns={17: annotation}, annToRLE=lambda _: mask)

        modules = dict(localization=SimpleNamespace(binary_mask=lambda _, value: value,
                                                     dilate_chebyshev=clipped_dilation,
                                                     signature=lambda obj: json.dumps(obj, sort_keys=True)),
                       mask_api=SimpleNamespace(decode=decode),
                       signed_adapter=SimpleNamespace(area_group=lambda _: 'medium'),
                       oracle_math=SimpleNamespace(signed_edit_oracle=lambda i, f, fn, fp: {'capacity': [fn, fp]}))
        cocos = {integer: reader(gi), floating: reader(gf)}
        first = adapter.build_pair(np, anchor, None, raw, cocos, modules)
        self.assertEqual(first['assigned']['overlap_counts'], {integer: (6, 10), floating: (6, 11)})
        self.assertIsNone(first['best'])
        assigned = first['assigned']
        previous = dict(image_id=9, GT_annotation_id=17, selected_raw_query_id=q,
                        assigned_native_rank=rank, assigned_query_id=q, original_hw=sizes,
                        integer_annotation_area=1024, area_group='medium',
                        overlap_counts=assigned['overlap_counts'], baseline_pass=assigned['baseline_pass'],
                        capacities={'band': assigned['capacities']['gt_band']},
                        raster_sha256=assigned['raster_sha256'],
                        regions={'band': assigned['regions']['gt_band']})
        both = adapter.build_pair(np, anchor, previous, raw, cocos, modules)
        self.assertEqual(both['assigned'], both['best'])
        self.assertEqual(decoder_calls, [b'selected', b'selected'])
        with self.assertRaisesRegex(ValueError, 'accepted archived best'):
            adapter.build_pair(np, anchor, {**previous, 'raster_sha256': {}}, raw, cocos, modules)
        with self.assertRaisesRegex(ValueError, 'retained'):
            adapter.build_pair(np, {**anchor, 'assigned_native_rank': rank + 1}, None, raw, cocos, modules)


    def test_gate_counts_only_new_production_crosses_and_pairs_all_four_cells(self):
        def query(gt_min, pred_min, *, baseline=False):
            return dict(
                baseline_pass={key: baseline for key in adapter.OUTCOMES},
                support_pixels=dict(gt_band=10, predicted_band=12, predicted_band_agreement=11, intersection=7),
                capacities={support: dict(FN=1, FP=1) for support in adapter.SUPPORTS},
                regions={support: dict(minimum_edits={mode: dict.fromkeys(adapter.OUTCOMES, distance)
                                                       for mode in adapter.MODES})
                         for support, distance in zip(adapter.SUPPORTS, (gt_min, pred_min))},
            )

        rows = [dict(image_id=i // 2, area_group='small', assigned=query(None, 16), best=None)
                for i in range(20)]
        rows[0]['best'] = query(4, None)
        rows[1]['assigned'] = query(16, 16)
        rows[1]['best'] = query(None, None)
        rows.append(dict(image_id=20, area_group='medium', assigned=query(0, 0, baseline=True), best=None))
        result = adapter.summarize_production(rows)
        self.assertEqual(result['gate'], dict(budget=16, min_new_dual=20, observed_new_dual=20, passed=True))
        production = result['production_assigned']['predicted_band']['all']
        self.assertEqual(production['baseline_pass']['dual'], 1)
        self.assertEqual(production['eligible_failures']['dual'], 20)
        primary = next(row for row in production['modes']['joint']['curve'] if row['budget'] == 16)
        self.assertEqual(primary['images_with_new_dual'], 10)
        self.assertEqual(primary['total_pass']['dual'], 21)
        self.assertEqual(production['modes']['joint']['minimum_dual_edits'],
                         dict(histogram=[dict(edits=16, pairs=20)], unreachable=0))
        intersections = result['paired_224']['rescue_intersections']['curve']
        self.assertIsNone(intersections[-1]['budget'])
        selected = next(row for row in intersections if row['budget'] == 16)
        self.assertEqual(selected['patterns'], {'0110': 1, '1100': 1})
        self.assertEqual(selected['pairwise']['assigned_predicted_band__best_gt_band'],
                         dict(both=1, left_only=1, right_only=0, neither=0))
        self.assertFalse(adapter.summarize_production(rows[1:])['gate']['passed'])


if __name__ == '__main__':
    unittest.main()
