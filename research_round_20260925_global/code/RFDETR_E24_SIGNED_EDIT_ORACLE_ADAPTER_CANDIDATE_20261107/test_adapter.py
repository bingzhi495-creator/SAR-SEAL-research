"""Synthetic adapter boundary checks. No actual SAR inputs or codec."""
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import adapter
from adapter import area_group, summarize


class OracleAdapterTests(unittest.TestCase):
    def test_exact_coco_boundaries_go_to_the_larger_disjoint_group(self):
        self.assertEqual([area_group(n) for n in (0, 1023.5, 1024, 9215.5, 9216)],
                         ['small', 'small', 'medium', 'medium', 'large'])

    def test_224_cohort_crossings_keep_fixed_denominator_and_no_identities(self):
        budgets = (0, 1, 2, 4, 8, 16, 32)
        modes = ('fn_only', 'fp_only', 'joint')

        def fixture(minimum):
            return {'minimum_edits': {mode: dict(integer=minimum, float=minimum, dual=minimum)
                                     for mode in modes}, 'threshold': (9, 10)}

        row = dict(image_id=999, GT_annotation_id=888, area_group='small',
                   baseline_pass=dict(integer=False, float=False, dual=False),
                   regions=dict(band=fixture(2), outside=fixture(None)))
        result = summarize([row], {'small': 3, 'medium': 0, 'large': 0}, budgets)
        total = result['all']
        self.assertEqual((total['cohort_pairs'], total['fixed_pairs_denominator']), (1, 3))
        self.assertEqual(total['regions']['band']['joint']['curve'][2]['dual'], 1)
        self.assertEqual(total['regions']['outside']['joint']['unlimited']['dual'], 0)
        self.assertNotIn('image_id', str(result))
        self.assertNotIn('GT_annotation_id', str(result))


    def test_exact_archived_pair_rebuild_keeps_one_mask_and_separate_capacities(self):
        integer, floating = adapter.VARIANTS
        sizes = [2, 3]
        prediction = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)
        int_mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        float_mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        areas = {'GT_union': 3, 'GT_disagreement': 1}
        errors = dict(FN=dict(band=1, outside=0), FP=dict(band=0, outside=1))
        raw = dict(image_id=9, original_hw=sizes,
                   raw_masks=[dict(size=sizes, counts='chosen') for _ in range(100)])
        anchor = dict(image_id=9, GT_annotation_id=17, assigned_native_rank=1, assigned_query_id=3,
                      by_GT={integer: dict(raw=1/3, retained=1/3, assigned=1/3),
                             floating: dict(raw=1/4, retained=1/4, assigned=1/4)})
        row = dict(image_id=9, GT_annotation_id=17, selected_raw_query_id=42,
                   prior_overlaps=anchor['by_GT'], selected_IoU_by_GT={integer: 1/3, floating: 1/4},
                   areas=areas, errors=errors)

        def decode(rle):
            return prediction if isinstance(rle, dict) and rle['counts'] == b'chosen' else rle

        def localize(np_api, gi, gf, mask):
            self.assertIs(mask, prediction)
            self.assertIs(gi, int_mask)
            self.assertIs(gf, float_mask)
            return dict(areas=areas, errors=errors)

        def reader(mask):
            ann = dict(image_id=9, category_id=1, area=1024, iscrowd=0)
            return SimpleNamespace(anns={17: ann}, annToRLE=lambda _: mask)

        def oracle(integer_pair, float_pair, fn, fp):
            self.assertEqual((integer_pair, float_pair), ((1, 3), (1, 4)))
            return dict(capacities=(fn, fp))

        modules = dict(localization=SimpleNamespace(binary_mask=lambda _, value: value, localize_pair=localize,
                                                               signature=lambda value: json.dumps(value, sort_keys=True)),
                       mask_api=SimpleNamespace(decode=decode),
                       oracle_math=SimpleNamespace(signed_edit_oracle=oracle, THRESHOLD=(9, 10)))
        cocos = {integer: reader(int_mask), floating: reader(float_mask)}
        result = adapter.rebuild_pair(np, anchor, row, raw, cocos, modules)
        self.assertEqual(result['regions'], dict(band=dict(capacities=(1, 0)), outside=dict(capacities=(0, 1))))
        self.assertEqual(result['overlap_counts'], {integer: (1, 3), floating: (1, 4)})
        self.assertEqual(result['selected_raw_query_id'], 42)
        self.assertEqual(result['area_group'], 'medium')
        self.assertEqual(set(result['raster_sha256']), {'prediction', integer, floating})
        self.assertEqual(len(set(result['raster_sha256'].values())), 3)
        self.assertEqual(result['raster_sha256']['prediction'],
                         hashlib.sha256(bytes((1, 0, 1, 0, 0, 0))).hexdigest())

        changed = {**row, 'selected_IoU_by_GT': {integer: 1/3, floating: 1/5}}
        with self.assertRaisesRegex(ValueError, 'exact archived'):
            adapter.rebuild_pair(np, anchor, changed, raw, cocos, modules)
        changed = {**row, 'errors': dict(FN=dict(band=0, outside=1), FP=errors['FP'])}
        with self.assertRaisesRegex(ValueError, 'localization counts'):
            adapter.rebuild_pair(np, anchor, changed, raw, cocos, modules)


    def test_original_pair_subsequence_and_fixed_group_denominators(self):
        class FixtureCOCO:
            def createIndex(self):
                self.imgs = {row['id']: row for row in self.dataset['images']}
                self.anns = {row['id']: row for row in self.dataset['annotations']}
                self.cats = {1: {}}

        document = dict(images=[dict(id=9, height=2, width=3)],
                        annotations=[dict(id=17, image_id=9, area=20), dict(id=18, image_id=9, area=1024)])
        raw = dict(image_id=9, original_hw=[2, 3], raw_masks=[None] * 100)
        anchors = [
            dict(image_id=9, GT_annotation_id=aid, assigned_native_rank=1, assigned_query_id=3,
                 by_GT={variant: dict(raw=maximum, retained=.75, assigned=.5) for variant in adapter.VARIANTS})
            for aid, maximum in ((17, .85), (18, .90))
        ]
        archived = [dict(GT_annotation_id=17)]
        budgets = (0, 1, 2, 4, 8, 16, 32)
        minima = {mode: dict(integer=1, float=1, dual=1) for mode in adapter.MODES}
        private = dict(area_group='small', baseline_pass=dict(integer=False, float=False, dual=False),
                       capacities={region: dict(FN=1, FP=0) for region in adapter.REGIONS},
                       regions={region: dict(minimum_edits=minima) for region in adapter.REGIONS})
        modules = dict(localization=SimpleNamespace(THRESHOLD=.9, RADIUS=2, ACCEPTED_COUNTS={}),
                       raw_geometry=SimpleNamespace(RAW_KEYS=set(raw), decompose=lambda _: {}),
                       gt_sensitivity=SimpleNamespace(validate_GT_pair=lambda *args: None),
                       coco_loader=SimpleNamespace(COCO=FixtureCOCO), mask_api=SimpleNamespace(),
                       oracle_math=SimpleNamespace(THRESHOLD=(9, 10), BUDGETS=budgets))
        with patch.multiple(adapter, DEV_IMAGES=1, GT_INSTANCES=2, FIXED_PAIRS=2, COHORT_PAIRS=1, COHORT_IMAGES=1,
                            ACCEPTED_CAPACITIES={'band': (1, 0), 'outside': (1, 0)}), \
                patch.object(adapter, 'rebuild_pair', return_value=private) as rebuild:
            report = adapter.evaluate_signed_edit_oracle(
                np, SimpleNamespace(check=lambda: None), integer_gt=document, floating_gt=document,
                accepted_raw_private_pairs=anchors, accepted_localization_rows=archived,
                raw_records=[raw], modules=modules)
        self.assertEqual(rebuild.call_count, 1)
        self.assertIs(rebuild.call_args.args[1], anchors[0])
        self.assertIs(rebuild.call_args.args[2], archived[0])
        self.assertEqual(report['summary']['groups']['all']['fixed_pairs_denominator'], 2)
        self.assertEqual(report['summary']['groups']['medium']['fixed_pairs_denominator'], 1)
        self.assertEqual(report['summary']['groups']['medium']['cohort_pairs'], 0)
        self.assertEqual(report['private_pair_records'], [private])


if __name__ == '__main__':
    unittest.main()
