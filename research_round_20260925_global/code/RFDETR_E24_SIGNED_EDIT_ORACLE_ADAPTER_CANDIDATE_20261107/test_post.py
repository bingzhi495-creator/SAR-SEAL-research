"""Synthetic contract only. Uses no SAR inputs or production implementation."""
from copy import deepcopy
from unittest import TestCase, main
from unittest.mock import patch

import post


BUDGETS = (0, 1, 2, 4, 8, 16, 32)
REGIONS = ('band', 'outside')
MODES = ('fn_only', 'fp_only', 'joint')
I, F = 'INTEGER_PRIMARY', 'LOCAL_CANDIDATE_FLOAT'
SCOPE = dict(images=1, GT_instances=3, fixed_pairs=2, cohort_pairs=1,
             cohort_images=1, other_fixed_pairs_not_evaluated=1, original_unpaired_GT=1)
CAPACITIES = dict(band=dict(FN=1, FP=1), outside=dict(FN=0, FP=0))


def fixture():
    def counts(a, b, c):
        return dict(integer=a, float=b, dual=c)

    row = dict(image_id=987, GT_annotation_id=654, selected_raw_query_id=17, assigned_native_rank=8,
               assigned_query_id=21, original_hw=[4, 4], integer_annotation_area=10, area_group='small',
               overlap_counts={I: [8, 10], F: [8, 11]}, baseline_pass=counts(False, False, False),
               capacities=deepcopy(CAPACITIES), raster_sha256={'prediction': 'a'*64, I: 'b'*64, F: 'c'*64},
               regions={})
    witness = {name: deepcopy(row[name]) for name in
               ('image_id', 'GT_annotation_id', 'selected_raw_query_id', 'raster_sha256')}
    groups = {}
    for group, measured, fixed in (('all', 1, 2), ('small', 1, 1), ('medium', 0, 1), ('large', 0, 0)):
        regions = {}
        for region in REGIONS:
            modes = {}
            for mode in MODES:
                allow = measured and region == 'band'
                at = (1, 2, 2) if allow and mode == 'joint' else (
                    (1, None, None) if allow and mode == 'fn_only' else (None, None, None))
                modes[mode] = dict(curve=[dict(budget=k, **counts(*(int(v is not None and v <= k) for v in at)))
                                          for k in BUDGETS],
                                   unlimited=counts(*(int(v is not None) for v in at)))
            regions[region] = modes
        groups[group] = dict(cohort_pairs=measured, fixed_pairs_denominator=fixed,
                             baseline_pass=counts(0, 0, 0), regions=regions)
    for region in REGIONS:
        curve = []
        for k in BUDGETS:
            point = {'budget': k}
            for mode in MODES:
                a = int(region == 'band' and k >= 1 and mode != 'fp_only')
                b = int(region == 'band' and ((k >= 1 and mode == 'fp_only') or (k >= 2 and mode == 'joint')))
                point[mode] = dict(additions=a, removals=b, integer_iou=[8+a, 10-b], float_iou=[8+a, 11-b])
            curve.append(point)
        minima = dict(fn_only=counts(1, None, None), fp_only=counts(None, None, None), joint=counts(1, 2, 2))
        if region == 'outside':
            minima = {mode: counts(None, None, None) for mode in MODES}
        row['regions'][region] = dict(threshold=[9, 10], minimum_edits=minima, curve=curve)
    summary = dict(schema='Root-RFDETR-E24-dual-GT-signed-edit-oracle-v1',
                   status='SIGNED_EDIT_ORACLE_COMPLETE_PENDING_ROOT_ACCEPTANCE', qualified=False,
                   TEST=False, AP_recomputed=False, COCO_matching_recomputed=False, new_training=False,
                   scope=deepcopy(SCOPE), threshold=[9, 10], budgets=list(BUDGETS), radius_pixels=2,
                   area_partition='Original integer COCO annotation area: [0,1024), [1024,9216), [9216,1e10]. '
                                  'Exact shared COCO boundaries are assigned to the larger group for disjoint totals.',
                   counts='Newly reaches threshold within the measured cohort. The fixed denominator includes '
                          'unevaluated pairs, so these counts are not a whole-population edited result.',
                   input_IoU_and_localization_exact=True, no_new_query_rank_search=True,
                   capacities=deepcopy(CAPACITIES), groups=groups)
    return summary, [row], [witness]


class PostTests(TestCase):
    def test_exact_two_direction_crossings_recomputed_without_exposing_identity(self):
        with patch.multiple(post, SCOPE=SCOPE, CAPACITIES=CAPACITIES):
            result = post.verify(*fixture())
        self.assertEqual(result['verified_pairs'], 1)
        self.assertEqual(result['verified_raster_fingerprints'], 3)
        self.assertNotIn('987', str(result))
        self.assertNotIn('654', str(result))

    def test_detects_corrupted_witness_and_disagrees_with_falsified_counts(self):
        with patch.multiple(post, SCOPE=SCOPE, CAPACITIES=CAPACITIES):
            summary, rows, witnesses = fixture()
            witnesses[0]['raster_sha256']['prediction'] = 'd' * 64
            with self.assertRaisesRegex(ValueError, 'raster witness'):
                post.verify(summary, rows, witnesses)

            summary, rows, witnesses = fixture()
            summary['groups']['small']['regions']['band']['joint']['unlimited']['dual'] = 0
            with self.assertRaisesRegex(ValueError, 'summary differs'):
                post.verify(summary, rows, witnesses)

            summary, rows, witnesses = fixture()
            rows[0]['regions']['band']['minimum_edits']['joint']['dual'] = 1
            with self.assertRaisesRegex(ValueError, 'minimum edit distances'):
                post.verify(summary, rows, witnesses)


if __name__ == '__main__':
    main()
