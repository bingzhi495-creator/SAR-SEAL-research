"""Closure and dispatch fixtures only, without actual score or label files."""
import copy
import json
from types import SimpleNamespace
import unittest

from run_diagnostic import (ORIGINAL_SCORE_SHA, ORIGINAL_RECEIPT_SHA,
                            validate_score_closure)


def closure():
    gate = dict(schema='RFDETR-fixed-point-refinement-gate-v1', primary='SU',
                seeds=[5217, 5218, 5219], advance=False,
                gates={'AP_vs_LW=S':False, 'geometry_vs_LW=S':False})
    score = dict(status='RFDETR_CENTER5202_DEV705_MATCHED_SCORE_ACTUAL', gate=gate,
                 DEV_GT_opened=True, prelabel_sha256='1'*64,
                 private_shape_witness_sha256='2'*64)
    receipt = dict(scope='exact sole original score natural exit',
                   launch=dict(session_id=74246), terminal=dict(exit_code=0),
                   observed_score_footer=dict(status=score['status'], result_sha256=ORIGINAL_SCORE_SHA,
                                              gate=copy.deepcopy(gate)),
                   verified_report_sha256=ORIGINAL_SCORE_SHA)
    return score, receipt


class ClosureTests(unittest.TestCase):
    def test_original_prelabel_and_sparse_metadata_join_without_reconstruction(self):
        import numpy as np
        from run_diagnostic import validate_score_prelabel, join_prepared
        from test_occupancy_adapter import released
        score = released('score')
        args = SimpleNamespace(readout_accepted_sha='3'*64, capture_accepted_sha='4'*64)
        approved = dict(edits_sha256='5'*64)
        original = SimpleNamespace(STOCK='6'*64, OLD_SCORE='7'*64)
        pre = dict(status='RFDETR_CENTER5202_DEV705_SCORE_PRELABEL',
                   verified_readout_sha256=args.readout_accepted_sha, sparse_sha256=approved['edits_sha256'],
                   target_native_sha256=original.STOCK, previous_native_score_sha256=original.OLD_SCORE,
                   images=705, DEV=True, FIT=False, TEST=False, GT_opened=False, AP=False)
        validate_score_prelabel(pre, args, approved, original)
        bad = dict(pre, TEST=True)
        with self.assertRaises(ValueError): validate_score_prelabel(bad, args, approved, original)
        pool = [dict(rank=7, query_id=99, score=.7, area=32)]
        row = dict(native_block=dict(image_id=1), pool=pool, source_prefix='A')
        sparse = dict(ordinal=1, image_id=1, group='A', original_hw=[8,8],
                      mapped_identity_sha256=score.mapped_identity_sha256(pool),
                      points=[dict(rank=7, query_id=99, coordinates=[])],
                      seeds={str(s):{} for s in score.SEEDS})
        result = join_prepared(np, [row], [sparse], score)
        self.assertIs(result[0]['pool'], pool)
        self.assertEqual(result[0]['points'][7].shape, (0,2))
        for replacement in (dict(sparse, group='B'), dict(sparse, points=[])):
            with self.assertRaises(ValueError): join_prepared(np, [row], [replacement], score)

    def test_geometry_only_dispatch_and_only_original_dev_label_names(self):
        import numpy as np
        from run_diagnostic import compute_geometry_only, load_dev_geometry_inputs
        import occupancy_adapter as adapter
        from test_occupancy_adapter import released, fixture, gate
        geometry, oracle, score = (released(name) for name in ('geometry','oracle','score'))
        records, all_edits = fixture(geometry)
        records = records[:1]
        key = geometry.identity(records[0])
        edits = {s:{key:all_edits[s][key]} for s in geometry.SEEDS}
        def forbidden(*args, **kwargs):
            raise AssertionError('AP/reconstruction must never be invoked')
        for name in ('score_dev','score_matched_population','score_official','prepare_reconstruction',
                     'reconstruct_predictions','_reconstruct'):
            setattr(score, name, forbidden)
        audit = dict(stock_matches=1, valid=1, supervised_pairs=2)
        fit = SimpleNamespace(attach_truth=lambda image, annotations, pool, points, helpers:(records,audit))
        gt = dict(images=[dict(id=1,height=8,width=8)], annotations=[])
        prepared = [dict(native=dict(image_id=1), pool=[dict(rank=7,query_id=99)],
                         points={7:records[0]['coordinates']}, sparse=dict(original_hw=[8,8],group='A',
                         seeds={str(s):{'7':edits[s][key]} for s in geometry.SEEDS}))]
        observed = geometry.evaluate_geometry(records, edits)
        original_score = dict(geometry=dict(summary=json.loads(json.dumps(observed['summary'])),
                                           stock_matching_counts=audit))
        result = compute_geometry_only(np, gt, prepared, original_score, eligibility=gate(), scorer=score,
                                       fit_step=fit, helpers=None, geometry=geometry, adapter=adapter, oracle=oracle)
        self.assertFalse(result['AP_computed'])
        self.assertEqual(result['stock_matching_counts'], audit)
        bad = copy.deepcopy(original_score)
        bad['geometry']['summary']['means']['SU']['group_equal_iou_points'] += .01
        with self.assertRaisesRegex(ValueError, 'original observed geometry'):
            compute_geometry_only(np, gt, prepared, bad, eligibility=gate(), scorer=score,
                                  fit_step=fit, helpers=None, geometry=geometry, adapter=adapter, oracle=oracle)
        names = []
        def dev_file(name):
            names.append(name)
            if name == 'DEV_ORIGINAL_GT_PRIVATE.json': return gt
            if name == 'DEV_EMPTY_RASTERS_PRIVATE.json':
                return [dict(kind='EMPTY_COCO_RASTER',annotation_id=i) for i in range(4)]
            raise AssertionError('No TEST or alternative label access')
        self.assertIs(load_dev_geometry_inputs(SimpleNamespace(dev_file=dev_file))[0], gt)
        self.assertEqual(names, ['DEV_ORIGINAL_GT_PRIVATE.json','DEV_EMPTY_RASTERS_PRIVATE.json'])

    def test_exact_closed_observed_lw_blockers_only(self):
        score, receipt = closure()
        self.assertEqual(validate_score_closure(score, receipt, ORIGINAL_SCORE_SHA, ORIGINAL_RECEIPT_SHA),
                         dict(original_score_closed=True, lw_ap_blocker=True, lw_geometry_blocker=True))
        for change in ('wrong_score_sha','wrong_receipt_sha','exit','session','footer','AP_pass','geometry_pass'):
            s, r = copy.deepcopy(score), copy.deepcopy(receipt)
            ss, rs = ORIGINAL_SCORE_SHA, ORIGINAL_RECEIPT_SHA
            if change == 'wrong_score_sha': ss = '0'*64
            if change == 'wrong_receipt_sha': rs = '0'*64
            if change == 'exit': r['terminal']['exit_code'] = 1
            if change == 'session': r['launch']['session_id'] = 74247
            if change == 'footer': r['observed_score_footer']['gate']['advance'] = True
            if change == 'AP_pass':
                s['gate']['gates']['AP_vs_LW=S'] = True
                r['observed_score_footer']['gate'] = copy.deepcopy(s['gate'])
            if change == 'geometry_pass':
                s['gate']['gates']['geometry_vs_LW=S'] = True
                r['observed_score_footer']['gate'] = copy.deepcopy(s['gate'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_score_closure(s, r, ss, rs)


if __name__ == '__main__':
    unittest.main()
