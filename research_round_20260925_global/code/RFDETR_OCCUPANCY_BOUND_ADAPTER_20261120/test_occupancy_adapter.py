"""Small synthetic fixtures through the released geometry interface only."""
import copy
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

from occupancy_adapter import evaluate_admitted


PINS = {
    'geometry': ('RFDETR_MATCHED_POINT_FORMAL_20261120/evaluate_geometry.py',
                 'fdee52489ba2df0e9db147f26d41432a3276638360f2858fb5b8c5b739953a44'),
    'oracle': ('RFDETR_FIXED_BUDGET_GEOMETRY_BOUND_20261120/oracle_geometry.py',
               '16e5929035d2e95f23b240d755bf9c9a80b3c53fd5eb6570467ad975b454edcf'),
    'score': ('RFDETR_MATCHED_POINT_FORMAL_20261120/score_dev.py',
              '271554ea01f5e5e24e48ff2c53c3229755615aa514624d7d89e5c9ec559fb79c'),
    'fit': ('RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py',
            '94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1'),
}


def released(name):
    relative, digest = PINS[name]
    path = Path(os.environ['PUBLIC_CODE_ROOT']) / relative
    assert sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location('_synthetic_oracle_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate():
    return dict(original_score_closed=True, lw_ap_blocker=True, lw_geometry_blocker=False)


def fixture(geometry):
    records = [
        dict(image_id=1, rank=7, query_id=99, group='A', role='valid', matched_gt_id=101,
             coordinates=np.array([[40, 0], [41, 1]], dtype=np.int64),
             truth=np.array([[1., 0.], [.5, .5]]),
             raw_continuous=dict(intersection=24., predicted_area=32, truth_area=32.)),
        dict(image_id=2, rank=8, query_id=98, group='A', role='valid', matched_gt_id=102,
             coordinates=np.empty((0, 2), dtype=np.int64), truth=None,
             raw_continuous=dict(intersection=24., predicted_area=32, truth_area=32.)),
        dict(image_id=3, rank=9, query_id=97, group='B', role='valid', matched_gt_id=103,
             coordinates=np.array([[50, 2], [51, 3]], dtype=np.int64),
             truth=np.zeros((2, 2)),
             raw_continuous=dict(intersection=24., predicted_area=40, truth_area=26.)),
    ]
    edits = {}
    for seed, k in zip(geometry.SEEDS, (1, 0, 1)):
        edits[seed] = {}
        for row in records:
            key = geometry.identity(row)
            coords = row['coordinates']
            policies = {}
            for arm in geometry.ARMS:
                count = (0 if arm == 'N' or len(coords) == 0 else 1 if arm in ('SF', 'SU=F') else k)
                chosen = ([] if not count else coords[:, 0].tolist() if arm == 'LW=S'
                          else coords[1].tolist())
                adds = sum(v in coords[:, 0] for v in chosen)
                drops = len(chosen) - adds
                policies[arm] = dict(flipped=chosen, edits=len(chosen), adds=adds, drops=drops,
                                     area_delta=adds-drops)
            edits[seed][key] = policies
    return records, edits


class AdapterTests(unittest.TestCase):
    def test_trigger_and_original_population_budget_guards(self):
        from occupancy_adapter import from_prepared
        class Unopened:
            def __iter__(self):
                raise AssertionError('No input access before trigger')
        for bad in ({}, dict(original_score_closed=False, lw_ap_blocker=True, lw_geometry_blocker=False),
                    dict(original_score_closed=True, lw_ap_blocker=False, lw_geometry_blocker=False),
                    dict(original_score_closed=1, lw_ap_blocker=True, lw_geometry_blocker=False)):
            with self.subTest(trigger=bad), self.assertRaises(ValueError):
                evaluate_admitted(Unopened(), {}, eligibility=bad, geometry_module=None, oracle_module=None)
            with self.subTest(wrapper_trigger=bad), self.assertRaises(ValueError):
                from_prepared(np, Unopened(), Unopened(), eligibility=bad, score_module=None,
                              fit_step=None, geometry_helpers=None, geometry_module=None, oracle_module=None)
        geometry, oracle = released('geometry'), released('oracle')
        records, edits = fixture(geometry)
        key = geometry.identity(records[0])
        mismatch = copy.deepcopy(edits)
        mismatch[5217][key]['LW=S'] = dict(flipped=[], edits=0, adds=0, drops=0, area_delta=0)
        with self.assertRaisesRegex(ValueError, 'realized edit counts'):
            evaluate_admitted(records, mismatch, eligibility=gate(), geometry_module=geometry, oracle_module=oracle)
        missing = copy.deepcopy(edits)
        del missing[5217][geometry.identity(records[1])]
        with self.assertRaisesRegex(ValueError, 'complete fixed candidate identities'):
            evaluate_admitted(records, missing, eligibility=gate(), geometry_module=geometry, oracle_module=oracle)
        empty = evaluate_admitted([], {seed:{} for seed in geometry.SEEDS}, eligibility=gate(),
                                  geometry_module=geometry, oracle_module=oracle)
        self.assertEqual(empty['private_witness'], [])
        self.assertIsNone(empty['summary']['comparisons']['O_F_minus_O_U']['delta_points'])

    def test_released_matcher_and_fractional_truth_composition(self):
        from occupancy_adapter import from_prepared
        from pycocotools import mask as codec
        geometry, oracle, score, fit = (released(name) for name in ('geometry', 'oracle', 'score', 'fit'))
        helpers = fit.load_geometry(Path(os.environ['PUBLIC_CODE_ROOT']), Path(os.environ['SHAPELY_SITE']))
        stock = np.zeros((8, 8), dtype=np.uint8)
        stock[:4] = 1
        encoded = codec.encode(np.asfortranarray(stock))
        mask = dict(size=encoded['size'], counts=encoded['counts'].decode('ascii'))
        pool = [dict(rank=7, query_id=99, score=.75, area=32, rle=mask, box_xyxy=[0., 0., 8., 4.])]
        prepared = []
        gt = dict(info={}, categories=[dict(id=1, name='ship')], images=[], annotations=[])
        for image_id in (1, 2):
            x = .5 if image_id == 1 else 0.
            gt['images'].append(dict(id=image_id, width=8, height=8))
            gt['annotations'].append(dict(id=image_id, image_id=image_id, category_id=1, iscrowd=0,
                                          area=(8-x)*4, bbox=[x, 0., 8-x, 4.],
                                          segmentation=[[x, 0., 8., 0., 8., 4., x, 4.]]))
            coords = np.array([[32, 0], [33, 1]], dtype=np.int64) if image_id == 1 else np.empty((0, 2), dtype=np.int64)
            arms = {a:dict(flipped=[33, 1], edits=2, adds=1, drops=1, area_delta=0)
                    for a in geometry.ARMS if a != 'N'} if image_id == 1 else {}
            arms.update({'LW=S':dict(flipped=[32, 33], edits=2, adds=2, drops=0, area_delta=2)}
                        if image_id == 1 else {})
            sparse = dict(original_hw=[8, 8], group='one_prefix',
                          seeds={str(seed):{'7':copy.deepcopy(arms)} for seed in geometry.SEEDS})
            prepared.append(dict(native=dict(image_id=image_id), sparse=sparse,
                                 pool=copy.deepcopy(pool), points={7:coords}))
        untouched = copy.deepcopy(gt)
        result = from_prepared(np, gt, prepared, eligibility=gate(), score_module=score,
                               fit_step=fit, geometry_helpers=helpers,
                               geometry_module=geometry, oracle_module=oracle)
        self.assertEqual(result['stock_matching_counts']['stock_matches'], 2)
        self.assertEqual(result['summary']['counts']['valid_unsupported_noop'], 1)
        row = result['private_witness'][0]
        self.assertEqual(row['matched_gt_id'], 1)
        self.assertEqual(row['oracle']['original']['I'], 30.)
        self.assertEqual(row['oracle']['oracles']['O_U']['removed_intersection'], .5)
        self.assertEqual(row['oracle']['oracles']['O_U']['drop_indices'], [0])
        self.assertEqual(row['observed']['SU']['I'], 29.)
        self.assertEqual(gt, untouched)
        self.assertFalse(fit.torch.cuda.is_initialized())

    def test_actual_seed_budgets_noops_and_source_equal_paired_bootstrap(self):
        geometry, oracle = released('geometry'), released('oracle')
        records, edits = fixture(geometry)
        result = evaluate_admitted(records, edits, eligibility=gate(),
                                   geometry_module=geometry, oracle_module=oracle)
        summary = result['summary']
        self.assertEqual(summary['counts']['valid'], 3)
        self.assertEqual(summary['counts']['valid_unsupported_noop'], 1)
        self.assertEqual(summary['source_groups'], 2)
        self.assertEqual(len(result['private_witness']), 9)
        for witness in result['private_witness']:
            expected_k = 0 if witness['seed'] == 5218 or witness['image_id'] == 2 else 1
            self.assertEqual(witness['oracle']['budget']['k'], expected_k)
            self.assertEqual(witness['oracle']['budget']['edits'], witness['observed']['SU']['edits'])
        # A has two candidates including a no-op. B has one. Seeds average first.
        grouped = np.array([((25/39-24/40) * (2/3)) / 2, 0.]) * 100
        draws = np.random.Generator(np.random.PCG64(5217)).integers(2, size=(4096, 2))
        expected = np.quantile(grouped[draws].mean(axis=1), [.025, .975]).tolist()
        contrast = summary['comparisons']['O_U_minus_SU']
        self.assertAlmostEqual(contrast['delta_points'], float(grouped.mean()))
        np.testing.assert_allclose(contrast['interval95_points'], expected, rtol=0, atol=1e-12)
        self.assertEqual(summary['bootstrap_draws'], 4096)
        self.assertEqual(summary['bootstrap_seed'], 5217)
        self.assertFalse(result['AP_computed'])
        self.assertFalse(result['input_admission_verified_here'])
        self.assertEqual(summary, evaluate_admitted(records, edits, eligibility=gate(),
                         geometry_module=geometry, oracle_module=oracle)['summary'])


if __name__ == '__main__':
    unittest.main()
