"""UNIT_FIXTURE: NumPy synthetic arrays, fake torch loader, no real packets."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnitTensor:
    def __init__(self, value, dtype):
        self.array = np.asarray(value, dtype=dtype)
        self.dtype = str(self.array.dtype)
        self.device = SimpleNamespace(type='cpu')
        self.requires_grad = False

    def detach(self):
        return self

    def numpy(self):
        return self.array


class UnitTorch:
    float32 = 'float32'
    int64 = 'int64'
    bool = 'bool'
    packets = {}

    @staticmethod
    def is_tensor(value):
        return isinstance(value, UnitTensor)

    @classmethod
    def load(cls, path, **kwargs):
        assert kwargs == dict(map_location='cpu', weights_only=True)
        return copy.deepcopy(cls.packets[str(path)])


def tensor(value, dtype='float32'):
    return UnitTensor(value, dtype)


def packet(dataset_index=0):
    native = np.full((100, 2, 3), -3, dtype=np.float32)
    native[0] = [[-1, 3, -3], [-3, -1, 3]]
    native[1] = [[-3, 3, -3], [-3, -3, -3]]
    gt = np.zeros((2, 2, 3), dtype=bool)
    gt[0, 0, 0] = True
    gt[1, 0, 1] = True
    raw = np.full(100, -1, dtype=np.int64)
    raw[0], raw[1], raw[99] = 0, 1, -2
    ann_ids = [10000+2*dataset_index, 10001+2*dataset_index]
    kinds = ['matched' if x >= 0 else 'background' if x == -1 else 'ignored' for x in raw]
    queries = []
    for q in range(100):
        flat = native[q].reshape(-1)
        omega = np.flatnonzero(np.abs(flat) <= np.float32(2)).astype(np.int64)
        # Tiny synthetic image: dilation of any positive pixel covers the image.
        c = np.arange(6, dtype=np.int64) if np.any(flat > 0) else omega
        u = np.zeros(len(c), dtype=np.float32)
        if q == 0:
            u[c == 0] = np.float32(2)
            u[c == 1] = np.float32(-4)
        combined = np.add(flat[c], u, dtype=np.float32)
        before, after = flat[c] > 0, combined > 0
        queries.append(dict(query_id=q, C_count=len(c), K=len(omega),
            C_indices=tensor(c, 'int64'), native_L_on_C=tensor(flat[c]), PR_u=tensor(u),
            PR_combined=tensor(combined), native_positive=tensor(before, 'bool'),
            actual_after_positive=tensor(after, 'bool'), actual_flip=tensor(before ^ after, 'bool'),
            StrongR_S_indices=tensor(omega, 'int64')))
    assignment = dict(natural_query_ids=list(range(100)), raw_GT_index_by_query=tensor(raw, 'int64'),
        raw_GT_annotation_id_by_query=[ann_ids[x] if x >= 0 else None for x in raw],
        supervision_kind_by_query=kinds, valid_targets=tensor(raw != -2, 'bool'),
        native_positive_query_indices=[0, 1], native_negative_query_indices=list(range(2, 99)),
        native_ignored_query_indices=[99], final_native_labels=tensor(np.zeros(100), 'int64'),
        final_native_label_weights=tensor(np.ones(100), 'int64'),
        final_native_mask_weights=tensor(np.r_[1, 1, np.zeros(98)]),
        background_empty_set_is_new_supervision_extension=True,
        original_native_negative_mask_supervised=False, ignore_never_zero_filled=True)
    return dict(schema='N22-FIT32-natural100-fullC-actual-GPU-action-packet-v1',
        dataset_index=dataset_index, identity=dict(image_id='UNIT_FIXTURE_'+str(dataset_index), ann_ids=ann_ids),
        natural_query_ids=list(range(100)), native_L=tensor(native),
        native_e0=tensor(np.zeros((1, 100, 256))), native_class_logits=tensor(np.zeros((1, 100, 2))),
        raw_GT=tensor(gt, 'bool'), raw_GT_ids=ann_ids, raw_target_mapping=(1, 0),
        assignment=assignment, native_stage_matches=[{} for _ in range(10)],
        all_ten_target_observations=[{} for _ in range(10)],
        processed_target_proof={'UNIT_FIXTURE': True},
        actual_raw_processed_GT_correspondence={'UNIT_FIXTURE': True},
        native_export_query_ids=list(reversed(range(100))), native_scores_by_query=tensor(np.zeros(100)),
        native_top_scores_by_export_order=tensor(np.zeros(100)),
        native_quality_by_export_order=tensor(np.zeros(100)), category_id=1,
        input_refs={'UNIT_FIXTURE': True}, queries=queries)


class VerifyBodyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = load(Path(__file__).with_name('verify_body.py'), 'N22_UNIT_FIXTURE_verify')

    def verify(self, value):
        return self.body.verify_packet(UnitTorch, np, value, check=lambda: None)

    def test_independent_real_pair_counts_include_native_zero_and_audit_others(self):
        result = self.verify(packet())
        self.assertEqual(len(result['records']), 2)
        row = result['records'][0]
        self.assertEqual((row['I'], row['U'], row['native_fn'], row['native_fp']), (0, 3, 1, 2))
        self.assertEqual((row['candidate_fn'], row['candidate_fp'], row['candidate_tp'], row['candidate_tn']), (1, 2, 0, 3))
        self.assertEqual((row['pr_beneficial_fn'], row['pr_beneficial_fp']), (1, 1))
        self.assertEqual((row['b_FN'], row['b_FP'], row['h_TP'], row['h_TN']), (1, 0, 0, 0))
        self.assertEqual((row['K'], row['selected_count']), (2, 2))
        self.assertEqual(result['audit_counts'], dict(image_count=1, collected_queries=100,
            unassigned_queries=97, ignored_queries=1, unmatched_gt=0, total_real_gt=2))
        self.assertEqual(result['checks']['all_query_count'], 100)
        self.assertEqual(result['checks']['cpu_fp32_combined_sign_mismatches'], 0)
        strata = result['regional_counts'][0]
        self.assertEqual(strata['Omega_inside']['C_count'], 2)
        self.assertEqual(strata['Omega_outside']['C_count'], 4)
        self.assertEqual(strata['Omega_inside']['pr_beneficial_fn'], 1)
        self.assertEqual(strata['Omega_outside']['pr_beneficial_fp'], 1)
        self.assertEqual(strata['Omega_outside']['native_fp_other_real_GT'], 1)
        self.assertEqual(strata['Omega_outside']['native_fp_pure_background'], 1)
        self.assertEqual(strata['Omega_outside']['pr_beneficial_fp_other_real_GT'], 1)
        for key in ('native_fn', 'native_fp', 'pr_beneficial_fn', 'pr_beneficial_fp'):
            self.assertEqual(strata['Omega_inside'][key]+strata['Omega_outside'][key], row[key])

    def test_C_outside_native_FN_is_kept_and_regions_never_get_separate_budgets(self):
        value = packet()
        value['native_L'].array[1].fill(-3)
        value['queries'][1] = copy.deepcopy(value['queries'][2])
        value['queries'][1]['query_id'] = 1
        result = self.verify(value)
        self.assertEqual(result['records'][1]['native_fn'], 1)
        region = result['regional_counts'][1]
        self.assertEqual(region['Omega_outside']['native_fn'], 1)
        self.assertEqual(region['Omega_outside']['C_count'], 0)
        self.assertEqual(region['Omega_outside']['pr_beneficial_fn'], 0)
        self.assertNotIn('K', region['Omega_inside'])
        self.assertNotIn('K', region['Omega_outside'])

    def test_separable_dilation_matches_square_zero_border_including_small_and_empty(self):
        for h, w in [(1, 1), (2, 3), (18, 19), (21, 22)]:
            for positions in [[], [(0, 0)], [(h-1, w-1)], [(h//2, w//2)]]:
                x = np.zeros((h, w), dtype=bool)
                for y, z in positions:
                    x[y, z] = True
                expected = np.zeros_like(x)
                for y, z in positions:
                    expected[max(0, y-8):min(h, y+9), max(0, z-8):min(w, z+9)] = True
                np.testing.assert_array_equal(self.body.dilate17(np, x), expected)

    def test_wrong_C_order_duplicates_membership_and_Omega_are_rejected(self):
        for mutate in [lambda p: p['queries'][0]['C_indices'].array.__setitem__(slice(0, 2), [1, 0]),
                       lambda p: p['queries'][0]['C_indices'].array.__setitem__(1, 0),
                       lambda p: p['queries'][0]['StrongR_S_indices'].array.__setitem__(0, 1),
                       lambda p: p['queries'][2].__setitem__('K', 1),
                       lambda p: p['queries'].__setitem__(slice(0, 2), [p['queries'][1], p['queries'][0]])]:
            value = packet()
            mutate(value)
            with self.assertRaises(ValueError):
                self.verify(value)

    def test_actual_GPU_sign_mismatch_refused_but_same_sign_numeric_difference_reported(self):
        value = packet()
        value['queries'][0]['PR_combined'].array[0] = np.float32(-1)
        with self.assertRaises(ValueError):
            self.verify(value)
        value = packet()
        # Actual GPU value differs numerically while its sign remains identical.
        value['queries'][0]['PR_combined'].array[0] = np.float32(1.25)
        self.assertEqual(self.verify(value)['checks']['cpu_gpu_combined_max_abs_difference'], 0.25)
        value = packet()
        value['queries'][0]['actual_after_positive'].array[0] = False
        with self.assertRaises(ValueError):
            self.verify(value)

    def test_true_GT_assignment_identity_dtype_and_full100_are_checked(self):
        changes = [lambda p: p['assignment']['raw_GT_index_by_query'].array.__setitem__(1, 0),
                   lambda p: p['assignment']['raw_GT_annotation_id_by_query'].__setitem__(0, -999),
                   lambda p: p['assignment']['supervision_kind_by_query'].__setitem__(2, 'matched'),
                   lambda p: p.__setitem__('raw_target_mapping', (0, 0)),
                   lambda p: p.__setitem__('natural_query_ids', list(range(99))),
                   lambda p: p['assignment'].__setitem__('final_native_label_weights', tensor(np.ones(100))),
                   lambda p: p['queries'][0].__setitem__('PR_u', tensor(np.zeros(6), 'float64'))]
        for change in changes:
            value = packet()
            change(value)
            with self.assertRaises(ValueError):
                self.verify(value)

    def test_owned_run_32_synthetic_packets_and_fresh_output_no_teacher_PR_or_F(self):
        base = Path(__file__).parent
        analysis = base/'analysis.py'
        oracle = base/'oracle.py'
        with tempfile.TemporaryDirectory(prefix='N22_UNIT_FIXTURE_') as temp:
            temp = Path(temp)
            values = [packet(i) for i in range(32)]
            refs = [{'path': str(temp/('UNIT_FIXTURE_PACKET_'+str(i)+'.pth'))} for i in range(32)]
            UnitTorch.packets = {ref['path']: value for ref, value in zip(refs, values)}
            rows = [dict(dataset_index=i, image_id=value['identity']['image_id'],
                         identity=value['identity'], packet_ref=ref)
                    for i, value, ref in zip(range(32), values, refs)]
            manifest_ref = dict(path=str(temp/'UNIT_FIXTURE_CAPTURE_MANIFEST.json'))
            q_ref = dict(path=str(temp/'UNIT_FIXTURE_GPU_Q.json'))
            Path(manifest_ref['path']).write_text(json.dumps(dict(
                schema='N22-FIT32-natural100-fullC-capture-report-v1', rows=rows, packet_refs=refs)))
            Path(q_ref['path']).write_text(json.dumps(dict(schema='N22-owned-pending-terminal-result-v1',
                operation='GPU_CAPTURE', status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
                result=dict(manifest_ref=manifest_ref, packet_refs=refs))))
            allowed = [q_ref['path'], manifest_ref['path'], str(analysis), str(oracle)] + [r['path'] for r in refs]
            calls = []
            class Control:
                def check(self):
                    pass
                def closed(self, ref):
                    self.check()
                    if ref['path'] not in allowed:
                        raise AssertionError('Unexpected external input')
                    calls.append(ref['path'])
                    return Path(ref['path'])
                def write_json(self, path, value):
                    path = Path(path)
                    self.assert_output(path)
                    with path.open('x', encoding='utf-8') as handle:
                        json.dump(value, handle)
                    assert json.loads(path.read_text()) == value
                    return dict(path=str(path))
                def assert_output(self, path):
                    assert path.parent == temp/'UNIT_FIXTURE_CPU_OUTPUT'
            spec = dict(output_directory=str(temp/'UNIT_FIXTURE_CPU_OUTPUT'),
                cuda_visible_devices='', GPU_capture_Q=q_ref, GPU_capture_REPORT=manifest_ref,
                captured_packet_refs=refs, analysis_source=dict(path=str(analysis)),
                oracle_source=dict(path=str(oracle)), selected_dataset_indices=list(range(32)),
                selected_identities=[p['identity'] for p in values])
            result = self.body.run(UnitTorch, np, spec, Control())
            self.assertEqual(result['summary']['paired_instances'], 64)
            self.assertEqual(result['summary']['native_iou_zero_pairs'], 32)
            self.assertEqual(result['summary']['images'], 32)
            self.assertEqual(result['summary']['natural_queries'], 3200)
            saved = json.loads(Path(result['analysis_ref']['path']).read_text())
            self.assertEqual(saved['population']['paired_instances'], 64)
            self.assertFalse(result['qualified'])
            self.assertFalse(result['is_AP_bound'])
            self.assertEqual(set(calls), set(allowed))
            with self.assertRaises(FileExistsError):
                self.body.run(UnitTorch, np, spec, Control())


if __name__ == '__main__':
    print('UNIT_FIXTURE_ONLY: synthetic NumPy/fake torch; no actual GPU outputs or scientific leaf.')
    unittest.main(verbosity=2)
