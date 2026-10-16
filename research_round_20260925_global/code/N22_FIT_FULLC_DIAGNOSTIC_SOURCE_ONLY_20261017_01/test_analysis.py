"""UNIT_FIXTURE only: pure fixed-pair counts; no real inputs, models or AP."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    before_imports = set(sys.modules)
    spec.loader.exec_module(module)
    assert not ({'torch', 'numpy'} & (set(sys.modules)-before_imports))
    return module


ORACLE_PATH = Path(__file__).with_name('oracle.py')


def paired(**changes):
    row = dict(pair_id='UNIT_FIXTURE_PAIR_0', image_id='UNIT_FIXTURE_IMAGE_0',
               query_index=0, gt_index=0, I=4, U=15, native_fn=6, native_fp=5,
               candidate_fn=4, candidate_fp=3, candidate_tp=2, candidate_tn=7,
               pr_beneficial_fn=3, pr_beneficial_fp=2,
               b_FN=1, b_FP=1, h_TP=1, h_TN=1, K=4, selected_count=4)
    row.update(changes)
    return row


def audit(pair_count=1, **changes):
    row = dict(image_count=1, collected_queries=100, unassigned_queries=100-pair_count,
               ignored_queries=0, unmatched_gt=1, total_real_gt=pair_count+1)
    row.update(changes)
    return row


def rational(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator)


class CountAnalysisContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = load(ORACLE_PATH, 'N22_UNIT_FIXTURE_existing_oracle')
        cls.analysis = load(Path(__file__).with_name('analysis.py'), 'N22_UNIT_FIXTURE_analysis')

    def analyze(self, rows, counts=None):
        return self.analysis.analyze_instances(rows, audit(len(rows)) if counts is None else counts,
                                               oracle=self.oracle)

    def test_exact_four_methods_crossings_and_descriptive_direction_table(self):
        result = self.analyze([paired()])
        self.assertEqual(result['schema'], 'N22_FIXED_PAIR_ACTION_ANALYSIS_V1')
        methods = result['instances'][0]['methods']
        self.assertEqual(methods['native']['iou'], rational(4, 15))
        self.assertEqual(methods['router_R']['iou'], rational(4, 15))
        self.assertEqual(methods['fixed_PR_beneficial_oracle']['iou'], rational(1, 2))
        self.assertEqual(methods['ideal_C_error_oracle']['iou'], rational(8, 15))
        fixed = result['instances'][0]['crossings_from_native']['fixed_PR_beneficial_oracle']
        self.assertEqual(fixed['up'], [rational(1, 2)])
        self.assertEqual(fixed['down'], [])
        table = result['descriptive_direction_table']
        self.assertEqual([r['direction'] for r in table],
                         ['router_selection', 'fixed_PR_action_availability'])
        self.assertEqual(table[0]['mean_iou_gap'], rational(7, 30))
        self.assertEqual(table[1]['mean_iou_gap'], rational(1, 30))
        self.assertFalse(result['claims']['is_AP_bound'])
        self.assertFalse(result['claims']['research_direction_selected'])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_native_zero_iou_real_pair_kept_in_all_fixed_denominators(self):
        zero = paired(pair_id='UNIT_FIXTURE_PAIR_1', query_index=1, gt_index=1,
                      I=0, U=3, native_fn=1, native_fp=2,
                      candidate_fn=1, candidate_fp=2, candidate_tp=0, candidate_tn=1,
                      pr_beneficial_fn=1, pr_beneficial_fp=2,
                      b_FN=0, b_FP=0, h_TP=0, h_TN=0, K=1, selected_count=1)
        result = self.analyze([paired(), zero])
        self.assertEqual(result['population']['paired_instances'], 2)
        self.assertEqual(result['population']['native_iou_zero_pairs'], 1)
        for row in result['method_table']:
            self.assertEqual(row['pair_denominator'], 2)
        native = result['method_table'][0]
        self.assertEqual(native['mean_iou'], rational(2, 15))
        fixed = result['method_table'][2]
        self.assertEqual(fixed['mean_iou'], rational(5, 12))

    def test_selected_noop_slots_and_K_zero_are_not_invented_flips(self):
        row = paired(K=4, selected_count=4, b_FN=1, b_FP=0, h_TP=0, h_TN=0)
        actual = self.analyze([row])['instances'][0]['methods']['router_R']
        self.assertEqual((actual['flips'], actual['unchanged_selected']), (1, 3))
        zero = paired(K=0, selected_count=0, b_FN=0, b_FP=0, h_TP=0, h_TN=0)
        result = self.analyze([zero])
        for method in result['instances'][0]['methods'].values():
            self.assertEqual(method['iou'], rational(4, 15))

    def test_unassigned_ignored_and_unmatched_GT_are_counts_only(self):
        counts = audit(unassigned_queries=97, ignored_queries=2, unmatched_gt=4, total_real_gt=5)
        result = self.analyze([paired()], counts)
        self.assertEqual(len(result['instances']), 1)
        self.assertEqual(result['population']['unassigned_queries'], 97)
        self.assertEqual(result['population']['ignored_queries'], 2)
        self.assertEqual(result['population']['unmatched_gt'], 4)
        empty = self.analyze([], audit(0, unmatched_gt=3, total_real_gt=3))
        self.assertEqual(empty['instances'], [])
        for method in empty['method_table']:
            self.assertIsNone(method['mean_iou'])
            self.assertEqual(method['pair_denominator'], 0)

    def test_feasibility_and_population_errors_rejected(self):
        invalid = [dict(U=14), dict(native_fn=-1), dict(pr_beneficial_fn=5),
                   dict(pr_beneficial_fp=4), dict(pr_beneficial_fn=0),
                   dict(selected_count=3), dict(selected_count=True),
                   dict(query_index=100), dict(gt_index=-1), dict(pair_id=''),
                   dict(I=0, U=0, native_fn=0, native_fp=0),
                   dict(candidate_tp=5), dict(h_TP=3), dict(K=-1)]
        for change in invalid:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.analyze([paired(**change)])
        for counts in [audit(collected_queries=99), audit(unassigned_queries=98),
                       audit(total_real_gt=9), audit(image_count=True)]:
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    self.analyze([paired()], counts)

    def test_duplicate_pair_query_or_GT_rejected_and_inputs_unchanged(self):
        first = paired()
        duplicates = [paired(query_index=1, gt_index=1),
                      paired(pair_id='UNIT_FIXTURE_PAIR_1', gt_index=1),
                      paired(pair_id='UNIT_FIXTURE_PAIR_1', query_index=1)]
        for second in duplicates:
            with self.assertRaises(ValueError):
                self.analyze([first, second])
        rows = [first]
        counts = audit()
        before = copy.deepcopy((rows, counts))
        self.analyze(rows, counts)
        self.assertEqual((rows, counts), before)


if __name__ == '__main__':
    print('UNIT_FIXTURE_ONLY: no real scientific input, AP or model execution.')
    unittest.main(verbosity=2)
