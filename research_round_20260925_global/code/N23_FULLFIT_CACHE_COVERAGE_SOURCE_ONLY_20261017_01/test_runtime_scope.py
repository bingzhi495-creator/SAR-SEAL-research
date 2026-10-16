"""Synthetic metadata only: keep the intake on the complete original FIT population."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name('run_intake.py')
ms = importlib.util.spec_from_file_location('n23_runtime_scope', path)
runtime = importlib.util.module_from_spec(ms)
ms.loader.exec_module(runtime)


def fixture():
    groups, ordinal = [], 0
    for image in range(741):
        n = 3 if image < 179 else 2
        groups.append(list(range(ordinal, ordinal + n)))
        ordinal += n
    pre = dict(schema='N23-full-FIT-original-sampled-coverage-prelock-v1',
        no_new_training=True, no_DEV=True, no_TEST=True, no_AP=True,
        expected_images=741, expected_real_GT=1661, original_draws_per_image=4096,
        dataset_indices=list(range(741)), fit_image_ids=list(range(1000,1741)),
        fit_ann_ids_by_image=groups, population_filter=None)
    spec = dict(selected_dataset_indices=list(range(741)))
    return pre, spec


class PopulationScopeTests(unittest.TestCase):
    def test_complete_unfiltered_population_is_admitted_without_mutation(self):
        pre, spec = fixture()
        saved = copy.deepcopy((pre, spec))
        runtime.validate_prelock(pre, spec)
        self.assertEqual((pre, spec), saved)

    def test_prior_32_image_diagnostic_is_not_this_intake(self):
        pre, spec = fixture()
        pre.update(schema='N22-FIT-full-C-fixed-action-opportunity-prelock-v1', expected_images=32)
        with self.assertRaises(AssertionError):
            runtime.validate_prelock(pre, spec)

    def test_missing_duplicate_or_reordered_image_is_rejected(self):
        for mutate in (
            lambda p,s:p['fit_image_ids'].pop(),
            lambda p,s:p['fit_image_ids'].__setitem__(1,p['fit_image_ids'][0]),
            lambda p,s:s['selected_dataset_indices'].reverse(),
            lambda p,s:p['dataset_indices'].__setitem__(1,0),
        ):
            pre, spec = fixture(); mutate(pre, spec)
            with self.assertRaises(AssertionError):
                runtime.validate_prelock(pre, spec)

    def test_real_GT_scope_cannot_be_replaced_by_a_claimed_count(self):
        for mutate in (
            lambda p:p['fit_ann_ids_by_image'][0].pop(),
            lambda p:p['fit_ann_ids_by_image'][1].__setitem__(0,p['fit_ann_ids_by_image'][0][0]),
            lambda p:p.update(expected_real_GT=1660),
        ):
            pre, spec = fixture(); mutate(pre)
            with self.assertRaises(AssertionError):
                runtime.validate_prelock(pre, spec)

    def test_outcome_filter_or_extra_scientific_scope_is_rejected(self):
        for key, value in [('population_filter',{'IoU_min':.5}),('no_new_training',False),
                           ('no_DEV',False),('no_TEST',False),('no_AP',False),
                           ('original_draws_per_image',1024)]:
            pre, spec = fixture();pre[key]=value
            with self.assertRaises(AssertionError):
                runtime.validate_prelock(pre, spec)


if __name__ == '__main__':
    unittest.main()
