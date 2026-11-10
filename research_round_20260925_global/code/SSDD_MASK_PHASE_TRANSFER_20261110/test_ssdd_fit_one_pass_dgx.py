import unittest

import torch
from rfdetr.datasets import _torchvision as native

import ssdd_fit_one_pass_dgx as audit
from test_paired_fit_transform import compose, fixture


def seed_cpu(seed):
    torch.random.default_generator.manual_seed(seed)


class OnePassAuditTest(unittest.TestCase):
    def test_direct_stock_mask_is_exact_and_restores_callers_rng(self):
        image, raw = fixture()
        transform = compose(native.Resize((5, 5)), native.RandomHorizontalFlip(p=.5))
        before = torch.random.get_rng_state()
        result = audit.audit_view(native, transform, image, raw, 5311,
                                  set_seed=seed_cpu, helpers=audit.load_audit_helpers())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        self.assertEqual(result["path"], "direct")
        self.assertTrue(result["direct_fixed_exact"])
        self.assertEqual(result["common_changed_masks"], 0)
        self.assertTrue(result["c_natural_oracle_exact"])
        self.assertTrue(result["c_b_conditioned_oracle_exact"])

    def test_crop_keeps_natural_empty_carrier_separate_from_fixed_support(self):
        image, raw = fixture()
        raw["boxes"][1] = torch.tensor([1., 1., 6., 6.])
        raw["masks"][1].zero_()
        raw["masks"][1, 0, 0] = True

        def crop(image_input, target_input):
            return native.crop(image_input, target_input, top=1, left=1, height=3, width=3)

        helpers = audit.load_audit_helpers()
        transform = compose(native.Resize((5, 5)), crop, native.Resize((4, 4)), native.RandomHorizontalFlip(p=1))
        view = audit.audit_view(native, transform, image, raw, 5312, set_seed=seed_cpu, helpers=helpers)
        self.assertEqual((view["path"], view["flip"]), ("crop", True))
        self.assertEqual((view["b_present"], view["c_present"], view["c_only_present"], view["c_empty"]), (1, 2, 1, 1))
        self.assertEqual((view["common_nonempty"], view["common_changed_masks"]), (1, 1))
        self.assertTrue(view["c_natural_oracle_exact"] and view["c_b_conditioned_oracle_exact"])
        direct = audit.audit_view(native, compose(native.Resize((5, 5))), image, raw, 5311,
                                  set_seed=seed_cpu, helpers=helpers)
        result = audit.summarize_audits([direct, view])
        self.assertEqual(result["decision"], "SUPPORTS_FIXED_ROW_ONE_PASS_CROP_CHANGE")
        self.assertEqual(result["by_path"]["crop"]["c_only_present"], 1)
        self.assertEqual(result["c_natural_oracle_exact_views"], 2)

    def test_empty_target_remains_zero_rows_in_both_actual_samplers(self):
        image, raw = fixture("empty")
        view = audit.audit_view(native, compose(native.Resize((5, 5))), image, raw, 5311,
                                set_seed=seed_cpu, helpers=audit.load_audit_helpers())
        self.assertEqual((view["b_present"], view["c_present"], view["common_nonempty"]), (0, 0, 0))
        self.assertTrue(view["direct_fixed_exact"] and view["c_b_conditioned_oracle_exact"])


if __name__ == "__main__":
    unittest.main()
