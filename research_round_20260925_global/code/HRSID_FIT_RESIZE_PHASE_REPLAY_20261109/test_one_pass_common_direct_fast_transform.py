import unittest
import warnings

from PIL import Image
import torch
from rfdetr.datasets import _torchvision as native, coco

from one_pass_common_direct_fast_transform import OnePassCommonDirectFastTransform
from one_pass_common_pair_transform import OnePassCommonPairTransform
from test_paired_fit_transform import draws, fixture, set_seed


def stock_fixture():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return coco.make_coco_transforms_square_div_64(
            "train", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
            patch_size=12, num_windows=1, aug_config=None, scale_jitter=True,
            gpu_postprocess=False, keypoint_flip_pairs=None,
        )


def direct_fixture():
    image = Image.new("RGB", (800, 800), (11, 29, 97))
    masks = torch.zeros((2, 800, 800), dtype=torch.bool)
    masks[0, 80:200, 151:270] = True
    masks[1, 1, 1] = True
    target = dict(
        boxes=torch.tensor([[151., 80., 270., 200.], [1., 1., 2., 2.]]),
        labels=torch.zeros(2, dtype=torch.int64), masks=masks,
        area=torch.tensor([14280., 1.]), iscrowd=torch.zeros(2, dtype=torch.int64),
        image_id=torch.tensor([31]), size=torch.tensor([800, 800]), orig_size=torch.tensor([800, 800]),
    )
    return image, target


class OnePassCommonDirectFastTransformTest(unittest.TestCase):
    def test_stock_direct_is_exact_both_arms_and_preserves_rng(self):
        image, target = direct_fixture()
        stock = stock_fixture()
        for mode in ("stepwise_center", "one_pass"):
            with self.subTest(mode=mode):
                expected_audit, actual_audit = [], []
                original = OnePassCommonPairTransform(native, stock, mode, audit=expected_audit.append)

                def audit(row):
                    actual_audit.append(row)
                    draws()

                fast = OnePassCommonDirectFastTransform(native, stock, mode, audit=audit)
                set_seed()
                torch.random.default_generator.manual_seed(3)
                expected_image, expected_target = original(image, target)
                expected_rng = draws()
                set_seed()
                torch.random.default_generator.manual_seed(3)
                actual_image, actual_target = fast(image, target)
                self.assertEqual(draws(), expected_rng)
                self.assertEqual(fast.fast_direct_visits, 1)
                self.assertEqual(actual_audit, expected_audit)
                self.assertFalse(actual_audit[0]["crop"])
                self.assertTrue(actual_audit[0]["flipped"])
                self.assertEqual(actual_audit[0]["native_present"], 2)
                self.assertEqual(actual_audit[0]["common_nonempty"], 1)
                self.assertTrue(torch.equal(actual_image, expected_image))
                self.assertEqual(actual_target.keys(), expected_target.keys())
                self.assertTrue(all(torch.equal(actual_target[key], expected_target[key]) for key in actual_target))



    def test_stock_no_flip_empty_direct_and_crop_fallback_match(self):
        image, target = direct_fixture()
        stock = stock_fixture()
        fields = {"boxes", "labels", "area", "iscrowd", "masks"}
        empty = {key: value[:0] if key in fields else value for key, value in target.items()}
        for seed, source, is_crop in ((0, empty, False), (1, target, True)):
            with self.subTest(seed=seed):
                before, after = [], []
                original = OnePassCommonPairTransform(native, stock, "one_pass", audit=before.append)
                fast = OnePassCommonDirectFastTransform(native, stock, "one_pass", audit=after.append)
                set_seed()
                torch.random.default_generator.manual_seed(seed)
                expected_image, expected_target = original(image, source)
                expected_rng = draws()
                set_seed()
                torch.random.default_generator.manual_seed(seed)
                actual_image, actual_target = fast(image, source)
                self.assertEqual(draws(), expected_rng)
                self.assertEqual(fast.fast_direct_visits, 0 if is_crop else 1)
                self.assertEqual(before, after)
                self.assertIs(after[0]["crop"], is_crop)
                self.assertTrue(torch.equal(actual_image, expected_image))
                self.assertTrue(all(torch.equal(actual_target[key], expected_target[key]) for key in actual_target))
                if not is_crop:
                    self.assertFalse(after[0]["flipped"])
                    self.assertEqual(actual_target["masks"].shape, (0, 576, 576))

    def test_custom_callable_and_nested_custom_operation_keep_the_exact_guard(self):
        image, target = direct_fixture()
        original_stock = stock_fixture()
        modified_stock = stock_fixture()
        original_resize = modified_stock.transforms[0].transform1.transforms[0]

        def change_output(transform):
            def changed(image_input, target_input):
                output, converted = transform(image_input, target_input)
                converted["masks"][0, 0, 0] = ~converted["masks"][0, 0, 0]
                return output, converted

            return changed

        class Custom:
            @property
            def transforms(self):
                raise AssertionError("Do not inspect custom callable internals")

            def __call__(self, image_input, target_input):
                return change_output(original_stock)(image_input, target_input)

        modified_stock.transforms[0].transform1.transforms[0] = change_output(original_resize)
        for custom in (Custom(), modified_stock):
            with self.subTest(custom=type(custom).__name__):
                fast = OnePassCommonDirectFastTransform(native, custom, "one_pass")
                set_seed()
                torch.random.default_generator.manual_seed(3)
                expected_rng = draws()
                set_seed()
                torch.random.default_generator.manual_seed(3)
                with self.assertRaisesRegex(ValueError, "Direct FIT masks must match exactly"):
                    fast(image, target)
                self.assertEqual(draws(), expected_rng)
                self.assertEqual(fast.fast_direct_visits, 0)

    def test_other_shapes_fall_back_and_failed_audit_leaves_no_fast_count(self):
        stock = stock_fixture()
        small_image, small_target = fixture()
        fast = OnePassCommonDirectFastTransform(native, stock, "one_pass")
        original = OnePassCommonPairTransform(native, stock, "one_pass")
        set_seed()
        torch.random.default_generator.manual_seed(0)
        expected_image, expected_target = original(small_image, small_target)
        set_seed()
        torch.random.default_generator.manual_seed(0)
        actual_image, actual_target = fast(small_image, small_target)
        self.assertEqual(fast.fast_direct_visits, 0)
        self.assertTrue(torch.equal(actual_image, expected_image))
        self.assertTrue(all(torch.equal(actual_target[key], expected_target[key]) for key in actual_target))

        def bad_audit(_row):
            draws()
            raise RuntimeError("audit refused")

        image, target = direct_fixture()
        fast = OnePassCommonDirectFastTransform(native, stock, "one_pass", audit=bad_audit)
        set_seed()
        torch.random.default_generator.manual_seed(0)
        expected_rng = draws()
        set_seed()
        torch.random.default_generator.manual_seed(0)
        with self.assertRaisesRegex(RuntimeError, "audit refused"):
            fast(image, target)
        self.assertEqual(fast.fast_direct_visits, 0)
        self.assertEqual(draws(), expected_rng)


if __name__ == "__main__":
    unittest.main()
