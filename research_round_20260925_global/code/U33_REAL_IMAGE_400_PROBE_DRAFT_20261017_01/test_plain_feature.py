"""One public feature-equivalence test; importing this file never imports Torch.

The first import intentionally requires the not-yet-implemented public factory.
ROOT owns the actual MissingModule RED and any later numerical execution.
After that, supply the reviewed Torch module and the source-closed legacy class
to make_real_plain_suite(torch, legacy_class); no model is built at import time.
"""

from plain_feature import make_plain_feature_backbone

import unittest


def make_real_plain_suite(torch, legacy_class):
    """Build ONE test using model1.UnetPlusPlus's actual two-argument ABI."""

    class RealPlainFeatureTest(unittest.TestCase):
        def test_original_feature_bitexact_and_legacy_identity_preserved(self):
            def snapshot(model):
                return {
                    "modules": tuple((name, id(value)) for name, value in model.named_modules()),
                    "parameters": tuple((name, id(value)) for name, value in model.named_parameters()),
                    "training": tuple((name, value.training) for name, value in model.named_modules()),
                    "requires_grad": tuple((name, value.requires_grad) for name, value in model.named_parameters()),
                }

            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(3101)
                # This archived class has no input_size/base_channels keyword.
                legacy = legacy_class(num_classes=1, deep_supervision=False).eval()
                original = snapshot(legacy)
                x = torch.linspace(-0.75, 0.75, 3 * 64 * 64,
                                   dtype=torch.float32, device="cpu").reshape(1, 3, 64, 64)
                old_head = legacy.final_super_0_4
                try:
                    legacy.final_super_0_4 = torch.nn.Identity()
                    with torch.no_grad():
                        expected = legacy(x)
                finally:
                    legacy.final_super_0_4 = old_head

                self.assertEqual(snapshot(legacy), original)
                self.assertEqual(tuple(expected.shape), (1, 64, 64, 64))

                feature_backbone = make_plain_feature_backbone(torch, legacy, input_size=64)
                self.assertEqual(snapshot(legacy), original)
                with torch.no_grad():
                    actual = feature_backbone(x)

                self.assertEqual(tuple(actual.shape), (1, 64, 64, 64))
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, expected.device)
                self.assertTrue(bool(torch.isfinite(expected).all()))
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertTrue(torch.equal(actual, expected), "Plain feature differs from original CONV0_4")
                self.assertIs(legacy.final_super_0_4, old_head)
                self.assertEqual(snapshot(legacy), original)

    return unittest.TestSuite([
        RealPlainFeatureTest("test_original_feature_bitexact_and_legacy_identity_preserved")
    ])
