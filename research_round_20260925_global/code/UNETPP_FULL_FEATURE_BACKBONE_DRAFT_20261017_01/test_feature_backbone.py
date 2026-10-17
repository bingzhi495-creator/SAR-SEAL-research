"""Pending real-legacy size64 contract, not a fake-full or executed result.

Root must supply the actual source-closed model_full.UnetPlusPlus class and
admitted Torch to make_real_legacy_suite. No Torch import or legacy construction
occurs at import. Factory import comes first so missing-module RED stays clean.
This test's resource/raster claim is restricted to one64x64 image in eval mode.
"""
import unittest

import feature_backbone as factory_module


def make_real_legacy_suite(torch,legacy_class):
    """Root owns loading the real class and any future actual numerical run."""
    if legacy_class.__name__!='UnetPlusPlus':
        raise ValueError('Supply the actual source-closed model_full.UnetPlusPlus class')

    class RealLegacySize64Contract(unittest.TestCase):
        def test_feature_forward_matches_real_legacy_with_temporary_identity_semantic_head(self):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(202610173201)
                legacy=legacy_class(num_classes=1,deep_supervision=False,use_x2_transformer=False,
                    input_size=64,instantiate_x2_transformer=False,include_legacy_x4_transformer=False)
                legacy.eval()
                original_modes={name:module.training for name,module in legacy.named_modules()}
                original_flags={name:parameter.requires_grad for name,parameter in legacy.named_parameters()}
                original_ids={name:id(parameter) for name,parameter in legacy.named_parameters()}
                backbone=factory_module.make_feature_backbone(torch,legacy)
                self.assertIs(backbone.legacy,legacy)
                self.assertEqual({name:module.training for name,module in legacy.named_modules()},original_modes)
                self.assertEqual({name:parameter.requires_grad for name,parameter in legacy.named_parameters()},original_flags)
                self.assertEqual({name:id(parameter) for name,parameter in legacy.named_parameters()},original_ids)
                self.assertIs(dict(backbone.named_modules())['legacy.CONV0_4'],legacy.CONV0_4)
                image=torch.randn(1,3,64,64)
                original_head=legacy.final_super_0_4
                try:
                    legacy.final_super_0_4=torch.nn.Identity()
                    with torch.no_grad():expected=legacy(image)
                finally:
                    legacy.final_super_0_4=original_head
                with torch.no_grad():actual=backbone(image)
                self.assertIs(legacy.final_super_0_4,original_head)
                self.assertEqual(tuple(actual.shape),(1,64,64,64))
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    return unittest.defaultTestLoader.loadTestsFromTestCase(RealLegacySize64Contract)
