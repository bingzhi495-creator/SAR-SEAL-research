"""Feature-only wrapper for archive/old_code/Unet++/model1.UnetPlusPlus.

Torch and a source-verified, already constructed legacy instance are supplied
by the caller. Importing this module constructs no model and imports no Torch.
The wrapper registers that same instance, preserves its modules/parameters and
their modes/flags, and returns CONV0_4 without evaluating any semantic head.
"""


def make_plain_feature_backbone(torch, legacy, input_size=400):
    """Wrap the existing plain64 model without initialization or graph changes."""
    if type(input_size) is not int or input_size < 32 or input_size % 16:
        raise ValueError("input_size must be an integer >=32 divisible by16")
    if not isinstance(legacy, torch.nn.Module):
        raise TypeError("legacy must be the supplied model1.UnetPlusPlus instance")
    if getattr(legacy, "filters", None) != [64, 128, 256, 512, 1024]:
        raise ValueError("The original plain64 five-scale legacy model is required")

    class PlainFeatureBackbone(torch.nn.Module):
        feature_module_name = "legacy.CONV0_4"
        feature_channels = 64

        def __init__(self):
            super().__init__()
            self.legacy = legacy
            self.input_size = input_size
            # Avoid .train(...) here: it would overwrite mixed child modes.
            self.training = legacy.training

        def inactive_parameter_names(self):
            """Only the four skipped semantic heads are inactive in model1."""
            prefixes = tuple("legacy.final_super_0_%d." % i for i in (1, 2, 3, 4))
            return tuple(sorted(name for name, _ in self.named_parameters()
                                if name.startswith(prefixes)))

        def forward(self, x):
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a Torch tensor")
            if tuple(x.shape) != (1, 3, self.input_size, self.input_size):
                raise ValueError("Expected one three-channel image on the declared square grid")

            m = self.legacy
            # Original model1 forward, in its original operation/cat order.
            x_0_0 = m.stage_0(x)
            x_1_0 = m.stage_1(m.pool(x_0_0))
            x_2_0 = m.stage_2(m.pool(x_1_0))
            x_3_0 = m.stage_3(m.pool(x_2_0))
            x_4_0 = m.stage_4(m.pool(x_3_0))

            x_0_1 = torch.cat([m.upsample_0_1(x_1_0), x_0_0], 1)
            x_0_1 = m.CONV0_1(x_0_1)

            x_1_1 = torch.cat([m.upsample_1_1(x_2_0), x_1_0], 1)
            x_1_1 = m.CONV1_1(x_1_1)

            x_2_1 = torch.cat([m.upsample_2_1(x_3_0), x_2_0], 1)
            x_2_1 = m.CONV2_1(x_2_1)

            x_3_1 = torch.cat([m.upsample_3_1(x_4_0), x_3_0], 1)
            x_3_1 = m.CONV3_1(x_3_1)

            x_2_2 = torch.cat([m.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
            x_2_2 = m.CONV2_2(x_2_2)

            x_1_2 = torch.cat([m.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
            x_1_2 = m.CONV1_2(x_1_2)

            x_1_3 = torch.cat([m.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
            x_1_3 = m.CONV1_3(x_1_3)

            x_0_2 = torch.cat([m.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
            x_0_2 = m.CONV0_2(x_0_2)

            x_0_3 = torch.cat([m.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
            x_0_3 = m.CONV0_3(x_0_3)

            x_0_4 = torch.cat([m.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
            x_0_4 = m.CONV0_4(x_0_4)
            return x_0_4

    return PlainFeatureBackbone()
