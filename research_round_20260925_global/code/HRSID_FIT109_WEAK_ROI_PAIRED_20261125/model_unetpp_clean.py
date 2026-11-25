import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(num_channels, norm_type="group", num_groups=8):
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "group":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="group"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels, norm_type=norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels, norm_type=norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CleanUNetPlusPlus(nn.Module):
    """A compact UNet++ baseline with no attention, transformer, DSConv, or coordinate head."""

    def __init__(
        self,
        num_classes=1,
        in_channels=3,
        base_channels=32,
        deep_supervision=False,
        norm_type="group",
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [base_channels * (2 ** i) for i in range(5)]

        self.pool = nn.MaxPool2d(2)
        self.up = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        self.conv0_0 = ConvBlock(in_channels, filters[0], norm_type=norm_type)
        self.conv1_0 = ConvBlock(filters[0], filters[1], norm_type=norm_type)
        self.conv2_0 = ConvBlock(filters[1], filters[2], norm_type=norm_type)
        self.conv3_0 = ConvBlock(filters[2], filters[3], norm_type=norm_type)
        self.conv4_0 = ConvBlock(filters[3], filters[4], norm_type=norm_type)

        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0], norm_type=norm_type)
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1], norm_type=norm_type)
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2], norm_type=norm_type)
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3], norm_type=norm_type)

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0], norm_type=norm_type)
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1], norm_type=norm_type)
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2], norm_type=norm_type)

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0], norm_type=norm_type)
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1], norm_type=norm_type)

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0], norm_type=norm_type)

        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward_features(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, x0_0.shape[2:])], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, x1_0.shape[2:])], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, x2_0.shape[2:])], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, x3_0.shape[2:])], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, x0_0.shape[2:])], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, x1_0.shape[2:])], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, x2_0.shape[2:])], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, x0_0.shape[2:])], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, x1_0.shape[2:])], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, x0_0.shape[2:])], dim=1))

        return {
            "x0_1": x0_1,
            "x0_2": x0_2,
            "x0_3": x0_3,
            "x0_4": x0_4,
        }

    def forward(self, x):
        features = self.forward_features(x)

        if self.deep_supervision:
            return [
                self.final1(features["x0_1"]),
                self.final2(features["x0_2"]),
                self.final3(features["x0_3"]),
                self.final4(features["x0_4"]),
            ]
        return self.final4(features["x0_4"])


UnetPlusPlus = CleanUNetPlusPlus
