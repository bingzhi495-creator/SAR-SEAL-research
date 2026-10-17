import torch
import torch.nn as nn
from dsconv import DSConv
import numpy as np
import torch.nn.functional as F
from head import CoordConv

class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x

class EncoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool = BlurPool(in_planes, stride=2)
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        max= self.pool(x)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.sim = SimAM(out_channels, out_channels)
        self.pool = BlurPool(out_channels, stride=2)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual

        out = self.relu(out)
        out3 = self.sim(out)
        return (out + out3)
    # def forward(self, x):
    #     residual = x
    #     if self.shortcut is not None:
    #         residual = self.shortcut(x)
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out1 = self.ca(out) * out
    #     # out2 = self.sa(out) * out
    #     out3 = self.sim(out)
    #     out = out1+out3
    #     out += out2
    #     out = self.relu(out)
    # return (out)


class SimAM(torch.nn.Module):
    def __init__(self, channels, out_channels, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class channel_shuffle(nn.Module):
    def __init__(self, groups=4):
        super(channel_shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=1, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class CBAM_block_up(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(CBAM_block_up, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class UnetPlusPlus(nn.Module):
    def __init__(
        self,
        num_classes,
        deep_supervision=False,
        use_x2_transformer=True,
        input_size=512,
        instantiate_x2_transformer=None,
        include_legacy_x4_transformer=False,
        coordconv_attention=True,
    ):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.use_x2_transformer = use_x2_transformer
        self.input_size = int(input_size)
        if instantiate_x2_transformer is None:
            instantiate_x2_transformer = True
        if self.use_x2_transformer and not instantiate_x2_transformer:
            raise ValueError("use_x2_transformer=True requires instantiate_x2_transformer=True")
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        #self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        #self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        #self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        #self.ds_stage_4x = DSConv(1024, 1024, 9, 1, 0, if_offset=True)
        #self.ds_stage_4y = DSConv(1024, 1024, 9, 1, 1, if_offset=True)
        #self.con4 = EncoderConv(3072, 1024)
        self.ds_stage_2x = DSConv(256, 256, 9, 1, 0, if_offset=True)
        self.ds_stage_2y = DSConv(256, 256, 9, 1, 1, if_offset=True)
        self.con2 = EncoderConv(768, 256)

        #self.ds_stage_1x = DSConv(128, 128, 9, 1, 0, if_offset=True)
        #self.ds_stage_1y = DSConv(128, 128, 9, 1, 1, if_offset=True)
        #self.con1 = EncoderConv(384, 128)



        self.pool = nn.MaxPool2d(2)

        self.cbam_stage_0 = CBAM_block(64,64)
        self.cbam_stage_1 = CBAM_block(64,128)
        self.cbam_stage_2 = CBAM_block(128,256)
        self.cbam_stage_3 = CBAM_block(256,512)
        self.cbam_stage_4 = CBAM_block(512,1024)

        self.cbam_stage_up_1_1 = CBAM_block_up(128*2, 128)
        self.cbam_stage_up_1_2 = CBAM_block_up(128*3, 128)
        self.cbam_stage_up_1_3 = CBAM_block_up(128*4, 128)
        self.cbam_stage_up_2_1 = CBAM_block_up(256*2, 256)
        self.cbam_stage_up_2_2 = CBAM_block_up(256*3, 256)
        self.cbam_stage_up_3_1 = CBAM_block_up(512*2, 512)




        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = CoordConv(64,1,use_attention=coordconv_attention)

        x2_tokens = (self.input_size // 4) * (self.input_size // 4)
        if instantiate_x2_transformer:
            self.encoder_x2 = torch.nn.TransformerEncoderLayer(x2_tokens, 4, x2_tokens, dropout=0.5)
        else:
            self.encoder_x2 = None
        if include_legacy_x4_transformer:
            x4_tokens = (self.input_size // 16) * (self.input_size // 16)
            self.encoder_x4 = torch.nn.TransformerEncoderLayer(x4_tokens, 4, x4_tokens, dropout=0.5)
        #self.encoder_x4 = torch.nn.TransformerEncoderLayer(16 * 16, 4, 16 * 16, dropout=0.5)
        #self.encoder_x0 = torch.nn.TransformerEncoderLayer(256 * 256, 4, 256 * 256, dropout=0.5)
        #self.encoder_x3 = torch.nn.TransformerEncoderLayer(32 * 32, 4, 32 * 32, dropout=0.5)
        #self.encoder_x1 = torch.nn.TransformerEncoderLayer(128 * 128, 4, 128 * 128, dropout=0.5)


    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_0_0 = self.cbam_stage_0(x_0_0)
        #x_1_0 = self.stage_1(self.pool(x_0_0))
        #x_1_0_x = self.ds_stage_1x(x_1_0)
        #x_1_0_y = self.ds_stage_1y(x_1_0)
        x_1_0 = self.cbam_stage_1(self.pool(x_0_0))
        #x_1_0 = self.con1(torch.cat([x_1_0, x_1_0_x, x_1_0_y], dim=1))

        x_2_0 = self.cbam_stage_2(self.pool(x_1_0))
        x_2_0_1 = self.stage_2(self.pool(x_1_0))
        old_shape = x_2_0.shape
        x_2_0 = x_2_0.view(old_shape[0], old_shape[1], old_shape[2] * old_shape[3])
        if self.use_x2_transformer:
            if self.encoder_x2 is None:
                raise RuntimeError("encoder_x2 is not instantiated but use_x2_transformer=True")
            x_2_0 = self.encoder_x2(x_2_0)
        x_2_0 = x_2_0.view(old_shape)
        x_2_0_x = self.ds_stage_2x(x_2_0_1)
        x_2_0_y = self.ds_stage_2y(x_2_0_1)
        #x_2_0 = self.cbam_stage_2(self.pool(x_1_0))
        x_2_0 = self.con2(torch.cat([x_2_0, x_2_0_x, x_2_0_y], dim=1))


        #x_3_0 = self.stage_3(self.pool(x_2_0))
        x_3_0 = self.cbam_stage_3(self.pool(x_2_0))

        x_4_0 = self.cbam_stage_4(self.pool(x_3_0))
        #x_4_0 = self.stage_4(self.pool(x_3_0))
        #old_shape = x_4_0.shape
        #x_4_0 = x_4_0.view(old_shape[0],old_shape[1],old_shape[2]*old_shape[3])
        #old_x4_0 = x_4_0
        #old_x4_0 = x_4_0[ :len(x_4_0)//2, :, :]
        #x_4_0 = x_4_0[len(x_4_0)//2:, :, :]
        #x_4_0 = self.encoder_x4(x_4_0)
        ##x_4_0 = torch.cat([old_x4_0, x_4_0], dim=0)
        #x_4_0 = x_4_0.view(old_shape)
        #x_4_0_x = self.ds_stage_4x(x_4_0)
        #x_4_0_y = self.ds_stage_4y(x_4_0)
        #x_4_0 = self.cbam_stage_4(self.pool(x_3_0))
        #x_4_0 = self.con4(torch.cat([x_4_0, x_4_0_x, x_4_0_y],dim=1))


        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.cbam_stage_up_1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.cbam_stage_up_2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.cbam_stage_up_3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.cbam_stage_up_2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.cbam_stage_up_1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.cbam_stage_up_1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)


if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 400, 400)).to(device)
    model = UnetPlusPlus(num_classes=1, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)

    print("deep_supervision: True")
    deep_supervision = True
    model = UnetPlusPlus(num_classes=1, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
        print(out.shape)

