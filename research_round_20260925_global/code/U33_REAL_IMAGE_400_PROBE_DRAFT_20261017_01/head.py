import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    def __init__(self, in_channels, drop=0.5, with_r=False, use_attention=True):
        super().__init__()
        self.with_r = with_r
        self.use_attention = use_attention
        if use_attention:
            self.pa = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(True),

                nn.Conv2d(in_channels//4, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid(),
            )


    def forward(self, input_tensor):
        """
        Args:
        input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
   
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        
        if self.use_attention:
            input_tensor1=self.pa(input_tensor)
            input_tensor2=input_tensor*input_tensor1;
        else:
            input_tensor2=input_tensor

        ret = torch.cat([
        input_tensor2,
        xx_channel.type_as(input_tensor),
        yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
            torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False, use_attention=True):
        super().__init__()
        self.addcoords = AddCoords(in_channels,with_r=with_r,use_attention=use_attention)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        )

    def forward(self, x):

        x = self.addcoords(x)
        x = self.conv(x)
        return x


