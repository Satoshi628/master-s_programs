#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

#----- 自作モジュール -----#
#None

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class spectral_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(spectral_conv, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        conv = nn.utils.spectral_norm(conv)
        
        self.conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            Mish()
        )

    def __call__(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32):
        super(Discriminator, self).__init__()

        self.inputs_conv = nn.Conv2d(in_channels, channel // 2, kernel_size=3, padding=1)
        self.seg_conv = nn.Conv2d(n_classes, channel // 2, kernel_size=3, padding=1)

        self.down1 = spectral_conv(channel, channel * 2)
        self.down2 = spectral_conv(channel * 2, channel * 4)
        self.down3 = spectral_conv(channel * 4, channel * 8)
        self.down4 = spectral_conv(channel * 8, channel * 16)
        self.outc = nn.Conv2d(channel * 16, 1, kernel_size=1)

    def forward(self, inputs, segmentation, weight=None):

        x = torch.cat([F.leaky_relu(self.inputs_conv(inputs)),
                        F.leaky_relu(self.seg_conv(segmentation))], dim=1)
        
        x1 = self.down1(x)
        if weight is not None:
            x1 = x1 + x1 * weight[0]
        
        x2 = self.down2(x1)
        if weight is not None:
            x2 = x2 + x2 * weight[1]
        
        x3 = self.down3(x2)
        if weight is not None:
            x3 = x3 + x3 * weight[2]

        x4 = self.down4(x3)

        out = self.outc(x4)

        return out, [x1, x2, x3, x4]
