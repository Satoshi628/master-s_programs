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
from .Self_attention import SelfAttention


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            Mish(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            Mish()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GT_PDA(nn.Module):
    def __init__(self, in_channels, n_classes, scale, weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.scale = scale
        self.weight_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(n_classes),
            nn.Upsample(scale_factor=scale, mode='bilinear')
        )
        #classesのlossを計算するためのcross entropy loss
        self.criterion = CrossEntropy(weight=weights)

        #lossを保存するもの一度実行すると更新されるため注意
        self.loss = 0.

    def forward(self, x, targets=None):
        #upsampleとsoftmaxを印加,[batch,class,H,W]
        classes = self.weight_conv(x)

        #cross entropy lossを計算
        if targets is not None:
            self.loss = self.criterion(classes, targets)

        if self.training & (targets is not None):
        #if targets is not None:
            #targetsのclassの箇所を抽出する
            targets_idx = targets.max(dim=1, keepdim=True)[1]
            weights = torch.gather(classes, dim=1, index=targets_idx)
        else:
            weights = classes.max(dim=1, keepdim=True)[0]

        #確信度が低い物(難易度が高いものにweightをかける)
        weights = 1. - weights

        #形をdown sampleして元のxの形に戻す
        weights = F.interpolate(weights, x.shape[-2:], mode='bilinear')
        
        x = x * weights + x

        return x, weights

    def get_loss(self):
        return self.loss


class Generator(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32, class_weights=None):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        #Conv層定義
        self.inc = DoubleConv(in_channels, channel)
        self.down1 = Down(channel, channel * 2)
        self.down2 = Down(channel * 2, channel * 4)
        self.down3 = Down(channel * 4, channel * 8)
        self.down4 = Down(channel * 8, channel * 8)
        
        self.up1 = Up(channel*16, channel*4)
        self.up2 = Up(channel*8, channel*2)
        self.up3 = Up(channel*4, channel)
        self.up4 = Up(channel*2, channel)
        self.outc = nn.Conv2d(channel, n_classes, kernel_size=1, bias=True)

        #Attention定義
        self.attention_d1 = SelfAttention(channel * 2, channel * 2, scale=4, d_k=channel)
        self.attention_d2 = SelfAttention(channel * 4, channel * 4, scale=4, d_k=channel)
        self.attention_d3 = SelfAttention(channel * 8, channel * 8, scale=2, d_k=channel)
        self.attention_d4 = SelfAttention(channel * 8, channel * 16, scale=2, d_k=channel)

        #GT_PDA定義
        self.gt_pda1 = GT_PDA(channel, self.n_classes, scale=2, weights=class_weights)
        self.gt_pda2 = GT_PDA(channel * 2, self.n_classes, scale=4, weights=class_weights)
        self.gt_pda3 = GT_PDA(channel * 4, self.n_classes, scale=8, weights=class_weights)

        #classesのlossを計算するためのcross entropy loss
        self.criterion = CrossEntropy(weight=class_weights)
        #lossを保存するもの一度実行すると更新されるため注意
        self.loss = 0.

    def forward(self, x, targets=None, memory=None):
        if memory is None:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

        else:
            x1 = self.inc(x)

            x2 = self.down1(x1)
            
            x2 = self.attention_d1(x2, memory[0])
            
            x3 = self.down2(x2)
            x3 = self.attention_d2(x3, memory[1])
            
            x4 = self.down3(x3)
            x4 = self.attention_d3(x4, memory[2])
            
            x5 = self.down4(x4)

            x5 = self.attention_d4(x5, memory[3])

        out1 = self.up1(x5, x4)
        out1, weights1 = self.gt_pda3(out1, targets=targets)  # trainモード時のみ実行

        out2 = self.up2(out1, x3)
        out2, weights2 = self.gt_pda2(out2, targets=targets)  # trainモード時のみ実行

        out3 = self.up3(out2, x2)
        out3, weights3 = self.gt_pda1(out3, targets=targets)  # trainモード時のみ実行

        x = self.up4(out3, x1)

        out = self.outc(x)
        if targets is not None:
            self.loss = self.criterion(out, targets)
        
        return out, [weights3, weights2, weights1]
    
    def get_Loss(self,lamda=[1.,1.,1.,1.]):
        loss = lamda[0] * self.loss + \
            lamda[1] * self.gt_pda1.get_loss() + \
            lamda[2] * self.gt_pda2.get_loss() + \
            lamda[3] * self.gt_pda3.get_loss()
        return loss



#loss定義
class CrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is None:
            self.weights = [1.]
        else:
            self.weights = weight

    def forward(self, inputs, targets):
        out = F.softmax(inputs, dim=1)
        out = torch.clamp(out,min=1e-7)
        logit = targets * torch.log(out)
        weights = inputs.new_tensor(self.weights)[None, :, None, None]
        logit = weights * logit
        return -logit.mean()
