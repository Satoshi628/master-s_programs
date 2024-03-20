#coding: utf-8
#----- 標準ライブラリ -----#
import sys

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

#----- 自作モジュール -----#
#None


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


class Class_contrastive_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, featrue_channels, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.contrastive_conv = nn.Sequential(
            nn.Conv2d(in_channels, featrue_channels * n_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(featrue_channels * n_classes)
        )

    def forward(self, x):
        out = self.contrastive_conv(x)
        class_feature = torch.chunk(out, self.n_classes, dim=1)
        return class_feature, out



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


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))

#FReLU
class FReLU(nn.Module):
    def __init__(self, in_c, k=3, s=1, p=1):
        super().__init__()
        #Depthwise Convolution
        #https://ai-scholar.tech/articles/treatise/mixconv-ai-367
        self.FReLU = nn.Conv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c, bias=False)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        tx = self.bn(self.FReLU(x))
        out = torch.max(x, tx)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, channel)
        self.down1 = Down(channel, channel*2)
        self.down2 = Down(channel*2, channel*4)
        self.down3 = Down(channel*4, channel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(channel*8, channel*16 // factor)
        self.up1 = Up(channel*16, channel*8 // factor, bilinear)
        self.up2 = Up(channel*8, channel*4 // factor, bilinear)
        self.up3 = Up(channel*4, channel*2 // factor, bilinear)
        self.up4 = Up(channel*2, channel, bilinear)
        self.outc = nn.Conv2d(channel, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

    def get_feature(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        return feature

class UNet_contrastive(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32, bilinear=True):
        super(UNet_contrastive, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, channel)
        self.down1 = Down(channel, channel*2)
        self.down2 = Down(channel*2, channel*4)
        self.down3 = Down(channel*4, channel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(channel*8, channel*16 // factor)
        self.up1 = Up(channel*16, channel*8 // factor, bilinear)
        self.up2 = Up(channel*8, channel*4 // factor, bilinear)
        self.up3 = Up(channel*4, channel*2 // factor, bilinear)
        self.up4 = Up(channel*2, channel, bilinear)
        self.outc = nn.Conv2d(channel, n_classes, kernel_size=1, bias=True)

    def forward(self, x, train=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        feat = x
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        if train:
            return out, feat
        else:
            return out

    def get_feature(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        return feature



class UNet_class_con(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32, bilinear=True):
        super(UNet_class_con, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, channel)
        self.down1 = Down(channel, channel*2)
        self.down2 = Down(channel*2, channel*4)
        self.down3 = Down(channel*4, channel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(channel*8, channel*16 // factor)
        self.up1 = Up(channel*16, channel*8 // factor, bilinear)
        self.up2 = Up(channel*8, channel*4 // factor, bilinear)
        self.up3 = Up(channel*4, channel*2 // factor, bilinear)
        self.up4 = Up(channel * 2, channel, bilinear)
        
        self.contrastive_Conv = Class_contrastive_Conv(channel, channel, n_classes)
        
        self.outc = nn.Conv2d(channel * n_classes, n_classes, kernel_size=1, bias=True)
        

    def forward(self, x, train=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        class_feature, x = self.contrastive_Conv(x)
        out = self.outc(x)
        if train:
            return out, class_feature
        else:
            return out

    def get_feature(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        return feature



class UNet_class_con_pretrained(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32, bilinear=True):
        super(UNet_class_con, self).__init__()

        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            decoder_channels=[channel * 16, channel * 8, channel * 4, channel * 2, channel],
            in_channels=in_channels,        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_classes,              # model output channels (number of classes in your dataset)
        )
        self.contrastive_Conv = Class_contrastive_Conv(channel, channel, n_classes)
        
        self.outc = nn.Conv2d(channel * n_classes, n_classes, kernel_size=1, bias=True)
        

    def forward(self, x, train=False):

        inputs = torch.rand([1, 1, 128, 128]).cuda()
        feature_map = model.encoder(inputs)
        outputs = model.decoder(*feature_map)
        classes = model.segmentation_head(outputs)
        out = self.outc(x)
        if train:
            return out, class_feature
        else:
            return out

    def get_feature(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        return feature