#coding: utf-8
###################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from Self_attention_code import SelfAttention
import _common_function as cf
###################################

channel = 16


class Squeeze(nn.Module):
    def __init__(self, in_ch, out_ch, r):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, in_ch//r, 1, 1, 0)
        self.bn = nn.BatchNorm2d(in_ch//r)
        self.conv2 = nn.Conv2d(in_ch//r, out_ch, 1, 1, 0)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv1(x)))
        h = self.conv2(h)

        return h


class Down_CBR(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Down_CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=2, stride=2, padding=0)
        self.bnc = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x, pool=False):
        if pool:
            h = F.relu(self.bnc(self.conv(x)))
            h = F.dropout(h, p=0.25)
            h = F.relu(self.bn1(self.conv1(h)))
            h = F.dropout(h, p=0.25)
        else:
            h = F.relu(self.bn1(self.conv1(x)))
            h = F.relu(self.bn2(self.conv2(h)))

        return h


class Up_DBR(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Up_DBR, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        self.bnd = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x, skip=None):

        h = F.relu(self.bnd(self.deconv(x)))

        if skip is None:
            h = F.relu(self.bn1(self.conv1(h)))
        else:
            h = F.relu(self.bn1(self.conv1(torch.cat([h, skip], dim=1))))
            h = F.dropout(h, p=0.25)
            h = F.relu(self.bn2(self.conv2(h)))
            h = F.dropout(h, p=0.25)
        return h


class Generator(nn.Module):
    def __init__(self, n_channels, n_classes, device, bilinear=True):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Down_CBR(n_channels, channel, channel)
        self.down1 = Down_CBR(channel, channel*2, channel*2)
        self.down2 = Down_CBR(channel*2, channel*4, channel*4)
        self.down3 = Down_CBR(channel*4, channel*8, channel*8)
        self.down4 = Down_CBR(channel * 8, channel * 16, channel * 16)

        self.up1 = Up_DBR(channel*16, channel*16, channel*8)
        self.up1_0 = Squeeze(channel * 8, n_classes, r=2)

        self.up2 = Up_DBR(channel*8, channel*8, channel*4)
        self.up2_0 = Squeeze(channel * 4, n_classes, r=2)

        self.up3 = Up_DBR(channel*4, channel*4, channel*2)
        self.up3_0 = Squeeze(channel * 2, n_classes, r=2)

        self.up4 = Up_DBR(channel*2, channel*2, channel)
        self.outc = nn.Conv2d(channel, n_classes, 1, 1, 0)

        #self.attention_d1 = SelfAttention(128, 64, 128)
        #self.attention_d2 = SelfAttention(256, 64, 256)
        #self.attention_d3 = SelfAttention(512, 64, 512)
        #self.attention_d4 = SelfAttention(1024, 64, 1024)

        self.attention_d1 = SelfAttention(channel*2)
        self.attention_d2 = SelfAttention(channel*4)
        self.attention_d3 = SelfAttention(channel * 8)
        self.attention_d4 = SelfAttention(channel * 16)

        self.ohv = cf.one_hot_vector(device, classification=5)

    def forward(self, x, memory=None, Train=False, **kwargs):

        _, _, height, width = x.shape
        upsample = nn.Upsample(
            size=height, scale_factor=None, mode='bilinear',  align_corners=True)

        if memory is None:
            # print(x)
            # input()
            x1 = self.inc(x, False)
            x2 = self.down1(x1, True)
            x3 = self.down2(x2, True)
            x4 = self.down3(x3, True)
            x5 = self.down4(x4, True)

        else:
            x1 = self.inc(x, False)

            x2 = self.down1(x1, True)
            #print("x2", x2.shape)
            #print("memory[0]", memory[0].shape)  # [16, 128, 128, 128]
            #print('memory[0]', torch.max(memory[0]))  # [16, 128, 128, 128]
            # tensor(0.8028, device='cuda:0')
            # input()
            x2 = self.attention_d1(x2, memory[0])
            

            x3 = self.down2(x2, True)
            #print("x3", x3.shape)
            #print("memory[1]", memory[1].shape)
            #print('memory[1]', torch.max(memory[1]))
            # input()
            x3 = self.attention_d2(x3, memory[1])
            
            x4 = self.down3(x3, True)
            #print("x4", x4.shape)
            #print("memory[2]", memory[2].shape)
            #print('memory[2]', torch.max(memory[2]))
            # input()
            x4 = self.attention_d3(x4, memory[2])
            
            

            x5 = self.down4(x4, True)
            #print("x5", x5.shape)
            #print(" memory[3]", memory[3].shape)
            #print(' memory[3]', torch.max(memory[3]))
            #input()
            x5 = self.attention_d4(x5, memory[3])

        if Train:  # Train is True
            t1 = self.up1(x5, x4)
            # #print('t1', t1.shape)  # t1 torch.Size([4, 64, 32, 32])
            # #print(t1.dtype)  # torch.float32
            # input()
            t0_1 = self.up1_0(t1)
            t0_1 = upsample(t0_1)
            # print('t0_1', t0_1.shape) #t0_1 torch.Size([4, 4, 256, 256])
            # print(t0_1.dtype)  # torch.float32
            # input()
            t1, heatmap1 = self.GT_PDA(t0_1, t1, kwargs['gt'])

            t2 = self.up2(t1, x3)
            t0_2 = self.up2_0(t2)
            t0_2 = upsample(t0_2)
            t2, heatmap2 = self.GT_PDA(t0_2, t2, kwargs['gt'])

            t3 = self.up3(t2, x2)
            t0_3 = self.up3_0(t3)
            t0_3 = upsample(t0_3)
            t3, heatmap3 = self.GT_PDA(t0_3, t3, kwargs['gt'])

            t4 = self.up4(t3, x1)

            out = F.softmax(self.outc(t4), dim=1)

            out1 = F.softmax(t0_1, dim=1)
            out2 = F.softmax(t0_2, dim=1)
            out3 = F.softmax(t0_3, dim=1)

            return out, out3, out2, out1, heatmap3, heatmap2, heatmap1

        else:
            t1 = self.up1(x5, x4)
            t2 = self.up2(t1, x3)
            t3 = self.up3(t2, x2)
            t4 = self.up4(t3, x1)
            out = F.softmax(self.outc(t4), dim=1)

            return out

    def GT_PDA(self, x, output, gt_onehot):
        _, _, width, height = output.shape

        # print(gt_onehot.shape)  # torch.Size([16, 256, 256])
        # input()

        heatmap = F.softmax(x, dim=1)
        # print(heatmap.shape)
        # print(gt_onehot.shape)
        gt_onehot = self.ohv.change(gt_onehot)

        # print(gt_onehot.shape)  # torch.Size([16, 5, 256, 256])
        # input()

        heatmap = heatmap * gt_onehot
        # print(heatmap.shape)
        # input()

        #heatmap = torch.tensor(heatmap.data, device=torch.device("cpu"))

        # torch.max(a, axis)→axisは軸(axis=0:col, axis=1:row)
        heatmap = torch.max(heatmap, 1, keepdim=True)
        #heatmap = torch.from_numpy(heatmap.astype(np.float32)).clone()
        # print(heatmap[0].dtype)
        heatmap = heatmap[0].int()
        heatmap = 1 - heatmap
        # print('heatmap', heatmap[0].dtype)  #torch.int32
        # input()

        heatmap = heatmap.float()
        # print(heatmap[0].dtype)  # torch.int64
        # input()

        heatmap = F.interpolate(heatmap, size=height, mode='bilinear').int()
        # print(' heatmap', heatmap.shape) #torch.Size([4, 1, 32, 32])
        # input()
        heatmap, _ = torch.broadcast_tensors(
            heatmap, output)
        # print(' heatmap', heatmap.shape)　heatmap torch.Size([4, 32, 64, 64])
        # print(' output', output.shape)　　output torch.Size([4, 32, 64, 64])
        # input()
        h = heatmap * output
        h = h + output
        return h, heatmap


class DCR(nn.Module):
    def __init__(self, ch0, ch1, sample='down', activation=F.relu, dropout=False):
        super(DCR, self).__init__()

        self.activation = activation
        self.dropout = dropout

        #self.weight = torch.normal(torch.tensor(0.0), torch.tensor(0.02))

        if sample == 'down':

            self.conv = nn.Conv2d(ch0, ch1, 4, 2, 1)

            self.c = nn.utils.spectral_norm(self.conv)

        else:
            self.c = nn.Conv2d(ch0, ch1, 4, 2, 1)

    def __call__(self, x):
        h = self.c(x)

        if self.dropout:
            h = F.dropout(h,p=0.25)
        if not self.activation is None:
            h = self.activation(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Discriminator, self).__init__()

        #self.inc = Conv_with_LeakyReLU(in_ch, channel)
        #self.down1 = Down_with_LeakyReLU(channel, channel*2)
        #self.down2 = Down_with_LeakyReLU(channel*2, channel*4)
        #factor = 2 if bilinear else 1
        #self.down3 = Down_with_LeakyReLU(channel * 4, channel * 8 // factor)
        #
        #self.up1 = Up_with_LeakyReLU(channel*8, channel*4 // factor, bilinear)
        #self.up2 = Up_with_LeakyReLU(channel*4, channel*2 // factor, bilinear)
        #self.up3 = Up_with_LeakyReLU(channel * 2, channel, bilinear)
        #
        #self.outc = nn.Conv2d(channel, out_ch, kernel_size=1)
        #d0_0:image_channel = 1
        #d0_1:class_channel = 5
        self.d0_0 = nn.Conv2d(1, channel//2, 3, 1, 1)
        self.d0_1 = nn.Conv2d(5, channel//2, 3, 1, 1)

        #self.inc = Down_CBR(in_ch, channel)

        self.down1 = DCR(channel, channel*2, sample='down',
                         activation=F.leaky_relu, dropout=True)
        self.down2 = DCR(channel*2, channel*4, sample='down',
                         activation=F.leaky_relu, dropout=True)
        self.down3 = DCR(channel * 4, channel * 8, sample='down',
                         activation=F.leaky_relu, dropout=True)
        self.down4 = DCR(channel * 8, channel * 16, sample='down',
                         activation=F.leaky_relu, dropout=True)
        self.down5 = nn.Conv2d(channel * 16, 1, 3, 1, 1)

    def forward(self, y1, y2, attention2=None, attention3=None, attention4=None):

        if attention2 is None:

            x = torch.cat([F.leaky_relu(self.d0_0(y1)),
                           F.leaky_relu(self.d0_1(y2))], dim=1)
            # print(x.shape)
            # input()
            x1 = self.down1(x)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x4 = self.down4(x3)
            out = self.down5(x4)

            return out, x1, x2, x3, x4  # [minibatch, 1, H, W]
        else:
            #print('y1', y1.size())
            #print('y2', y2.size())
            # input()
            x = torch.cat([F.leaky_relu(self.d0_0(y1)),
                           F.leaky_relu(self.d0_1(y2))], dim=1)

            x1 = self.down1(x)
            # print(attention2.shape)
            # print(x2.shape)

            #attention2 = attention2.repeat(1, 2, 1, 1)
            x1_h = attention2 * x1
            x1_h = x1_h + x1

            x2 = self.down2(x1_h)
            #attention3 = attention3.repeat(1, 2, 1, 1)
            x2_h = attention3 * x2
            x2_h = x2_h + x2

            x3 = self.down3(x2_h)
            #attention4 = attention4.repeat(1, 2, 1, 1)
            x3_h = attention4 * x3
            x3_h = x3_h + x3

            x4 = self.down4(x3_h)

            out = self.down5(x4)

            return out  # [minibatch, 1, H, W]


"""
model = Generator(n_channels=3, n_classes=4, device="cpu")

data1 = torch.rand([8, 3, 256, 256])


out_list = model(data1)

print(len(out_list))
for out in out_list:
    print(out.shape)
    print(out.dtype)
"""
"""model = Discriminator(in_ch=8, out_ch=1)

data1 = torch.rand([8, 3, 256, 256])
data2 = torch.rand([8, 4, 256, 256])


out, x1, x2, x3, x4 = model(data1, data2)
print(out.shape)
print(out.dtype)

print(x1.shape)
print(x1.dtype)

print(x2.shape)
print(x2.dtype)

print(x3.shape)
print(x3.dtype)

print(x4.shape)
print(x4.dtype)"""
