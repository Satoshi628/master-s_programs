#coding: utf-8
###################################
import torch
import torch.nn as nn
import torch.nn.functional as F
###################################

class SegNet(nn.Module):
   # 初期設定
    def __init__(self, in_ch=1, n_class=4):
        super(SegNet, self).__init__()
        # Convolution Layer
        self.conv11 = nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv41 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv51 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv61 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv71 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv72 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_f = nn.Conv2d(in_channels=16, out_channels=n_class, kernel_size=1, stride=1, padding=0)

        # Downsampling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deconvolution Layer
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        # Batch Normalization Layer
        self.bn11 = nn.BatchNorm2d(num_features=16)
        self.bn12 = nn.BatchNorm2d(num_features=16)
        self.bn13 = nn.BatchNorm2d(num_features=16)
        self.bn21 = nn.BatchNorm2d(num_features=32)
        self.bn22 = nn.BatchNorm2d(num_features=32)
        self.bn23 = nn.BatchNorm2d(num_features=32)
        self.bn31 = nn.BatchNorm2d(num_features=64)
        self.bn32 = nn.BatchNorm2d(num_features=64)
        self.bn33 = nn.BatchNorm2d(num_features=64)
        self.bn41 = nn.BatchNorm2d(num_features=128)
        self.bn42 = nn.BatchNorm2d(num_features=128)
        self.bn43 = nn.BatchNorm2d(num_features=128)
        self.bn51 = nn.BatchNorm2d(num_features=64)
        self.bn52 = nn.BatchNorm2d(num_features=64)
        self.bn61 = nn.BatchNorm2d(num_features=32)
        self.bn62 = nn.BatchNorm2d(num_features=32)
        self.bn71 = nn.BatchNorm2d(num_features=16)
        self.bn72 = nn.BatchNorm2d(num_features=16)
        self.bnde1 = nn.BatchNorm2d(num_features=64)
        self.bnde2 = nn.BatchNorm2d(num_features=32)
        self.bnde3 = nn.BatchNorm2d(num_features=16)



    def __call__(self, x):
        # Block1
        h = F.relu(self.bn11(self.conv11(x)))                                       
        h = F.relu(self.bn12(self.conv12(h)))
        h1 = F.relu(self.bn13(self.conv13(h)))
        h = self.pool1(h1)

        # Block2
        h = F.relu(self.bn21(self.conv21(h)))                                       
        h = F.relu(self.bn22(self.conv22(h)))
        h2 = F.relu(self.bn23(self.conv23(h)))
        h = self.pool2(h2)

        # Block3
        h = F.relu(self.bn31(self.conv31(h)))                                       
        h = F.relu(self.bn32(self.conv32(h)))
        h3 = F.relu(self.bn33(self.conv33(h)))
        h = self.pool3(h3)

        # Block4
        h = F.relu(self.bn41(self.conv41(h)))                                       
        h = F.relu(self.bn42(self.conv42(h)))
        h = F.relu(self.bn43(self.conv43(h)))

        # Block5
        h = F.relu(self.bnde1(self.deconv1(h)))                                       
        h = F.relu(self.bn51(self.conv51(h)))
        h = F.relu(self.bn52(self.conv52(h)))

        # Block6
        h = F.relu(self.bnde2(self.deconv2(h)))                                       
        h = F.relu(self.bn61(self.conv61(h)))
        h = F.relu(self.bn62(self.conv62(h)))

        # Block7
        h = F.relu(self.bnde3(self.deconv3(h)))                                       
        h = F.relu(self.bn71(self.conv71(h)))
        h = F.relu(self.bn72(self.conv72(h)))

        # discriminate
        y = self.conv_f(h)

        return y

