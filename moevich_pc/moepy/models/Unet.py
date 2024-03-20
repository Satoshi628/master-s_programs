#coding: utf-8
#----- 標準ライブラリ -----#
import sys
#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
#----- 自作モジュール -----#
#None

class UNet(nn.Module):
   # 初期設定
    def __init__(self, in_ch=3, n_class=3):
        super(UNet, self).__init__()
        # Convolution Layer
        self.conv11 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv51 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv61 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv71 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_f = nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1, stride=1, padding=0)
        # covidだったらin_channels=256 Cell_Datasetだったら192

        #1×1conv(新)
        self.conv73 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv74 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv75 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv76 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Downsampling Layer  Dawnsampling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deconvolution Layer  Upsampling
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        
        # Batch Normalization Layer
        self.bn11 = nn.BatchNorm2d(num_features=64)
        self.bn12 = nn.BatchNorm2d(num_features=64)
        self.bn13 = nn.BatchNorm2d(num_features=64)
        self.bn21 = nn.BatchNorm2d(num_features=128)
        self.bn22 = nn.BatchNorm2d(num_features=128)
        self.bn23 = nn.BatchNorm2d(num_features=128)
        self.bn31 = nn.BatchNorm2d(num_features=256)
        self.bn32 = nn.BatchNorm2d(num_features=256)
        self.bn33 = nn.BatchNorm2d(num_features=256)
        self.bn41 = nn.BatchNorm2d(num_features=512)
        self.bn42 = nn.BatchNorm2d(num_features=512)
        self.bn43 = nn.BatchNorm2d(num_features=512)
        self.bn51 = nn.BatchNorm2d(num_features=256)
        self.bn52 = nn.BatchNorm2d(num_features=256)
        self.bn61 = nn.BatchNorm2d(num_features=128)
        self.bn62 = nn.BatchNorm2d(num_features=128)
        self.bn71 = nn.BatchNorm2d(num_features=64)
        self.bnde1 = nn.BatchNorm2d(num_features=256)
        self.bnde2 = nn.BatchNorm2d(num_features=128)
        self.bnde3 = nn.BatchNorm2d(num_features=64)

        self.bn72 = nn.BatchNorm2d(num_features=64)
        self.bn73 = nn.BatchNorm2d(num_features=64)
        self.bn74 = nn.BatchNorm2d(num_features=64)
        self.bn75 = nn.BatchNorm2d(num_features=64)
        self.bn76 = nn.BatchNorm2d(num_features=64)


    def __call__(self, x):
        # Block1  Encoder
        h = F.relu(self.bn11(self.conv11(x)))                                      
        h = F.relu(self.bn12(self.conv12(h)))
        h1 = F.relu(self.bn13(self.conv13(h)))
        h = self.pool1(h1)  #1/2

        # Block2  Encoder
        h = F.relu(self.bn21(self.conv21(h)))                                       
        h = F.relu(self.bn22(self.conv22(h)))
        h2 = F.relu(self.bn23(self.conv23(h)))
        h = self.pool2(h2)  #1/4

        # Block3  Encoder
        h = F.relu(self.bn31(self.conv31(h)))                                       
        h = F.relu(self.bn32(self.conv32(h)))
        h3 = F.relu(self.bn33(self.conv33(h)))
        h = self.pool3(h3)  #1/8

        # Block4  Middle
        h = F.relu(self.bn41(self.conv41(h)))                                       
        h = F.relu(self.bn42(self.conv42(h)))
        h = F.relu(self.bn43(self.conv43(h)))

        # Block5  Decoder
        h = F.relu(self.bnde1(self.deconv1(h)))   #512->256
        h = torch.cat((h3, h), dim=1)             #256+256=512
        h = F.relu(self.bn51(self.conv51(h)))     #512->256
        h = F.relu(self.bn52(self.conv52(h)))     #256->256   0-1

        # Block6  Decoder
        h = F.relu(self.bnde2(self.deconv2(h)))   #256->128
        h = torch.cat((h2, h), dim=1)             #128+128=256                                   
        h = F.relu(self.bn61(self.conv61(h)))     #256->128
        h = F.relu(self.bn62(self.conv62(h)))     #128->128   0-1

        # Block7  Decoder
        h = F.relu(self.bnde3(self.deconv3(h)))   #128->64
        h = torch.cat((h1, h), dim=1)             #64+64=128                       
        h = F.relu(self.bn71(self.conv71(h)))     #128->64
        #h = F.relu(self.bn72(self.conv72(h)))     #64->64   0-1

        h_0 = self.bn73(self.conv73(h))  #class0 -1-1
        h_1 = self.bn74(self.conv74(h))  #class1 -1-1
        h_2 = self.bn75(self.conv75(h))  #class2 -1-1
        #h_3 = self.bn76(self.conv76(h))  #class3 -1-1

        #print("h_0", h_0.shape)   #[4, 64, 256, 256]
        #print("h_1", h_1.shape)   #[4, 64, 256, 256]
        #print("h_2", h_2.shape)   #[4, 64, 256, 256]
        #print("h_3", h_3.shape)   #[4, 64, 256, 256]
        

        # discriminate
        #y = self.conv_f(F.relu(h))
        #h = torch.cat((h_0, h_1, h_2, h_3), dim=1)
        h = torch.cat((h_0, h_1, h_2), dim=1)
       
        #print("h", h.shape)   #[4, 256, 256, 256]

        y = self.conv_f(F.relu(h))
        #print("y", y.shape)
  
        #return y, h_0, h_1, h_2, h_3
        return y, h_0, h_1, h_2



class UNet_4(nn.Module):
    # 初期設定
    def __init__(self, in_ch=3, n_class=4):
        super(UNet_4, self).__init__()
        # Convolution Layer
        self.conv11 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv51 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv61 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv71 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_f = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1, stride=1, padding=0)
        # covidだったらin_channels=256 Cell_Datasetだったら192

        #1×1conv(新)
        self.conv73 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv74 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv75 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv76 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Downsampling Layer  Dawnsampling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deconvolution Layer  Upsampling
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        
        # Batch Normalization Layer
        self.bn11 = nn.BatchNorm2d(num_features=64)
        self.bn12 = nn.BatchNorm2d(num_features=64)
        self.bn13 = nn.BatchNorm2d(num_features=64)
        self.bn21 = nn.BatchNorm2d(num_features=128)
        self.bn22 = nn.BatchNorm2d(num_features=128)
        self.bn23 = nn.BatchNorm2d(num_features=128)
        self.bn31 = nn.BatchNorm2d(num_features=256)
        self.bn32 = nn.BatchNorm2d(num_features=256)
        self.bn33 = nn.BatchNorm2d(num_features=256)
        self.bn41 = nn.BatchNorm2d(num_features=512)
        self.bn42 = nn.BatchNorm2d(num_features=512)
        self.bn43 = nn.BatchNorm2d(num_features=512)
        self.bn51 = nn.BatchNorm2d(num_features=256)
        self.bn52 = nn.BatchNorm2d(num_features=256)
        self.bn61 = nn.BatchNorm2d(num_features=128)
        self.bn62 = nn.BatchNorm2d(num_features=128)
        self.bn71 = nn.BatchNorm2d(num_features=64)
        self.bnde1 = nn.BatchNorm2d(num_features=256)
        self.bnde2 = nn.BatchNorm2d(num_features=128)
        self.bnde3 = nn.BatchNorm2d(num_features=64)

        self.bn72 = nn.BatchNorm2d(num_features=64)
        self.bn73 = nn.BatchNorm2d(num_features=64)
        self.bn74 = nn.BatchNorm2d(num_features=64)
        self.bn75 = nn.BatchNorm2d(num_features=64)
        self.bn76 = nn.BatchNorm2d(num_features=64)


    def __call__(self, x):
        # Block1  Encoder
        h = F.relu(self.bn11(self.conv11(x)))                                      
        h = F.relu(self.bn12(self.conv12(h)))
        h1 = F.relu(self.bn13(self.conv13(h)))
        h = self.pool1(h1)  #1/2

        # Block2  Encoder
        h = F.relu(self.bn21(self.conv21(h)))                                       
        h = F.relu(self.bn22(self.conv22(h)))
        h2 = F.relu(self.bn23(self.conv23(h)))
        h = self.pool2(h2)  #1/4

        # Block3  Encoder
        h = F.relu(self.bn31(self.conv31(h)))                                       
        h = F.relu(self.bn32(self.conv32(h)))
        h3 = F.relu(self.bn33(self.conv33(h)))
        h = self.pool3(h3)  #1/8

        # Block4  Middle
        h = F.relu(self.bn41(self.conv41(h)))                                       
        h = F.relu(self.bn42(self.conv42(h)))
        h = F.relu(self.bn43(self.conv43(h)))

        # Block5  Decoder
        h = F.relu(self.bnde1(self.deconv1(h)))   #512->256
        h = torch.cat((h3, h), dim=1)             #256+256=512
        h = F.relu(self.bn51(self.conv51(h)))     #512->256
        h = F.relu(self.bn52(self.conv52(h)))     #256->256   0-1

        # Block6  Decoder
        h = F.relu(self.bnde2(self.deconv2(h)))   #256->128
        h = torch.cat((h2, h), dim=1)             #128+128=256                                   
        h = F.relu(self.bn61(self.conv61(h)))     #256->128
        h = F.relu(self.bn62(self.conv62(h)))     #128->128   0-1

        # Block7  Decoder
        h = F.relu(self.bnde3(self.deconv3(h)))   #128->64
        h = torch.cat((h1, h), dim=1)             #64+64=128                       
        h = F.relu(self.bn71(self.conv71(h)))     #128->64
        #h = F.relu(self.bn72(self.conv72(h)))     #64->64   0-1

        h_0 = self.bn73(self.conv73(h))  #class0 -1-1
        h_1 = self.bn74(self.conv74(h))  #class1 -1-1
        h_2 = self.bn75(self.conv75(h))  #class2 -1-1
        h_3 = self.bn76(self.conv76(h))  #class3 -1-1

        #print("h_0", h_0.shape)   #[4, 64, 256, 256]
        #print("h_1", h_1.shape)   #[4, 64, 256, 256]
        #print("h_2", h_2.shape)   #[4, 64, 256, 256]
        #print("h_3", h_3.shape)   #[4, 64, 256, 256]
        

        # discriminate
        #y = self.conv_f(F.relu(h))
        h = torch.cat((h_0, h_1, h_2, h_3), dim=1)
        #h = torch.cat((h_0, h_1, h_2), dim=1)

        #print("h", h.shape)   #[4, 256, 256, 256]

        y = self.conv_f(F.relu(h))
        #print("y", y.shape)

        return y, h_0, h_1, h_2, h_3
        #return y, h_0, h_1, h_2


