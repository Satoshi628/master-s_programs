## t-SNEの適応
from sklearn.manifold import TSNE
# 結果の可視化
import matplotlib.pyplot as plt
#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import argparse
import random
from tqdm import tqdm
#from models.Unet import UNet
from models.Unet import UNet
from utils.dataset import Covid19_Loader
import utils.utils as ut
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter
############################

############## test関数 ##############
def test():
    # モデル→推論モード
    model.eval()

    # 初期設定
    correct = 0
    total = 0
    test_transform = transforms.Compose([transforms.ToTensor(), #0~1正規化+Tensor型に変換
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #データの標準化
                                 ])
    # 画像はcovid-19のtest画像を利用
    #img = Image.open('Dataset/covid19/Image/test/img_100.png').convert('RGB')
    img = Image.open('covid19/Image/train/img_2.png').convert('RGB')
    
    # ラベル読み込み # Image.open = 読み込み .convert("L") = GrayScaleで読み込み
    #L = Image.open('Dataset/covid19/Label/test/img_100.png').convert("L")
    L = Image.open('covid19/Label/train/img_2.png').convert("L")

    # 画像の正規化(train画像)
    inputs = test_transform(img)
    to_tensor = transforms.ToTensor()
    L = to_tensor(L) * 255
    
    inputs = inputs.cuda(device).unsqueeze(0)
    print(inputs.shape)

    # 教師ラベルをlongモードに変換
    #入力画像をモデルに入力
    #y, h = model(inputs)
    y, h_0, h_1, h_2, h_3 = model(inputs)

    h = torch.cat((h_0, h_1, h_2, h_3), dim=1)

    # 出力をsoftmax関数に(0~1)
    y = F.softmax(y, dim=1)

    # 最大ベクトル
    _, predicted = y.max(1)

    return predicted, h, L
"""
class UNet(nn.Module):
   # 初期設定
    def __init__(self, in_ch=1, n_class=4):
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
        self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_f = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1, stride=1, padding=0)

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
        self.bn72 = nn.BatchNorm2d(num_features=64)
        self.bnde1 = nn.BatchNorm2d(num_features=256)
        self.bnde2 = nn.BatchNorm2d(num_features=128)
        self.bnde3 = nn.BatchNorm2d(num_features=64)
        

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
        h = F.relu(self.bnde1(self.deconv1(h)))   
        h = torch.cat((h3, h), dim=1)             #64+64=128
        h = F.relu(self.bn51(self.conv51(h)))
        h = F.relu(self.bn52(self.conv52(h)))

        # Block6  Decoder
        h = F.relu(self.bnde2(self.deconv2(h))) 
        h = torch.cat((h2, h), dim=1)             #32+32=64                                   
        h = F.relu(self.bn61(self.conv61(h)))
        h = F.relu(self.bn62(self.conv62(h)))

        # Block7  Decoder
        h = F.relu(self.bnde3(self.deconv3(h)))  
        h = torch.cat((h1, h), dim=1)             #16+16=32                       
        h = F.relu(self.bn71(self.conv71(h)))
        #h = F.relu(self.bn72(self.conv72(h)))  #0-1
        h = self.bn72(self.conv72(h))  #-1-1
        

        # discriminate
        #y = self.conv_f(h)
        y = self.conv_f(F.relu(h))
        #y = F.relu(self.conv_f(h))
  
        return y,h
"""

class UNet(nn.Module):
   # 初期設定
    def __init__(self, in_ch=3, n_class=4):
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
        self.conv_f = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1, stride=1, padding=0)

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
        #print("h", h.shape)   #[4, 256, 256, 256]

        y = self.conv_f(F.relu(h))
  
        return y, h_0, h_1, h_2, h_3


##### GPU設定 #####

device = torch.device('cuda:0') 

# モデル設定
model = UNet(in_ch=3, n_class=4).cuda(device)


# 学習済みモデルのロード                     -----> SquareLossで学習したモデルに変える
model_path = PATH ="result91/model.pth"
model.load_state_dict(torch.load(model_path))

y, h, L = test()
print(L.shape)

y = y.view(y.size(1) * y.size(2))  # (W*Hのデータ数にする)
L = L.view(L.size(1) * L.size(2))
L = L.to(torch.int32)  # float32 -> int32
#print(y.size())
y = y.tolist()  # tensor -> int型
L = L.tolist()

h = h.view(1, h.size(1), h.size(2) * h.size(3))
h = h[0]  # (h.size(1), W*H)
h = h.permute([1, 0])  # 次元を入れ替え  (データ数, 次元)


X_test = h.to('cpu').detach().numpy().astype(np.float32)  # tensor -> numpy

# 入力画像の確認
print(X_test.shape)
print(type(X_test))

# t-SNEの適応
# 2次元のデータに落とし込む
# X_testにt-SNEを適応し、得られた低次元データをX_tsneに格納する

tsne = TSNE(n_components = 2) # n_componentsは低次元データの次元数
X_tsne = tsne.fit_transform(X_test)  # X_test : 入力画像とそのラベルが格納されている


# t-SNEの可視化
colors = ['black', 'blue', 'green', 'red']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1) # 横軸の最小値・最大値を(float, float)で指定
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)

for color_index in range(len(colors)):  # 黒青緑赤の順に描画する
    if color_index != 'black':
        for i in range(len(X_test)):
            if L[i] == color_index:
                    plt.text(
                    X_tsne[i, 0],
                    X_tsne[i, 1],
                    str(L[i]),
                    color = colors[L[i]]
                )


plt.xlabel('t-SNE Feature1')
plt.ylabel('t-SNE Feature2')

"""
for i in range(5):
    plt.savefig(f'img{i:03d}.png')
"""

plt.savefig("img91-1.png")
#plt.savefig("result33.png")

