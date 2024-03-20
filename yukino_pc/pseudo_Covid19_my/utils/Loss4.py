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
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
############################

class SquareLoss(torch.nn.Module):
    def __init__(self, device):
        super(SquareLoss, self).__init__()
        self.device = device

    def forward(self, h_0, h_1, h_2, targets, y):   # output 出力(64次元)  targets ラベル
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        b, c, h, w = h_0.size()


        #errorflag = [False, False, False, False]
        errorflag = [False, False, False]

        a0 = h_0.permute(0, 2, 3, 1)[targets == 0] #a0にclass0のピクセルのみを入れる
        #print("a0", a0.shape)
        a1 = h_1.permute(0, 2, 3, 1)[targets == 1] #a1にclass1のピクセルのみ入れる
        a2 = h_2.permute(0, 2, 3, 1)[targets == 2] #a2にclass2のピクセルのみ入れる
        #a3 = h_3.permute(0, 2, 3, 1)[targets == 3] #a3にclass3のピクセルのみ入れる


        if a0.size(0) == 0: #class0のピクセルが0個のとき
            errorflag[0] = True #errorflagのclass0をTrueにする
            a0 = a0.new_zeros([1,c]) #a0を1にする(0だと計算できないので無理やり1にする)
        if a1.size(0) == 0:
            errorflag[1] = True
            a1 = a1.new_zeros([1,c])
        if a2.size(0) == 0:
            errorflag[2] = True
            a2 = a2.new_zeros([1,c])
        """
        if a3.size(0) == 0:
            errorflag[3] = True
            a3 = a3.new_zeros([1,c])
        """
        #print("a1", a1.shape)
        #print("a2", a2.shape)
        #print("a3", a3.shape)
        

        #a0～a3の各クラスからランダムに1点選ぶ
        #0～(個数-1)までの値から1つ選ぶ
        y0 = torch.randint(a0.size(0), [1])  
        y1 = torch.randint(a1.size(0), [1])
        y2 = torch.randint(a2.size(0), [1])
        #y3 = torch.randint(a3.size(0), [1])
        

        y0 = a0[y0, :]  #[1,64]
        y1 = a1[y1, :]
        y2 = a2[y2, :]
        #y3 = a3[y3, :]


        #選んだ点(y0～y3)と特徴マップ(h_0～h_3)の大きさを整形
        #y0～y3の大きさを[c] -> [1, c, 1, 1]にする
        y0 = y0.view(1, y0.size(0), y0.size(1), 1, 1) #[1, 個数, c, 1, 1]
        y1 = y1.view(1, y1.size(0), y1.size(1), 1, 1)
        y2 = y2.view(1, y2.size(0), y2.size(1), 1, 1)
        #y3 = y3.view(1, y3.size(0), y3.size(1), 1, 1)

        h_0 = h_0.view(b, 1, h_0.size(1), h, w) #[1, 個数, c, 256, 256]
        h_1 = h_1.view(b, 1, h_1.size(1), h, w)
        h_2 = h_2.view(b, 1, h_2.size(1), h, w)
        #h_3 = h_3.view(b, 1, h_3.size(1), h, w)

        
        #ベクトルの大きさを1にしておく
        #output = F.normalize(output, dim=1)
        y0 = F.normalize(y0, dim=2)  #channelに対して行う
        y1 = F.normalize(y1, dim=2)  
        y2 = F.normalize(y2, dim=2)  
        #y3 = F.normalize(y3, dim=2)  

        h_0 = F.normalize(h_0, dim=2)  #channelに対して行う
        h_1 = F.normalize(h_1, dim=2)  
        h_2 = F.normalize(h_2, dim=2)  
        #h_3 = F.normalize(h_3, dim=2)

        #選んだ点(a0～a3)と特徴マップ(h_0～h_1)を演算(cos類似度の計算)
        inner0 = (y0 * h_0).sum(dim=2)
        inner1 = (y1 * h_1).sum(dim=2)
        inner2 = (y2 * h_2).sum(dim=2)
        #inner3 = (y3 * h_3).sum(dim=2)
        


        ###選んだ点に応じてラベルを見て同じものには1、それ以外には-1を[256,256]に入れる
        map0 = 2.0 * (0 == targets) - 1.0  #class0の点は1それ以外は-1
        map1 = 2.0 * (1 == targets) - 1.0  #class1の点は1それ以外は-1
        map2 = 2.0 * (2 == targets) - 1.0  #class2の点は1それ以外は-1
        #map3 = 2.0 * (3 == targets) - 1.0  #class3の点は1それ以外は-1

        
        ###平均二乗誤差の計算
        loss0 = pow((inner0 - map0), 2)
        loss1 = pow((inner1 - map1), 2)
        loss2 = pow((inner2 - map2), 2)
        #loss3 = pow((inner3 - map3), 2)
        """
        loss = loss0 + loss1 + loss2 + loss3
        loss = loss.mean()/4
        #print(loss.shape)
        """
        #class0が0個のときはerrorflag[0]がTrueとなり1になるので、
        #1-errorflag[0]=0となり0個のclassのlossは計算されない
        loss0 = loss0.mean()*(1.-errorflag[0]) 
        loss1 = loss1.mean()*(1.-errorflag[1])
        loss2 = loss2.mean()*(1.-errorflag[2])
        #loss3 = loss3.mean()*(1.-errorflag[3])

        #loss4 = (loss0 + loss1 + loss2 + loss3)/4
        loss4 = (loss0 + loss1 + loss2)/4

        #return loss, h, w
        #return loss0, loss1, loss2, loss3, loss4
        return loss0, loss1, loss2, loss4
        
