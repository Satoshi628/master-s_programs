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
from .Generator import Generator
from .Discriminator import Discriminator

class AML(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32, class_weights=None):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.channel = channel
        
        self.model_G = Generator(self.in_channels, self.n_classes, self.channel, class_weights)
        self.model_D = Discriminator(self.in_channels, self.n_classes, self.channel)
    
    def first_run(self, inputs):
        #一回目の実行、勾配計算なし
        with torch.no_grad():
            #G,D同士の結合はない
            out, _ = self.model_G(inputs, targets=None, memory=None)

            segmentation = F.softmax(out, dim=1)

            #D_featureは長さ4の配列
            _, feature_D = self.model_D(inputs, segmentation, weight=None)
        return feature_D

    def learn_G(self, inputs, targets):
        feature_D = self.first_run(inputs)
        
        #weightsは長さ3の配列
        out_G, weights = self.model_G(inputs, targets=targets, memory=feature_D)

        #偽segmentation
        segmentation = F.softmax(out_G, dim=1)

        out_D, _ = self.model_D(inputs, segmentation, weight=weights)

        return segmentation, out_D, weights  # weightsはDiscriminatorに使用
    
    def learn_D(self, inputs, segmentation, weights):
        out_D, _ = self.model_D(inputs, segmentation, weight=weights)
        
        return out_D

    def forward(self, x):
        #一回目の実行、勾配計算なし
        feature_D = self.first_run(x)
        
        out_G, _ = self.model_G(x, targets=None, memory=feature_D)
        
        return out_G
    
    def get_G_Loss(self):
        return self.model_G.get_Loss()

