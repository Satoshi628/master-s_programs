#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
#None


class classifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d([1, 1])
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.GAP(x).flatten(-3)
        out = self.linear(out)
        return out
