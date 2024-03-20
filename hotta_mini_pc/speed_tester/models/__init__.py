#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
from .backbone.resnet import ResidualBlock_Small, ResidualBlock_Large, ResNet


class classifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d([1, 1])
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.GAP(x).flatten(-3)
        out = self.linear(out)
        return out

class Resnet_Classification(nn.Module):
    def __init__(self, name="resnet50", in_channels=3, num_classes=1):
        super().__init__()
        if name == "resnet18":
            self.backbone = ResNet(ResidualBlock_Small, [2, 2, 2, 2], in_channels)
        elif name == "resnet34":
            self.backbone = ResNet(ResidualBlock_Small, [3, 4, 6, 3], in_channels)
        elif name == "resnet50":
            self.backbone = ResNet(ResidualBlock_Large, [3, 4, 6, 3], in_channels)
        else:
            KeyError("name must be included in \"resnet18\", \"resnet34\", \"resnet50\"")
        self.classifier = classifier(2048, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        return out