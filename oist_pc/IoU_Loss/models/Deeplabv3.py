#coding: utf-8
#----- 標準ライブラリ -----#
import sys

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#----- 自作モジュール -----#
#None


class Deeplabv3(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(num_classes=n_classes)

    def __call__(self, inputs):
        outputs = self.model(inputs)

        return outputs['out']