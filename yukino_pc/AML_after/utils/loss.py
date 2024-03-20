#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
#None

#loss定義
#targetsがone hot vectorのCross Entropy Loss
class CrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is None:
            self.weights = [1.]
        else:
            self.weights = weight

    def forward(self, inputs, targets):
        out = F.softmax(inputs, dim=1)
        out = torch.clamp(out, min=1e-7)
        logit = targets * torch.log(out)
        weights = inputs.new_tensor(self.weights)[None, :, None, None]
        logit = weights * logit
        return -logit.mean()

