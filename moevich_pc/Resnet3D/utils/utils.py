#coding: utf-8
#----- 標準ライブラリ -----#
import numbers
import random

#----- 専用ライブラリ -----#
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F

#----- 自作モジュール -----#
#None

# to Tensor
class CT_ndarrayToTensor(object):
    def __init__(self, WW, WL):
        self.WW = WW
        self.WL = WL
    def __call__(self, img):
        img = torch.from_numpy(img)
        minimum = self.WL - self.WW // 2
        maximum = self.WL + self.WW // 2
        img = (img.clamp(minimum, maximum) - minimum) / self.WW
        return img



# to Tensor (Multi)
class Multi_CT_ndarrayToTensor(object):
    def __init__(self, WW=[256,1024,100], WL=[0,0,50]):
        assert len(WW) == len(WL), "WWとWLで、チャンネルの数が違います"
        self.WW = WW
        self.WL = WL
    
    def __call__(self, img):
        img = torch.from_numpy(img)
        img = torch.cat([norm for norm in self._norm(img)], dim=1)

        return img

    def _norm(self, img_tensor):
        for WW, WL in zip(self.WW, self.WL):
            minimum = WL - WW // 2
            maximum = WL + WW // 2
            norm_img = (img_tensor.clamp(minimum, maximum) - minimum) / WW
            yield norm_img

