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
