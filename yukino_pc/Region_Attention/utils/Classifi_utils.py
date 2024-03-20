#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import numpy as np
import torch

#----- 自作モジュール -----#
#None


# to Tensor
class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, img):
        # mutable https://pytorch.org/docs/stable/generated/torch.from_numpy.html
        img = torch.from_numpy(img) / 255.
        return img
