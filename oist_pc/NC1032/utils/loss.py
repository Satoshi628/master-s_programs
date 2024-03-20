#coding: utf-8
#----- Standard Library -----#
#None

#----- Public Package -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#----- Module -----#
#None


class FPFN_MSE_Loss(nn.Module):
    def __init__(self, FP_weight=1.0, FN_weight=1.0):
        super().__init__()
        self.FP_weight = FP_weight
        self.FN_weight = FN_weight

    def forward(self, outputs, targets):
        FP_area_flag = (outputs - targets) > 0
        
        mse_loss = (outputs - targets) ** 2

        #FPエリアの重み付け
        mse_loss[FP_area_flag] = self.FP_weight * mse_loss[FP_area_flag]
        #FNエリアの重み付け
        mse_loss[~FP_area_flag] = self.FN_weight * mse_loss[~FP_area_flag]
        return mse_loss.mean()
