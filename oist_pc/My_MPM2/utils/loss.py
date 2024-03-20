#coding: utf-8
#----- 標準ライブラリ -----#
#None
#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as ttf
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#----- 自作ライブラリ -----#
#None

class RMSE_Q_Loss(nn.Module):
    def __init__(self,q=0.8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.q = q

    def forward(self, outputs, targets):
        outputs_norm = outputs.norm(dim=1)
        targets_norm = targets.norm(dim=1)
        dis = targets_norm - outputs_norm
        dis_q = torch.max((self.q - 1) * dis, self.q * dis)
        dis_q_mse_loss = torch.mean((dis_q) ** 2)

        mse_loss = torch.sqrt(self.mse(outputs,targets))
        return mse_loss + dis_q_mse_loss
