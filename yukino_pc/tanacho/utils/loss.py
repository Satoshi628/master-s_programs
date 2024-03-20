#coding: utf-8
#----- 標準ライブラリ -----#
# None
#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
# None

class Contrastive_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # targets=>[batch,256,256]
        outputs = F.normalize(outputs, dim=1)
        match = outputs[:, None] * outputs[None]
        match = match.sum(dim=-1)
        targets = (targets[:, None] == targets[None]) * 1.
        loss = (match - targets) ** 2
        loss = loss.mean()
        return loss


class Distance_Loss(nn.Module):
    def __init__(self, margin=10):
        super().__init__()
        self.margin = margin

    def forward(self, outputs, targets):
        Distance = (outputs[:, None] - outputs[None]) ** 2
        Distance = torch.sqrt(Distance.sum(dim=-1))

        match = (targets[:, None] == targets[None]) * 1.0

        Loss = match * Distance + (1 - match) * F.relu(Distance - self.margin)
        
        mask = torch.eye(Loss.shape[0], device=Loss.device) == 0
        Loss = Loss[mask]
        print(Loss.mean())
        return Loss.mean()

