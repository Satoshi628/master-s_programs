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

class Classification():
    def __init__(self,classes):
        self.classes = classes
        self.total = 0.
        self.Top1 = 0.
        self.Top5 = 0.

    
    def _reset(self):
        self.total = 0.
        self.Top1 = 0.
        self.Top5 = 0.
    
    def update(self, outputs, targets):
        """calculate accuracy

        Args:
            outputs (torch.Tensor[batch,classes]): model output
            targets (torch.Trnsor[batch]): label Long
        """        
        _, class_idx = outputs.max(dim=-1)
        self.total += class_idx.size(0)
        self.Top1 += (class_idx == targets).sum().item()

        _, class_idx = torch.topk(outputs, 5, dim=-1)
        self.Top5 += torch.any(class_idx == targets[:, None], dim=-1).sum().item()


    def __call__(self):
        Top1_acc = self.Top1 / self.total
        Top5_acc = self.Top5 / self.total
        self._reset()
        return Top1_acc, Top5_acc
