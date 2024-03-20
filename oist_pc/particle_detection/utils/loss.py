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

PAD_ID = -1
NONE_TOKEN_ID = -2
NONE_TOKEN_IDX = 0

class Contrastive_Loss(nn.Module):
    def __init__(self, overlap_range=[25., 30., 35.]):
        super().__init__()
        self.overlap_range = overlap_range
        self.mse_fnc = nn.MSELoss()

    def Contrastive_loss_calc(self, feature, pre_coord, label_id):
        #検出位置の特徴量を取得
        idx = torch.arange(pre_coord.size(0) * pre_coord.size(1) * pre_coord.size(2), device=pre_coord.device).view(pre_coord.shape[:3])
        batch_idx = idx // (pre_coord.size(1) * pre_coord.size(2))
        length_idx = idx % pre_coord.size(1)

        #[batch,length,num,channel]
        selected_feature = feature[batch_idx, :, length_idx, pre_coord[:, :, :, 1], pre_coord[:, :, :, 0]]

        selected_feature = F.normalize(selected_feature, dim=-1)


        PAD_flag_all = pre_coord[:, :, :, 0] < 0
        
        #l=0のとき[:-0]はsize0になるためfor分外に書く必要がある

        #類似度計算
        similar = (selected_feature[:, :, :, None] * selected_feature[:, :, None]).sum(dim=-1)

        # [batch,length,num,num]
        PAD_flag = PAD_flag_all[:, :, :, None] | PAD_flag_all[:, :, None]

        #距離計算 [batch,length,num,num]
        distance = (pre_coord[:, :, :, None] - pre_coord[:, :, None, :]) ** 2
        distance = torch.sqrt(distance.sum(dim=-1))
        distance = distance < self.overlap_range[0]
        distance[PAD_flag] = False

        #ラベル番号一致flag [batch,length,num,num]
        label_match_flag = (label_id[:, :, :, None] == label_id[:, :, None]) * 1.0
        loss = ((similar[distance] - label_match_flag[distance]) ** 2).mean()

        for l in range(1, 4):
            #類似度計算
            similar = (selected_feature[:, :-l, :, None] * selected_feature[:, l:, None]).sum(dim=-1)

            # [batch,length,num,num]
            PAD_flag = PAD_flag_all[:, :-l, :, None] | PAD_flag_all[:, l:, None]

            #距離計算 [batch,length,num,num]
            distance = (pre_coord[:, :-l, :, None] - pre_coord[:, l:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))
            distance = distance < self.overlap_range[l - 1]
            distance[PAD_flag] = False

            #ラベル番号一致flag [batch,length,num,num]
            label_match_flag = (label_id[:, :-l, :, None] == label_id[:, l:, None]) * 1.0
            
            loss = loss + ((similar[distance] - label_match_flag[distance]) ** 2).mean()


        return loss / 4

    def forward(self, predicts, feature, pre_coord, targets, label_id):
        
        mse_loss = self.mse_fnc(predicts.squeeze(1), targets)
        contrastive_loss = self.Contrastive_loss_calc(feature, pre_coord, label_id)
        return mse_loss+contrastive_loss


class weighted_MSELoss(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha

    def forward(self, predicts, targets):
        false_positive = (targets < predicts)
        false_positive = (targets - predicts)[false_positive] ** 2
        detection_error = (targets > predicts)
        detection_error = (targets - predicts)[detection_error] ** 2

        if self.alpha:
            alpha = self.alpha
        else:
            alpha = torch.clip(false_positive.sum(), min=1e-6) / torch.clip(detection_error.sum() + false_positive.sum(), min=1e-6)
        
        mse_loss = alpha * false_positive.mean() + (1 - alpha) * detection_error.mean()

        return mse_loss