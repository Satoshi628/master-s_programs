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


class connected_loss(nn.Module):
    def __init__(self, move_limit=[25., 30., 35.]):
        super().__init__()
        self.move_limit = move_limit

    def delayed_id_connection(self, label_id):
        #id 抽出 & None token idの追加(-2のid)
        label_id = torch.cat([label_id.new_full([*label_id.shape[:2], 1], NONE_TOKEN_ID), label_id], dim=2)
        
        for delay in range(1, 4): #i => 1, 2, 3
            label_map = label_id[:, :-delay, :, None] == label_id[:, delay:, None]
            
            present_map = torch.clone(label_map)
            present_map[:, :, :, 0] = present_map[:, :, :, 0] | ~torch.any(present_map, dim=-1)
            next_map = torch.clone(label_map)
            next_map[:, :, 0, :] = next_map[:, :, 0, :] | ~torch.any(next_map, dim=-2)

            present_map[(label_id[:, :-delay, :, None] == PAD_ID) | (label_id[:, delay:, None] == PAD_ID)] = False
            next_map[(label_id[:, :-delay, :, None] == PAD_ID) | (label_id[:, delay:, None] == PAD_ID)] = False
            yield present_map, next_map  # [batch,length-i,num+1]
    
    def delayed_predict_connection(self, vector, coord):
        #vector.size() => [batch,length,num,dim]
        #ベクトルの大きさを1にする
        #ゼロベクトルは0のまま
        vector = F.normalize(vector, dim=-1)
        for delay in range(1, 4):
            #内積計算(ノルム1なのでCos類似度)
            similar = (vector[:, :-delay, :, None] * vector[:, delay:, None]).sum(dim=-1)
            
            # distance restriction
            distance = (coord[:, :-delay, :, None] - coord[:, delay:, None]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))
            distance = distance < self.move_limit[delay - 1]
            # accepted None Token matching
            distance[:, :, NONE_TOKEN_IDX] = True
            distance[:, :, :, NONE_TOKEN_IDX] = True

            PAD_map = (coord[:, :-delay, :, None, 0] == PAD_ID) | (coord[:, delay:, None, :, 0] == PAD_ID)
            
            similar = similar.masked_fill(~distance, -9e15)
            similar = similar.masked_fill(PAD_map, -9e15)
            
            yield F.softmax(similar / 0.001, dim=-1), F.softmax(similar / 0.001, dim=-2)  #現在に対しての未来をsoftmax,未来に対しての現在をsoftmax

    def forward(self, vector, label_id, coord):
        """tとt+iの関連性マップのCross Entropy Loss関数

        Args:
            vector (tensor[batch,length,num+1,dim]): vector of  model output
            label_id (tesor[batch,length,num]): 正解検出のid
            coord (tesor[batch,length,num,2(x,y)]): 正解検出のid

        Returns:
            tensor[1]: 関連性マップのcross Entropy Loss
        """

        present_loss = 0
        next_loss = 0

        for idx, ((pre_pre_map, pre_nxt_map), (lab_pre_map, lab_nxt_map)) in enumerate(zip(self.delayed_predict_connection(vector, coord), self.delayed_id_connection(label_id))):
            present_loss = present_loss + torch.log(torch.clamp(pre_pre_map[lab_pre_map], min=1e-7)).mean()
            next_loss = next_loss + torch.log(torch.clamp(pre_nxt_map[lab_nxt_map], min=1e-7)).mean()
        loss = present_loss + next_loss
        return - loss / idx


class Domain_Adaptation(nn.Module):
    def __init__(self):
        super().__init__()
        self.GRL = GradientReversal(scale=1.0)
        self.Discriminator = nn.Sequential(nn.Linear(32, 64,bias=False),
                                            nn.BatchNorm1d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(64, 128,bias=False),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, 2)
                                            )
        self.softmax = nn.Softmax(1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, Feature, label):
        F = self.GRL(Feature)
        F = F.mean(dim=(-1, -2, -3))
        logit = self.Discriminator(F)
        predict = self.softmax(logit)

        loss = self.loss(predict, label)
        return loss


class GradientReversalFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_forward, scale):
        ctx.save_for_backward(scale)
        return input_forward

    @staticmethod
    def backward(ctx, grad_backward):
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None

class GradientReversal(nn.Module):
    def __init__(self, scale: float):
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.scale)



class movement_Loss2(nn.Module):
    def __init__(self, move_limit=[25., 30., 35.]):
        super().__init__()
        self.move_limit = move_limit
    
    def forward(self, pre_move, label_move):
        """tとt+iの関連性マップのCross Entropy Loss関数

        Args:
            pre_move (tensor[batch,length,num,3,2(x,y)]): movement of model output
            label_move (tesor[batch,length,num,4,2(x,y)]): movement of label

        Returns:
            tensor[1]: 関連性マップのcross Entropy Loss
        """
        no_PAD_flag = torch.any(label_move[:, :, :, 0] >= 0, dim=-1) * 1.0
        
        print("予測値")
        print(pre_move.max())
        print(pre_move.min())
        print(pre_move.mean())
        print("正解値")
        print(label_move[:, :, :, 1:].max())
        print(label_move[:, :, :, 1:].min())
        print(label_move[:, :, :, 1:].mean())

        loss1 = (label_move[:, :, :, 1] - pre_move[:, :, :, 0]) ** 2
        loss1 = torch.sqrt(loss1.sum(dim=-1))
        loss1 = loss1 * no_PAD_flag

        loss2 = (label_move[:, :, :, 2] - pre_move[:, :, :, 1]) ** 2
        loss2 = torch.sqrt(loss2.sum(dim=-1))
        loss2 = loss2 * no_PAD_flag

        loss3 = (label_move[:, :, :, 3] - pre_move[:, :, :, 2]) ** 2
        loss3 = torch.sqrt(loss3.sum(dim=-1))
        loss3 = loss3 * no_PAD_flag

        loss = (loss1.mean() + loss2.mean() + loss3.mean()) / 3
        
        return loss




class movement_Loss(nn.Module):
    def __init__(self, move_limit=[25., 30., 35.]):
        super().__init__()
        self.move_limit = move_limit
    
    def forward(self, pre_move, label_move):
        """tとt+iの関連性マップのCross Entropy Loss関数

        Args:
            pre_move (tensor[batch,length,num,3,2(x,y)]): movement of model output
            label_move (tesor[batch,length,num,4,2(x,y)]): movement of label

        Returns:
            tensor[1]: 関連性マップのcross Entropy Loss
        """
        no_PAD_flag = torch.any(label_move[:, :, :, 0] >= 0, dim=-1) * 1.0
        
        print("予測値")
        print(pre_move.max())
        print(pre_move.min())
        print(pre_move.mean())
        print("正解値")
        print(label_move[:, :, :, 1:].max())
        print(label_move[:, :, :, 1:].min())
        print(label_move[:, :, :, 1:].mean())

        loss1 = (label_move[:, :-1, :, 1] - pre_move[:, :-1, :, 0]) ** 2
        loss1 = torch.sqrt(loss1.sum(dim=-1))
        loss1 = loss1 * no_PAD_flag[:, :-1]

        loss = loss1.mean()
        
        return loss
