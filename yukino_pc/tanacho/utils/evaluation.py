#coding: utf-8
#----- 標準ライブラリ -----#
#None
#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#----- 自作モジュール -----#
#None

PAD_ID = -1

def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """index tensorをone hot vectorに変換する関数

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): one hot vectorの次元。index tensorの最大値以上の値でなくてはならない
        dim (int, optional): one hot vectorをどこの次元に組み込むか. Defaults to -1.
        bool_ (bool, optional): Trueにするとbool型になる。Falseの場合はtorch.float型. Defaults to False.

    Raises:
        TypeError: index tensor is not torch.long
        ValueError: index tensor is greater than vector_dim

    Returns:
        torch.tensor: one hot vector
    """    
    if bool_:
        data_type = bool
    else:
        data_type = torch.float
    if tensor.dtype != torch.long:
        raise TypeError("入力テンソルがtorch.long型ではありません")
    if tensor.max() >= vector_dim:
        raise ValueError(f"入力テンソルのindex番号がvector_dimより大きくなっています\ntensor.max():{tensor.max()}")

    
    #one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor,]
    
    #one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    #もし-1ならそのまま出力
    if dim == -1:
        return vector
    #もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1 #omsertは-1が最後から一つ手前
    
    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector


class mAP():
    def __init__(self, device):
        self.AP = 0.
        self.data_num = 0

        #AP計算用重み
        self.weight = torch.arange(10, device=device) + 1
        self.weight = 1 / self.weight
    
    def _reset(self):
        self.AP = 0.
        self.data_num = 0
    
    def update(self, category_pre, color_pre, category_label, color_label):
        category_pre = F.softmax(category_pre, dim=-1)
        color_pre = F.softmax(color_pre, dim=-1)
        
        score = (category_pre[:, :, None] * color_pre[:, None]) ** 0.5
        
        category_label = one_hot_changer(category_label, vector_dim=2)
        color_label = one_hot_changer(color_label, vector_dim=29)
        label = (category_label[:, :, None] * color_label[:, None])
        
        score = score.flatten(1, -1)
        label = label.flatten(1, -1)
        
        score_values, score_index = torch.topk(score, 10)
        AP = torch.gather(label, dim=-1, index=score_index)
        AP = self.weight[None] * AP

        self.AP += AP.sum()
        self.data_num += AP.shape[0]
    
    def __call__(self):  #IoUとmean_IoUが出力される。
        mAP = self.AP / self.data_num

        return mAP