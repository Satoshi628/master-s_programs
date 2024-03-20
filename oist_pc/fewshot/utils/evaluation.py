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


class IoU():
    def __init__(self, classification):
        self.__classification = classification
        self.__Intersection = 0.
        self.__union = 0.
    
    def _reset(self):
        self.__Intersection = 0.
        self.__union = 0.
    
    def update(self, outputs, targets):
        predicted = outputs.max(dim=1)[1]
        #one hot vector化
        pre_one_hot_vector = one_hot_changer(predicted, self.__classification, dim=1, bool_=True)
        label_one_hot_vector = one_hot_changer(targets, self.__classification, dim=1, bool_=True)
        
        #ラベルと一致するとandでそのクラスがTrueになる。
        #unionはorで求めることができる
        #batch,H,Wでそれらを加算する
        #self.__Intersection.size() => [class]
        self.__Intersection += (pre_one_hot_vector & label_one_hot_vector).sum(dim=(0, 2, 3))
        self.__union += (pre_one_hot_vector | label_one_hot_vector).sum(dim=(0, 2, 3))

    def __call__(self):  #IoUとmean_IoUが出力される。
        IoU = (self.__Intersection / (self.__union + 1e-24))
        mean_IoU = IoU.mean()

        self._reset()

        eval_dict = {f"class{i} IoU":IoU[i] for i in range(self.__classification)}
        eval_dict["mIoU"] = mean_IoU

        return eval_dict

    def keys(self):
        for i in range(self.__classification):
            key = f"class{i} IoU"
            yield key
        yield "mIoU"

class Dice():
    def __init__(self, classification):
        self.__classification = classification
        self.__Intersection = 0.
        self.__union = 0.
    
    def _reset(self):
        self.__Intersection = 0.
        self.__union = 0.
    
    def update(self, outputs, targets):
        predicted = outputs.max(dim=1)[1]
        #one hot vector化
        pre_one_hot_vector = one_hot_changer(predicted, self.__classification, dim=1, bool_=True)
        label_one_hot_vector = one_hot_changer(targets, self.__classification, dim=1, bool_=True)
        
        #ラベルと一致するとandでそのクラスがTrueになる。
        #unionはorで求めることができる
        #batch,H,Wでそれらを加算する
        #self.__Intersection.size() => [class]
        self.__Intersection += (pre_one_hot_vector & label_one_hot_vector).sum(dim=(0, 2, 3))
        self.__union += (pre_one_hot_vector | label_one_hot_vector).sum(dim=(0, 2, 3))

    def __call__(self):  #IoUとmean_IoUが出力される。
        Dice = (2 * self.__Intersection / (self.__Intersection + self.__union + 1e-24))
        mean_Dice = Dice.mean()

        self._reset()

        eval_dict = {f"class{i} Dice":Dice[i] for i in range(self.__classification)}
        eval_dict["mDice"] = mean_Dice

        return eval_dict

    def keys(self):
        for i in range(self.__classification):
            key = f"class{i} Dice"
            yield key
        yield "mDice"

# Hausdorff distance
# 精度がある程度出なければ動作が重く,メモリも増えるため
# testのみの使用推奨
class HD():
    def __init__(self, classification, device="cpu"):
        self.__classification = classification
        self.device = device # GPU
        self.__HD = [0 for _ in range(classification)]
        self.sample_number = [0 for _ in range(classification)]
        
    def _reset(self):
        self.__HD = [0 for _ in range(self.__classification)]
        self.sample_number = [0 for _ in range(self.__classification)]

    def update(self, outputs, targets):
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        predicted = outputs.max(dim=1)[1]

        #one hot vector化
        pre_one_hot_vector = one_hot_changer(predicted, self.__classification, dim=1, bool_=True)
        label_one_hot_vector = one_hot_changer(targets, self.__classification, dim=1, bool_=True)
        
        #重なった箇所はdistance=0であるため計算に考慮しない
        zero_distance_area = pre_one_hot_vector & label_one_hot_vector
        pre_one_hot_vector = pre_one_hot_vector & ~zero_distance_area
        label_one_hot_vector = label_one_hot_vector & ~zero_distance_area

        #重なっていない部分の位置を抽出 [num,4(batch,class,y,x)]
        pre_position = torch.nonzero(pre_one_hot_vector)
        label_position = torch.nonzero(label_one_hot_vector)

        # PAD処理が面倒なためfor文で回す
        for batch_idx in range(outputs.shape[0]):
            #[num,3(class,x,y)]
            pre_cxy = pre_position[pre_position[:, 0] == batch_idx][:, [1, 3, 2]]
            label_cxy = label_position[label_position[:, 0] == batch_idx][:, [1, 3, 2]]
            
            for class_idx in range(self.__classification):
                pre_xy = pre_cxy[pre_cxy[:, 0] == class_idx][:, [1, 2]]
                label_xy = label_cxy[label_cxy[:, 0] == class_idx][:, [1, 2]]
                
                # 検出できなかったり、ラベルがないときは何もしない
                if pre_xy.shape[0] == 0 or label_xy.shape[0] == 0:
                    continue
                # サンプル数をカウント
                self.sample_number[class_idx] += 1

                pre_label_dist = self.norm(pre_xy, label_xy)
                label_pre_dist = self.norm(label_xy, pre_xy)

                pre_label_dist = pre_label_dist.min(dim=-1)[0]
                pre_label_dist = pre_label_dist.max(dim=-1)[0]

                label_pre_dist = label_pre_dist.min(dim=-1)[0]
                label_pre_dist = label_pre_dist.max(dim=-1)[0]

                HD = torch.max(pre_label_dist, label_pre_dist)
                self.__HD[class_idx] += HD.sum(dim=0).item()

    def norm(self, v1, v2):
        v1 = v1.float()
        v2 = v2.float()

        distance = (v1.unsqueeze(-2) - v2.unsqueeze(-3)) ** 2
        distance = torch.sqrt(distance.sum(dim=-1))
        return distance

    def __call__(self):  #class HDとmean HD が出力される
        class_HD = torch.tensor(self.__HD)
        sample_num = class_HD.new_tensor(self.sample_number)
        sample_num = torch.clamp(sample_num, min=1e-7)
        HD = class_HD / sample_num
        m_HD = HD.mean()

        self._reset()

        eval_dict = {f"class{i} HD":HD[i] for i in range(self.__classification)}
        eval_dict["mHD"] = m_HD

        return eval_dict

    def keys(self):
        for i in range(self.__classification):
            key = f"class{i} HD"
            yield key
        yield "mHD"


"""PAD処理がめんどくさすぎて没
# Hausdorff distance
# 精度がある程度出なければ動作が重く,メモリも増えるため
# testのみの使用推奨
class HD():
    def __init__(self, classification, device="cpu"):
        self.__classification = classification
        self.device = device # GPU
        self.__HD = 0.
        self.sample_number = 0
        
    def _reset(self):
        self.__HD = 0.
        self.sample_number = 0

    
    def update(self, outputs, targets):
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        B, C = outputs.shape[:2]
        predicted = outputs.max(dim=1)[1]

        #one hot vector化
        pre_one_hot_vector = one_hot_changer(predicted, self.__classification, dim=1, bool_=True)
        label_one_hot_vector = one_hot_changer(targets, self.__classification, dim=1, bool_=True)
        
        #重なった箇所はdistance=0であるため計算に考慮しない
        zero_distance_area = pre_one_hot_vector & label_one_hot_vector
        pre_one_hot_vector = pre_one_hot_vector & ~zero_distance_area
        label_one_hot_vector = label_one_hot_vector & ~zero_distance_area

        #重なっていない部分の位置を抽出 [num,3[batch*class,y,x]]
        pre_position = torch.nonzero(pre_one_hot_vector.flatten(0, 1))
        label_position = torch.nonzero(label_one_hot_vector.flatten(0, 1))

        #batch,classごとに分ける[batch,class,num,2[x,y]]
        pre_position = [pre_position[pre_position[:, 0] == idx][:, [2, 1]] for idx in range(B * C)]
        label_position = [label_position[label_position[:, 0] == idx][:, [2, 1]] for idx in range(B * C)]

        pre_position = pad_sequence(pre_position, batch_first=True, padding_value=PAD_ID)
        label_position = pad_sequence(label_position, batch_first=True, padding_value=PAD_ID)

        pre_position = pre_position.view(B, C, -1, 2)
        label_position = label_position.view(B, C, -1, 2)
        
        pre_PAD_flag = torch.all(pre_position == PAD_ID, dim=-1)
        label_PAD_flag = torch.all(label_position == PAD_ID, dim=-1)

        pre_label_dis = (pre_position[:, :, :, None] - label_position[:, :, None]) ** 2
        pre_label_dis = torch.sqrt(pre_label_dis.sum(dim=-1))
        pre_label_dis = pre_label_dis.masked_fill_(label_PAD_flag.unsqueeze(2), 1e9) # PADの影響をなくすため
        pre_label_dis = pre_label_dis.min(dim=-1)[0]
        pre_label_dis = pre_label_dis.masked_fill_(pre_PAD_flag, -1e9) # PADの影響をなくすため
        pre_label_dis = pre_label_dis.max(dim=-1)[0]

        label_pre_dis = (label_position[:, :, :, None] - pre_position[:, :, None]) ** 2
        label_pre_dis = torch.sqrt(label_pre_dis.sum(dim=-1))
        label_pre_dis = label_pre_dis.masked_fill_(pre_PAD_flag.unsqueeze(2), 1e9) # PADの影響をなくすため
        label_pre_dis = label_pre_dis.min(dim=-1)[0]
        label_pre_dis = label_pre_dis.masked_fill_(label_PAD_flag, -1e9) # PADの影響をなくすため
        label_pre_dis = label_pre_dis.max(dim=-1)[0]

        HD = torch.max(pre_label_dis, label_pre_dis)
        self.__HD = HD.sum(dim=0)
        self.sample_number += HD.shape[0]

    def __call__(self):  #class HDとmean HD が出力される。
        HD = self.__HD / self.sample_number
        m_HD = self.__HD.mean() / self.sample_number

        self._reset()

        return HD, m_HD

"""