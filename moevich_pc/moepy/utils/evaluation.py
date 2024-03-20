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

#疑似ラベルのignore classを定義
IGNORE = 211

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
        raise ValueError("入力テンソルのindex番号がvector_dimより大きくなっています")
    
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
        self.__Intersection = 0
        self.__union = 0

    def _reset(self):
        self.__Intersection = 0
        self.__union = 0
    
    def update(self, outputs, targets, mode='softmax'):
        assert mode in ['softmax','sigmoid','linear'], "IoUのmodeが例外を発生させています。"
        
        #各種モードで処理
        if mode == "softmax":
            outputs = F.softmax(outputs, dim=1)
        elif mode == "sigmoid":
            outputs = torch.sigmoid(outputs)
        elif mode == "linear":
            pass


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

    def __call__(self):#IoUとmean_IoUが出力される。
        IoU = (self.__Intersection / (self.__union + 1e-24)).float()
        mean_IoU = IoU.mean()

        self._reset()

        return IoU, mean_IoU

#疑似ラベルのprecision Recall F1 Accracyを求める関数
#Precition  予測が正解した割合
#Recall     ラベル(正解)がどれだけ予測されたかの割合
#F1         上記2つの総合的な評価
#Accracy    直感的な正解率
#参考サイト: https://tech.ledge.co.jp/entry/metrics
class Pseudo_accracy():
    def __init__(self, classification):
        self.__classification = classification
        self.__TP = 0
        self.__FP = 0
        self.__FN = 0

    def _reset(self):
        self.__TP = 0
        self.__FP = 0
        self.__FN = 0
    
    def update(self, outputs, targets):
        #outputsにはignoreが含まれている

        #one hot vector にしたいため、outputsのignoreをself.__classificationの値にする
        outputs[outputs == IGNORE] = self.__classification

        #one hot vector化 ignoreは最後のクラス
        pre_one_hot_vector = one_hot_changer(outputs, self.__classification + 1, dim=1, bool_=True)
        label_one_hot_vector = one_hot_changer(targets, self.__classification + 1, dim=1, bool_=True)
        


    def __call__(self):#IoUとmean_IoUが出力される。
        IoU = (self.__Intersection / (self.__union + 1e-24)).float()
        mean_IoU = IoU.mean()

        self._reset()

        return IoU, mean_IoU

