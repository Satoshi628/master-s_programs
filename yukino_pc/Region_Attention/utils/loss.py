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

    # one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor]

    # one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    # もし-1ならそのまま出力
    if dim == -1:
        return vector
    # もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1  # omsertは-1が最後から一つ手前

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            # N,C,H,W => N,C,H*W
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class IoULoss(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets=>[batch,256,256]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        loss = (intersectoin + 1e-24) / (union + 1e-24)
        loss = loss.mean()
        return 1-loss


class Edge_IoULoss(nn.Module):
    def __init__(self, n_class, edge_range=3, lamda=1.0):
        super().__init__()
        self.n_class = n_class
        #self.avgPool = nn.AvgPool2d(edge_range, stride=1, padding=(edge_range - 1) // 2)
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        self.lamda = lamda

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        return object_edge_inside_flag

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets [batch,H,W] => [batch,class,H,W]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        # normal IoU loss
        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        IoU_loss = IoU_loss.mean()

        # edge IoU Loss
        predicts_idx = outputs.argmax(dim=1)
        predicts_seg_map = one_hot_changer(predicts_idx, self.n_class, dim=1)
        predict_edge = self.edge_decision(predicts_seg_map)
        targets_edge = self.edge_decision(targets)

        outputs = outputs * predict_edge
        targets = targets * targets_edge

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = edge_IoU_loss.mean()

        loss = 1 - IoU_loss + self.lamda * (1 - edge_IoU_loss)

        return loss

#物体の中心位置を求め、境界線と中心位置の距離をラベルと二乗誤差を計算し損失に加える
#中心位置の求め方は何回か平滑化フィルターを通してから極値を求める感じ


class Center_Edge_IoULoss(nn.Module):
    def __init__(self, n_class, edge_range=3, lamda=1.0):
        super().__init__()
        self.n_class = n_class
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, padding=edge_range)
        self.lamda = lamda

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        return object_edge_inside_flag

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets [batch,H,W] => [batch,class,H,W]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        # normal IoU loss
        intersectoin = (outputs * targets)
        total = (outputs + targets)
        union = total - intersectoin
        IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        IoU_loss = IoU_loss

        # edge IoU Loss
        predicts_idx = outputs.argmax(dim=1)
        predicts_seg_map = one_hot_changer(predicts_idx, self.n_class, dim=1)
        predict_edge = self.edge_decision(predicts_seg_map)
        targets_edge = self.edge_decision(targets)

        outputs = outputs[predict_edge == 1.0]
        targets = targets[targets_edge == 1.0]

        intersectoin = (outputs * targets).sum()
        total = (outputs + targets).sum()
        union = total - intersectoin
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = edge_IoU_loss

        loss = 1 - IoU_loss + self.lamda * (1 - edge_IoU_loss)

        return loss

#中心位置を特定し、半径rのbound boxでのIoUを計算する
class Center_Edge_IoULoss2(nn.Module):
    def __init__(self, n_class, edge_range=3, bb_range=7, lamda=1.0):
        super().__init__()
        self.n_class = n_class
        self.edge_avgPool = nn.AvgPool2d(edge_range, padding=(edge_range - 1) // 2)
        self.bb_avgPool = nn.AvgPool2d(bb_range, padding=(bb_range - 1) // 2)
        self.lamda = lamda

        #ガウシアンカーネル定義:kernel_size = 4*sigma+0.5
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        #極大値を探索するためのMaxpooling
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.edge_avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        return object_edge_inside_flag

    def make_center_bb(self,seg_map):
        h = seg_map
        h = (h == 1.0) * (h == self.maxpool(h)) * 1.0
        for _ in range(3):
            h = self.gaussian_filter(h)
            h = (h != 0) * (h == self.maxpool(h)) * 1.0
        
        bb = self.bb_avgPool(h) != 0.
        
        return 
    
    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets [batch,H,W] => [batch,class,H,W]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        # normal IoU loss
        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        IoU_loss = IoU_loss.mean()

        # edge IoU Loss
        predicts_idx = outputs.argmax(dim=1)
        predicts_seg_map = one_hot_changer(predicts_idx, self.n_class, dim=1)
        predict_edge = self.edge_decision(predicts_seg_map)
        targets_edge = self.edge_decision(targets)

        outputs = outputs * predict_edge
        targets = targets * targets_edge

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = edge_IoU_loss.mean()

        loss = 1 - IoU_loss + self.lamda * (1 - edge_IoU_loss)

        return loss
