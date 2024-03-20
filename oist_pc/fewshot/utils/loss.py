#coding: utf-8
#----- 標準ライブラリ -----#
# None
import math

#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm

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
        raise ValueError(f"入力テンソルのindex番号がvector_dimより大きくなっています\ntensor.max():{tensor.max()}")

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

# class balancing weight
def get_class_balancing_weight(class_count, class_num):
    weight = class_count.sum() / (class_num * class_count)
    return torch.tensor(weight)

# class rate weight
def get_class_rate_weight(class_count, class_num):
    weight = 1 - class_count / class_count.sum()
    weight = torch.tensor(weight)
    weight = weight.clamp(min=0.1)
    return weight


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
    def __init__(self, n_class, weight=None):
        super().__init__()
        self.n_class = n_class
        if weight is None:
            self.weight = torch.ones([self.n_class])
        else:
            self.weight = weight

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets=>[batch,256,256]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        loss = (intersectoin + 1e-24) / (union + 1e-24)
        loss = 1 - loss
        loss = self.weight.to(loss)[None] * loss
        loss = loss.mean()
        return loss


class Edge_IoULoss(nn.Module):
    def __init__(self, n_class, edge_range=3, lamda=1.0, weight=None):
        super().__init__()
        self.n_class = n_class
        
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        if weight is None:
            self.weight = torch.ones([self.n_class])
        else:
            self.weight = weight
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

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)

        IoU_loss = 1 - IoU_loss
        IoU_loss = self.weight.to(IoU_loss)[None] * IoU_loss

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
        edge_IoU_loss = 1 - edge_IoU_loss
        edge_IoU_loss = self.weight.to(edge_IoU_loss)[None] * edge_IoU_loss


        return IoU_loss.mean() + edge_IoU_loss.mean() * self.lamda


class difficult_select_IoULoss(nn.Module):
    def __init__(self, n_class, threshold=0.7, lamda=1.0):
        super().__init__()
        self.n_class = n_class
        self.threshold = threshold
        self.lamda = lamda

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets [batch,H,W] => [batch,class,H,W]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        # difficult select IoU Loss
        predicts_value, _ = outputs.max(dim=1)
        selected_area = 1.0 * (predicts_value < self.threshold)

        outputs = outputs * selected_area[:, None]
        targets = targets * selected_area[:, None]

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        Loss = (intersectoin + 1e-24) / (union + 1e-24)
        Loss = Loss.mean()

        Loss = 1 - Loss

        return Loss * self.lamda

#一様にピクセルをとってくる
class Contrastive_Loss(nn.Module):
    def __init__(self, n=64, weight=None):
        super().__init__()
        self.n = torch.Size([n])
        self.CE = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, output, feature, targets):
        CE_loss = self.CE(output, targets)
        b,c,H,W = feature.shape
        #probability distribution
        PD = torch.ones([b, H, W], device=feature.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする
        
        sampler = Categorical(PD)
        # [n, batch]
        sample_idx = sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        sample_idx = sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        choice_feature = torch.gather(feature, dim=-1, index=sample_idx)
        # [batch,n]
        choice_targets = torch.gather(targets, dim=-1, index=sample_idx[:, 0])
        
        choice_feature = F.normalize(choice_feature,dim=1)
        match = (choice_targets[:, :, None] == choice_targets[:, None]) * 1.0
        cos_sim = choice_feature[:, :, :, None] * choice_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        loss = (cos_sim - match) ** 2
        
        return CE_loss + loss.mean()


#クラスが同じ確率になるようにピクセルをとってくる
class Contrastive_Loss_classbalance(nn.Module):
    def __init__(self, n_class, n=64, weight=None):
        super().__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.CE = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, output, feature, targets):
        CE_loss = self.CE(output, targets)
        b,c,H,W = feature.shape
        #probability distribution
        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]

        sampler = Categorical(PD)
        # [n, batch]
        sample_idx = sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        sample_idx = sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        choice_feature = torch.gather(feature, dim=-1, index=sample_idx)
        # [batch,n]
        choice_targets = torch.gather(targets, dim=-1, index=sample_idx[:, 0])
        
        choice_feature = F.normalize(choice_feature,dim=1)
        match = (choice_targets[:, :, None] == choice_targets[:, None]) * 1.0
        cos_sim = choice_feature[:, :, :, None] * choice_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        loss = (cos_sim - match) ** 2
        
        return CE_loss + loss.mean()


#anchorを物体の中心のピクセルからchildを物体の輪郭からとってくる
class Contrastive_Loss_edge(nn.Module):
    def __init__(self, n_class, edge_range=3, n=64, weight=None):
        super().__init__()
        self.n_class = n_class
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.n = torch.Size([n])
        self.CE = nn.CrossEntropyLoss(weight=weight)
    
    def anchor_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag
    
    def forward(self, output, feature, targets):
        CE_loss = self.CE(output, targets)
        b,c,H,W = feature.shape
        #probability distribution
        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.anchor_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        inside_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        edge_PD = edge_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(inside_PD == 0., dim=-1)
        inside_PD[zero_flag] = 1 / inside_PD.shape[-1]
        zero_flag = torch.all(edge_PD == 0., dim=-1)
        edge_PD[zero_flag] = 1 / edge_PD.shape[-1]
        
        #GPUを用いて計算するとエラーが出る場合がある。
        inside_sampler = Categorical(inside_PD.cpu())
        edge_sampler = Categorical(edge_PD.cpu())

        # [n, batch]
        inside_sample_idx = inside_sampler.sample(self.n).cuda(output.device)
        edge_sample_idx = edge_sampler.sample(self.n).cuda(output.device)

        # [n, batch] => [batch,c,n]
        inside_sample_idx = inside_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        edge_sample_idx = edge_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        choice_feature_anchor = torch.gather(feature, dim=-1, index=inside_sample_idx)
        choice_feature_child = torch.gather(feature, dim=-1, index=edge_sample_idx)
        # [batch,n]
        choice_targets_anchor = torch.gather(targets, dim=-1, index=inside_sample_idx[:, 0])
        choice_targets_child = torch.gather(targets, dim=-1, index=edge_sample_idx[:, 0])
        
        choice_feature_anchor = F.normalize(choice_feature_anchor, dim=1)
        choice_feature_child = F.normalize(choice_feature_child,dim=1)
    
        match = (choice_targets_anchor[:, :, None] == choice_targets_child[:, None]) * 1.0
        cos_sim = choice_feature_anchor[:, :, :, None] * choice_feature_child[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        loss = (cos_sim - match) ** 2
        
        return CE_loss + loss.mean()


class Contrastive_Loss_moepy(torch.nn.Module):
    def __init__(self, n_class, weight=None):
        super(Contrastive_Loss_moepy, self).__init__()
        self.n_class = n_class
        
        self.CE = nn.CrossEntropyLoss(weight=weight)

    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        b, c, h, w = class_feature[0].size()

        Loss = 0

        #errorflag = [False, False, False, False]
        errorflag = [0 for _ in range(self.n_class)]
        for class_num in range(self.n_class):
            feature_vector = class_feature[class_num].permute(0, 2, 3, 1)[targets == class_num]  #a0にclass0のピクセルのみを入れる
            if feature_vector.size(0) == 0: #class0のピクセルが0個のとき
                errorflag[class_num] = 1 #errorflagのclass0をTrueにする
                feature_vector = feature_vector.new_zeros([1, c])  #a0を1にする(0だと計算できないので無理やり1にする)

            rand_idx = torch.randint(feature_vector.size(0), [1])
            vector = feature_vector[rand_idx, :]  #[1,c]
            vector = vector.view(1, vector.size(0), vector.size(1), 1, 1) #[1, 個数, c, 1, 1]
            feature_map = class_feature[class_num].view(b, 1, class_feature[class_num].size(1), h, w) #[1, 個数, c, 256, 256]
            
            vector = F.normalize(vector, dim=2)  #channelに対して行う
            feature_map = F.normalize(feature_map, dim=2)  #channelに対して行う

            inner = (vector * feature_map).sum(dim=2)
            mask = 2.0 * (class_num == targets) - 1.0  #class0の点は1それ以外は-1
            MSE_Loss = pow((inner - mask), 2)
            MSE_Loss = MSE_Loss.mean() * (1. - errorflag[class_num])
            Loss = Loss + MSE_Loss

        return Loss / self.n_class + self.CE(output, targets)


class Contrastive_Loss_moepy2(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", edge_range=3, n=64, weight=None):
        super(Contrastive_Loss_moepy2, self).__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)

        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
            print(choice_type)
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
            print(choice_type)
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
            print(choice_type)
        

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]

        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        b, c, h, w = class_feature[0].size()

        #バッチ内にクラスがない場合があるため、平均化のための重み作成
        mean_weight = b * self.n[0] ** 2
        

        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            anchor_feature = F.normalize(anchor_feature,dim=1)
            child_feature = F.normalize(child_feature,dim=1)
            match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
            cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
            cos_sim = cos_sim.sum(dim=1)

            MSE_Loss = (cos_sim - match) ** 2
            Loss = Loss + MSE_Loss.sum() / mean_weight


        return Loss + self.CE(output, targets)


class Contrastive_Loss_moepy3(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", edge_range=3, n=64, weight=None):
        super(Contrastive_Loss_moepy3, self).__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)

        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
            print(choice_type)
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
            print(choice_type)
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
            print(choice_type)

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]

        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        b, c, h, w = class_feature[0].size()

        #バッチ内にクラスがない場合があるため、平均化のための重み作成
        mean_weight = b * self.n[0] ** 2
        

        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            anchor_feature = F.normalize(anchor_feature,dim=1)
            child_feature = F.normalize(child_feature, dim=1)
            
            match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
            cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
            cos_sim = cos_sim.sum(dim=1)

            MSE_Loss = (cos_sim - match) ** 2

            anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
            anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
            anchor_cos_sim = anchor_cos_sim.sum(dim=1)

            anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

            Loss = Loss + (MSE_Loss.sum() + anchor_MSE_Loss.sum()) / mean_weight

        return Loss + self.CE(output, targets)


class Contrastive_Loss_moepy4(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", loss_type="entropy", kernel_type="normal", edge_range=3, n=64, temperature=0.07, weight=None):
        super(Contrastive_Loss_moepy4, self).__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.temperature = temperature
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)


        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
            print(choice_type)
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
            print(choice_type)
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
            print(choice_type)
        
        if loss_type == "Cos01":
            self.loss_function = self.Loss_Cos01
            print(loss_type)
        if loss_type == "Cos11":
            self.loss_function = self.Loss_Cos_11
            print(loss_type)
        if loss_type == "entropy":
            self.loss_function = self.Loss_logsoftmax
            print(loss_type)
        
        if kernel_type == "normal":
            self.softmax_function = self.normal_softmax
            print(kernel_type)
        if kernel_type == "linear":
            self.softmax_function = self.linear_kernel_softmax
            print(kernel_type)
        if kernel_type == "gausiann":
            self.softmax_function = self.gausiaan_kernel_softmax
            print(kernel_type)

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]

        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def Loss_Cos01(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1. * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 1. * (anchor_targets[:, :, None] == anchor_targets[:, None])
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def Loss_Cos_11(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def normal_softmax(self, anchor_targets, child_targets, anchor_feature, child_feature):
        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1.0 * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)
        return torch.exp(match * cos_sim / self.temperature).sum(dim=-1) / torch.exp((1 - match) * cos_sim / self.temperature).sum(dim=-1)

    def linear_kernel_softmax(self, anchor_targets, child_targets, anchor_feature, child_feature):
        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1.0 * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)
        return (match * cos_sim / self.temperature).sum(dim=-1) / ((1 - match) * cos_sim / self.temperature).sum(dim=-1)


    def gausiaan_kernel_softmax(self, anchor_targets, child_targets, anchor_feature, child_feature):
        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1.0 * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] - child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)
        return (match * cos_sim / self.temperature).sum(dim=-1) / ((1 - match) * cos_sim / self.temperature).sum(dim=-1)


    def Loss_logsoftmax(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """-log(softmax(exp(Cos(An,Ch))). negative pairs are at denominator. positive pairs are at numerator.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        softmax = self.softmax_function(anchor_targets, child_targets, anchor_feature, child_feature)

        entropy_Loss = -torch.log(softmax)

        softmax = self.softmax_function(anchor_targets, anchor_targets, anchor_feature, anchor_feature)

        anchor_entropy_Loss = -torch.log(softmax)

        return entropy_Loss.mean() + anchor_entropy_Loss.mean()


    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            Loss = Loss + self.loss_function(anchor_targets, child_targets, anchor_feature, child_feature)

        return Loss + self.CE(output, targets)


class Contrastive_Loss_moepy5(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", loss_type="entropy", edge_range=3, n=64, k=16, weight=None):
        super().__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.kappa = k
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)


        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
        
        print(choice_type)
        
        if loss_type == "Cos01":
            self.loss_function = self.Loss_Cos01
        if loss_type == "Cos11":
            self.loss_function = self.Loss_Cos_11

        print(loss_type)
        print("tvMF")
        print("5です")

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]

        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def Loss_Cos01(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1. * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 1. * (anchor_targets[:, :, None] == anchor_targets[:, None])
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def Loss_Cos_11(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        # t-vMF
        #anchor_cos_sim = (1. + anchor_cos_sim).div(1. + (1. - anchor_cos_sim).mul(self.kappa)) - 1.

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()


    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            Loss = Loss + self.loss_function(anchor_targets, child_targets, anchor_feature, child_feature)

        return Loss + self.CE(output, targets)


class Contrastive_Loss_moepy_tvmf(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", loss_type="entropy", edge_range=3, n=64, k=16, weight=None):
        super().__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.kappa = k
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)


        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
        
        print(choice_type)
        
        if loss_type == "Cos01":
            self.loss_function = self.Loss_Cos01
        if loss_type == "Cos11":
            self.loss_function = self.Loss_Cos_11

        print(loss_type)
        print("tvMF")
        print("5です")

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]

        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def Loss_Cos01(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1. * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 1. * (anchor_targets[:, :, None] == anchor_targets[:, None])
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def Loss_Cos_11(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        # t-vMF
        anchor_cos_sim = (1. + anchor_cos_sim).div(1. + (1. - anchor_cos_sim).mul(self.kappa)) - 1.

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()


    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            Loss = Loss + self.loss_function(anchor_targets, child_targets, anchor_feature, child_feature)

        return Loss + self.CE(output, targets)



class Contrastive_Loss_moepy6(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", loss_type="entropy", edge_range=3, n=64, k=16, weight=None):
        super().__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.kappa = k
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)


        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
            print(choice_type)
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
            print(choice_type)
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
            print(choice_type)
        
        if loss_type == "Cos01":
            self.loss_function = self.Loss_Cos01
            print(loss_type)
        if loss_type == "Cos11":
            self.loss_function = self.Loss_Cos_11
            print(loss_type)
        print("tvMF_my")

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、すべてゼロになる
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]

        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def Loss_Cos01(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1. * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        # my-vMF
        exp_2k =math.exp(-2*self.kappa) 
        cos_sim = 2. * (torch.exp(-self.kappa * torch.sqrt(2 - 2 * cos_sim)) - exp_2k).div(1 - exp_2k) - 1.


        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 1. * (anchor_targets[:, :, None] == anchor_targets[:, None])
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def Loss_Cos_11(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        # my-vMF
        # torch.clampはnan対策。まるめ誤差によりcos_simは1を超える場合あり
        exp_2k = math.exp(-2 * self.kappa)
        sqrt_assurance = torch.clamp(2. - 2. * cos_sim, min=1e-7)
        cos_sim = 2. * (torch.exp(-self.kappa * torch.sqrt(sqrt_assurance)) - exp_2k).div(1 - exp_2k) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        # my-vMF
        # torch.clampはnan対策。まるめ誤差によりcos_simは1を超える場合あり
        sqrt_assurance = torch.clamp(2. - 2. * anchor_cos_sim, min=1e-7)
        anchor_cos_sim = 2. * (torch.exp(-self.kappa * torch.sqrt(sqrt_assurance)) - exp_2k).div(1 - exp_2k) - 1.

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            Loss = Loss + self.loss_function(anchor_targets, child_targets, anchor_feature, child_feature)

        return Loss + self.CE(output, targets)


class Contrastive_Loss_moepy7(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", loss_type="entropy", edge_range=3, n=64, k=16, weight=None):
        super().__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.kappa = k
        self.exp_2k = math.exp(-(2) ** (1 / self.kappa))
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)


        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
            print(choice_type)
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
            print(choice_type)
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
            print(choice_type)
        
        if loss_type == "Cos01":
            self.loss_function = self.Loss_Cos01
            print(loss_type)
        if loss_type == "Cos11":
            self.loss_function = self.Loss_Cos_11
            print(loss_type)
        print("tvMF_my2")

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = 1 / torch.clamp(inside_class_rate, min=1e-7) / self.n_class
        #inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        #inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = 1 / torch.clamp(edge_class_rate, min=1e-7) / self.n_class
        #edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        #edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様分布にする
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def Loss_Cos01(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1. * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        # my-vMF
        exp_2k =math.exp(-2*self.kappa) 
        cos_sim = 2. * (torch.exp(-self.kappa * torch.sqrt(2 - 2 * cos_sim)) - exp_2k).div(1 - exp_2k) - 1.


        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 1. * (anchor_targets[:, :, None] == anchor_targets[:, None])
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def Loss_Cos_11(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        # my-vMF
        # torch.clampはnan対策。まるめ誤差によりcos_simは1を超える場合あり
        sqrt_assurance = torch.clamp(2. - 2. * cos_sim, min=1e-7)
        cos_sim = 2. * (torch.exp(-(sqrt_assurance) ** (1 / (2 * self.kappa))) - self.exp_2k).div(1 - self.exp_2k) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        # my-vMF
        # torch.clampはnan対策。まるめ誤差によりcos_simは1を超える場合あり
        sqrt_assurance = torch.clamp(2. - 2. * anchor_cos_sim, min=1e-7)
        anchor_cos_sim = 2. * (torch.exp(-(sqrt_assurance) ** (1 / (2 * self.kappa))) - self.exp_2k).div(1 - self.exp_2k) - 1.

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            Loss = Loss + self.loss_function(anchor_targets, child_targets, anchor_feature, child_feature)

        return Loss + self.CE(output, targets)


class Contrastive_Loss_moepy7(torch.nn.Module):
    def __init__(self, n_class, choice_type="uniform", loss_type="entropy", edge_range=3, n=64, k=16, weight=None):
        super().__init__()
        self.n_class = n_class
        self.n = torch.Size([n])
        self.kappa = k
        self.exp_2k = math.exp(-(2) ** (1 / self.kappa))
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)
        self.CE = nn.CrossEntropyLoss(weight=weight)


        if choice_type == "uniform":
            self.choice_fnc = self.uniform_choice
            print(choice_type)
        if choice_type == "class_balance":
            self.choice_fnc = self.class_balance_choice
            print(choice_type)
        if choice_type == "edge":
            self.choice_fnc = self.edge_inside_choice
            print(choice_type)
        
        if loss_type == "Cos01":
            self.loss_function = self.Loss_Cos01
            print(loss_type)
        if loss_type == "Cos11":
            self.loss_function = self.Loss_Cos_11
            print(loss_type)
        print("tvMF_my2")

    def uniform_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = torch.ones([b, H, W], device=targets.device)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする

        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def class_balance_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        class_rate = PD.sum(dim=(-2, -1))
        class_rate = class_rate / class_rate.sum(dim=-1, keepdim=True)
        class_rate = 1 - class_rate
        PD = PD * class_rate[:, :, None, None]
        PD = PD.sum(dim=1)
        PD = PD / torch.clamp(PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        PD = PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様にする
        zero_flag = torch.all(PD == 0., dim=-1)
        PD[zero_flag] = 1 / PD.shape[-1]
        
        anchor_PD = child_PD = PD
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def edge_region_maker(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        object_inside_inside_flag = seg_map * (smooth_map == seg_map)

        return object_inside_inside_flag, object_edge_inside_flag

    def edge_inside_choice(self, feature, targets, classes=0):
        #batch内の画像にクラスが含まれないのなら計算しない
        class_non_flag = torch.any((targets == classes).flatten(-2), dim=-1)
        feature = feature[class_non_flag]
        targets = targets[class_non_flag]

        b, c, H, W = feature.shape

        if b == 0:
            return None, None, None, None

        PD = one_hot_changer(targets, self.n_class, dim=1)
        inside_PD, edge_PD = self.edge_region_maker(PD)

        inside_class_rate = inside_PD.sum(dim=(-2, -1))
        inside_class_rate = 1 / torch.clamp(inside_class_rate, min=1e-7) / self.n_class
        #inside_class_rate = inside_class_rate / inside_class_rate.sum(dim=-1, keepdim=True)
        #inside_class_rate = 1 - inside_class_rate
        inside_PD = inside_PD * inside_class_rate[:, :, None, None]
        inside_PD = inside_PD.sum(dim=1)
        inside_PD = inside_PD / torch.clamp(inside_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        anchor_PD = inside_PD.flatten(-2) #マップ1枚からサンプリングするためフラットにする

        edge_class_rate = edge_PD.sum(dim=(-2, -1))
        edge_class_rate = 1 / torch.clamp(edge_class_rate, min=1e-7) / self.n_class
        #edge_class_rate = edge_class_rate / edge_class_rate.sum(dim=-1, keepdim=True)
        #edge_class_rate = 1 - edge_class_rate
        edge_PD = edge_PD * edge_class_rate[:, :, None, None]
        edge_PD = edge_PD.sum(dim=1)
        edge_PD = edge_PD / torch.clamp(edge_PD.sum(dim=(-2, -1), keepdim=True), min=1e-7)  #正規化
        child_PD = edge_PD.flatten(-2)  #マップ1枚からサンプリングするためフラットにする
        
        # 1class のみ移っている場合、一様分布にする
        zero_flag = torch.all(anchor_PD == 0., dim=-1)
        anchor_PD[zero_flag] = 1 / anchor_PD.shape[-1]
        zero_flag = torch.all(child_PD == 0., dim=-1)
        child_PD[zero_flag] = 1 / child_PD.shape[-1]
        
        anchor_sampler = Categorical(anchor_PD)
        child_sampler = Categorical(child_PD)
        # [n, batch]
        anchor_sample_idx = anchor_sampler.sample(self.n)
        child_sample_idx = child_sampler.sample(self.n)
        # [n, batch] => [batch,c,n]
        anchor_sample_idx = anchor_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)
        child_sample_idx = child_sample_idx[:, :, None].permute(1, 2, 0).expand(-1, c, -1)

        # [batch,c,H,W] => [batch,c,H*W]
        feature = feature.flatten(-2)
        # [batch,H,W] => [batch,H*W]
        targets = targets.flatten(-2)

        # [batch,c,n]
        anchor_feature = torch.gather(feature, dim=-1, index=anchor_sample_idx)
        child_feature = torch.gather(feature, dim=-1, index=child_sample_idx)
        # [batch,n]
        anchor_targets = torch.gather(targets, dim=-1, index=anchor_sample_idx[:, 0])
        child_targets = torch.gather(targets, dim=-1, index=child_sample_idx[:, 0])

        return anchor_feature, anchor_targets, child_feature, child_targets

    def Loss_Cos01(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 1. * (anchor_targets[:, :, None] == child_targets[:, None])
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        # my-vMF
        exp_2k = math.exp(-2*self.kappa) 
        cos_sim = 2. * (torch.exp(-self.kappa * torch.sqrt(2 - 2 * cos_sim)) - exp_2k).div(1 - exp_2k) - 1.


        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 1. * (anchor_targets[:, :, None] == anchor_targets[:, None])
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def Loss_Cos_11(self, anchor_targets, child_targets, anchor_feature, child_feature):
        """(Ⅰ-Cos(An,Ch))^2. I=0or1.

        Args:
            anchor_targets (Tensor[batch, num]): anchor class label
            child_targets (Tensor[batch, num]): child class label
            anchor_feature (Tensor[batch, dim, num]): anchor feature vector
            child_feature (Tensor[batch, dim, num]): child feature vector
        """        

        anchor_feature = F.normalize(anchor_feature,dim=1)
        child_feature = F.normalize(child_feature, dim=1)

        match = 2. * (anchor_targets[:, :, None] == child_targets[:, None]) - 1
        cos_sim = anchor_feature[:, :, :, None] * child_feature[:, :, None]
        cos_sim = cos_sim.sum(dim=1)

        # t-vMF
        #cos_sim = (1. + cos_sim).div(1. + (1. - cos_sim).mul(self.kappa)) - 1.

        # my-vMF
        # torch.clampはnan対策。まるめ誤差によりcos_simは1を超える場合あり
        sqrt_assurance = torch.clamp(2. - 2. * cos_sim, min=1e-7)
        cos_sim = 2. * (torch.exp(-(sqrt_assurance) ** (1 / (2 * self.kappa))) - self.exp_2k).div(1 - self.exp_2k) - 1.

        MSE_Loss = (cos_sim - match) ** 2

        anchor_match = 2. * (anchor_targets[:, :, None] == anchor_targets[:, None]) - 1
        anchor_cos_sim = anchor_feature[:, :, :, None] * anchor_feature[:, :, None]
        anchor_cos_sim = anchor_cos_sim.sum(dim=1)

        # my-vMF
        # torch.clampはnan対策。まるめ誤差によりcos_simは1を超える場合あり
        sqrt_assurance = torch.clamp(2. - 2. * anchor_cos_sim, min=1e-7)
        anchor_cos_sim = 2. * (torch.exp(-(sqrt_assurance) ** (1 / (2 * self.kappa))) - self.exp_2k).div(1 - self.exp_2k) - 1.

        anchor_MSE_Loss = (anchor_cos_sim - anchor_match) ** 2

        return MSE_Loss.mean() + anchor_MSE_Loss.mean()

    def forward(self, output, class_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        for class_num in range(self.n_class):
            anchor_feature, anchor_targets, child_feature, child_targets = self.choice_fnc(class_feature[class_num], targets, classes=class_num)
            if anchor_feature is None:
                continue

            Loss = Loss + self.loss_function(anchor_targets, child_targets, anchor_feature, child_feature)

        return Loss + self.CE(output, targets)



# Contrastive Learning for Label Efficient Semantic Segmentation
class CLLE_Loss(torch.nn.Module):
    def __init__(self, n_class, weight=None):
        super().__init__()
        self.n_class = n_class

    def Within_image_loss(self, feature, JIT_feature, targets):
        
        # [b,N,N]
        cos_sim = (feature[:, :, :, None] * JIT_feature[:, :, None]).sum(dim=1)
        
        cos_sim = F.softmax(cos_sim / 0.07, dim=-1)
        
        Loss = 0
        for class_num in range(self.n_class):
            targets_class = targets == class_num
            #[b,N,N]
            match_flag = targets_class[:, :, None] == targets_class[:, None]

            Loss = Loss - torch.log(cos_sim[match_flag]).mean()

        return Loss / self.n_class

    def Cross_image_loss(self, feature, JIT_feature, targets):
        # [b,N,N]
        itself_cos_sim = (feature[:, :, :, None] * JIT_feature[:, :, None]).sum(dim=1)
        itself_cos_sim = torch.exp(itself_cos_sim / 0.07)

        #1つずらす
        JIT_feature = torch.roll(JIT_feature, 1, dims=0)
        roll_targets = torch.roll(targets, 1, dims=0)
        # [b,N,N]
        cross_cos_sim = (feature[:, :, :, None] * JIT_feature[:, :, None]).sum(dim=1)
        cross_cos_sim = torch.exp(cross_cos_sim / 0.07)

        Loss = 0
        for class_num in range(self.n_class):
            targets_class = targets == class_num
            roll_targets_class = roll_targets == class_num
            #[b,N,N]
            match_flag = targets_class[:, :, None] == targets_class[:, None]
            roll_match_flag = targets_class[:, :, None] == roll_targets_class[:, None]
            
            class_cross_cos_sim = roll_match_flag * cross_cos_sim
            class_cos = torch.cat([itself_cos_sim, class_cross_cos_sim], dim=-1)
            class_match_flag = torch.cat([match_flag, roll_match_flag], dim=-1)

            #softmax
            class_cos = class_cos / (1e-7 + class_cos.sum(dim=-1, keepdim=True))

            Loss = Loss - torch.log(class_cos[class_match_flag]).mean()

        return Loss/ self.n_class

    def forward(self, output, JIT_output, feature, JIT_feature, targets):  # output 出力(64次元)  targets ラベル
        #class_feature => [tensor[b,c,H,W], ..., tensor[b,c,H,W]] 
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        
        Loss = 0

        targets = F.interpolate(targets[None].float(), size=feature.shape[-2:])[0]
        targets = targets.long()
        feature = F.normalize(feature, dim=1).flatten(-2)
        JIT_feature = F.normalize(JIT_feature, dim=1).flatten(-2)
        targets = targets.flatten(-2)

        Loss = Loss + self.Within_image_loss(feature, JIT_feature, targets)*0.5
        Loss = Loss + self.Cross_image_loss(feature, JIT_feature, targets)*0.5


        return Loss


class SquareLoss(torch.nn.Module):
    def __init__(self):
        super(SquareLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()


    def forward(self, logits, feature, targets):
        CE_Loss = self.CE(logits, targets)
        
        h_0, h_1, h_2, h_3 = feature
        b, c, h, w = h_0.size()


        errorflag = [False, False, False, False]
        #errorflag = [False, False, False]

        a0 = h_0.permute(0, 2, 3, 1)[targets == 0] #a0にclass0のピクセルのみを入れる
        #print("a0", a0.shape)
        a1 = h_1.permute(0, 2, 3, 1)[targets == 1] #a1にclass1のピクセルのみ入れる
        a2 = h_2.permute(0, 2, 3, 1)[targets == 2] #a2にclass2のピクセルのみ入れる
        a3 = h_3.permute(0, 2, 3, 1)[targets == 3] #a3にclass3のピクセルのみ入れる


        if a0.size(0) == 0: #class0のピクセルが0個のとき
            errorflag[0] = True #errorflagのclass0をTrueにする
            a0 = a0.new_zeros([1,c]) #a0を1にする(0だと計算できないので無理やり1にする)
        if a1.size(0) == 0:
            errorflag[1] = True
            a1 = a1.new_zeros([1,c])
        if a2.size(0) == 0:
            errorflag[2] = True
            a2 = a2.new_zeros([1,c])
        
        if a3.size(0) == 0:
            errorflag[3] = True
            a3 = a3.new_zeros([1,c])
        
        #print("a1", a1.shape)
        #print("a2", a2.shape)
        #print("a3", a3.shape)
        

        #a0～a3の各クラスからランダムに1点選ぶ
        #0～(個数-1)までの値から1つ選ぶ
        y0 = torch.randint(a0.size(0), [1])  
        y1 = torch.randint(a1.size(0), [1])
        y2 = torch.randint(a2.size(0), [1])
        y3 = torch.randint(a3.size(0), [1])
        

        y0 = a0[y0, :]  #[1,64]
        y1 = a1[y1, :]
        y2 = a2[y2, :]
        y3 = a3[y3, :]


        #選んだ点(y0～y3)と特徴マップ(h_0～h_3)の大きさを整形
        #y0～y3の大きさを[c] -> [1, c, 1, 1]にする
        y0 = y0.view(1, y0.size(0), y0.size(1), 1, 1) #[1, 個数, c, 1, 1]
        y1 = y1.view(1, y1.size(0), y1.size(1), 1, 1)
        y2 = y2.view(1, y2.size(0), y2.size(1), 1, 1)
        y3 = y3.view(1, y3.size(0), y3.size(1), 1, 1)

        h_0 = h_0.view(b, 1, h_0.size(1), h, w) #[1, 個数, c, 256, 256]
        h_1 = h_1.view(b, 1, h_1.size(1), h, w)
        h_2 = h_2.view(b, 1, h_2.size(1), h, w)
        h_3 = h_3.view(b, 1, h_3.size(1), h, w)

        
        #ベクトルの大きさを1にしておく
        #output = F.normalize(output, dim=1)
        y0 = F.normalize(y0, dim=2)  #channelに対して行う
        y1 = F.normalize(y1, dim=2)  
        y2 = F.normalize(y2, dim=2)  
        y3 = F.normalize(y3, dim=2)  

        h_0 = F.normalize(h_0, dim=2)  #channelに対して行う
        h_1 = F.normalize(h_1, dim=2)  
        h_2 = F.normalize(h_2, dim=2)  
        h_3 = F.normalize(h_3, dim=2)

        #選んだ点(a0～a3)と特徴マップ(h_0～h_1)を演算(cos類似度の計算)
        inner0 = (y0 * h_0).sum(dim=2)
        inner1 = (y1 * h_1).sum(dim=2)
        inner2 = (y2 * h_2).sum(dim=2)
        inner3 = (y3 * h_3).sum(dim=2)
        


        ###選んだ点に応じてラベルを見て同じものには1、それ以外には-1を[256,256]に入れる
        map0 = 2.0 * (0 == targets) - 1.0  #class0の点は1それ以外は-1
        map1 = 2.0 * (1 == targets) - 1.0  #class1の点は1それ以外は-1
        map2 = 2.0 * (2 == targets) - 1.0  #class2の点は1それ以外は-1
        map3 = 2.0 * (3 == targets) - 1.0  #class3の点は1それ以外は-1

        
        ###平均二乗誤差の計算
        loss0 = pow((inner0 - map0), 2)
        loss1 = pow((inner1 - map1), 2)
        loss2 = pow((inner2 - map2), 2)
        loss3 = pow((inner3 - map3), 2)
        
        #class0が0個のときはerrorflag[0]がTrueとなり1になるので、
        #1-errorflag[0]=0となり0個のclassのlossは計算されない
        loss0 = loss0.mean()*(1.-errorflag[0]) 
        loss1 = loss1.mean()*(1.-errorflag[1])
        loss2 = loss2.mean()*(1.-errorflag[2])
        loss3 = loss3.mean()*(1.-errorflag[3])

        loss4 = (loss0 + loss1 + loss2 + loss3)/4
        #loss4 = (loss0 + loss1 + loss2)/4
        loss4 = loss4 + loss0 + loss1 + loss2 + loss3

        #return loss, h, w
        return loss4 + CE_Loss
        #return loss0, loss1, loss2, loss4


class SquareLoss2(torch.nn.Module):
    def __init__(self):
        super(SquareLoss, self).__init__()

    def forward(self, h_0, h_1, h_2, h_3, targets1, y, h_00, h_11, h_22, h_33, targets2, yy):   # output 出力(64次元)  targets ラベル
    #def forward(self, h_0, h_1, h_2, targets1, y, h_00, h_11, h_22, targets2, yy):   # output 出力(64次元)  targets ラベル
        #output.size() => [b,c,h,w] [4, 64, 256, 256]
        #targets.size() => [b,h,w] [4, 256, 256]
        #print("outputkata", type(output))  <class 'torch.Tensor'>
        h_0, h_1, h_2, h_3 = output
        b, c, h, w = h_0.size()  #inputs1のb, c, h, w
        bb, cc, hh, ww = h_00.size() #inputs2のb, c, h, w

        #inputs1の方で点を作る
        #inputs2は特徴マップの方
        errorflag = [False, False, False, False]

        a0 = h_0.permute(0, 2, 3, 1)[targets1 == 0] #a0にclass0のピクセルのみを入れる
        #print("a0", a0.shape)
        a1 = h_1.permute(0, 2, 3, 1)[targets1 == 1] #a1にclass1のピクセルのみ入れる
        a2 = h_2.permute(0, 2, 3, 1)[targets1 == 2] #a2にclass2のピクセルのみ入れる
        a3 = h_3.permute(0, 2, 3, 1)[targets1 == 3] #a3にclass3のピクセルのみ入れる


        if a0.size(0) == 0: #class0のピクセルが0個のとき
            errorflag[0] = True #errorflagのclass0をTrueにする
            a0 = a0.new_zeros([1,c]) #a0を1にする(0だと計算できないので無理やり1にする)
        if a1.size(0) == 0:
            errorflag[1] = True
            a1 = a1.new_zeros([1,c])
        if a2.size(0) == 0:
            errorflag[2] = True
            a2 = a2.new_zeros([1,c])
        
        if a3.size(0) == 0:
            errorflag[3] = True
            a3 = a3.new_zeros([1,c])
        
        #print("a1", a1.shape)
        #print("a2", a2.shape)
        #print("a3", a3.shape)
        

        #a0～a3の各クラスからランダムに1点選ぶ
        #0～(個数-1)までの値から1つ選ぶ
        y0 = torch.randint(a0.size(0), [1])  
        y1 = torch.randint(a1.size(0), [1])
        y2 = torch.randint(a2.size(0), [1])
        y3 = torch.randint(a3.size(0), [1])
        

        y0 = a0[y0, :]  #[1,64]
        #print(y0.shape)
        y1 = a1[y1, :]
        y2 = a2[y2, :]
        y3 = a3[y3, :]


        #選んだ点(y0～y3)(inputs1の点)と特徴マップ(h_00～h_33)(inputs2のマップ)の大きさを整形
        #y0～y3の大きさを[c] -> [1, c, 1, 1]にする
        y0 = y0.view(1, y0.size(0), y0.size(1), 1, 1) #[1, 個数, c, 1, 1]
        y1 = y1.view(1, y1.size(0), y1.size(1), 1, 1)
        y2 = y2.view(1, y2.size(0), y2.size(1), 1, 1)
        y3 = y3.view(1, y3.size(0), y3.size(1), 1, 1)

        h_00 = h_00.view(bb, 1, h_00.size(1), hh, ww) #[1, 個数, c, 256, 256]
        h_11 = h_11.view(bb, 1, h_11.size(1), hh, ww)
        h_22 = h_22.view(bb, 1, h_22.size(1), hh, ww)
        h_33 = h_33.view(bb, 1, h_33.size(1), hh, ww)

        
        #ベクトルの大きさを1にしておく
        #output = F.normalize(output, dim=1)
        y0 = F.normalize(y0, dim=2)  #channelに対して行う
        y1 = F.normalize(y1, dim=2)  
        y2 = F.normalize(y2, dim=2)  
        y3 = F.normalize(y3, dim=2)  

        h_00 = F.normalize(h_00, dim=2)  #channelに対して行う
        h_11 = F.normalize(h_11, dim=2)  
        h_22 = F.normalize(h_22, dim=2)  
        h_33 = F.normalize(h_33, dim=2)

        #選んだ点(a0～a3)と特徴マップ(h_00～h_33)を演算(cos類似度の計算)
        inner0 = (y0 * h_00).sum(dim=2)
        inner1 = (y1 * h_11).sum(dim=2)
        inner2 = (y2 * h_22).sum(dim=2)
        inner3 = (y3 * h_33).sum(dim=2)
        


        ###選んだ点に応じてラベルを見て同じものには1、それ以外には-1を[256,256]に入れる
        map0 = 2.0 * (0 == targets2) - 1.0  #class0の点は1それ以外は-1
        map1 = 2.0 * (1 == targets2) - 1.0  #class1の点は1それ以外は-1
        map2 = 2.0 * (2 == targets2) - 1.0  #class2の点は1それ以外は-1
        map3 = 2.0 * (3 == targets2) - 1.0  #class3の点は1それ以外は-1

        
        ###平均二乗誤差の計算
        loss0 = pow((inner0 - map0), 2)
        loss1 = pow((inner1 - map1), 2)
        loss2 = pow((inner2 - map2), 2)
        #loss3 = pow((inner3 - map3), 2)
        """
        loss = loss0 + loss1 + loss2 + loss3
        loss = loss.mean()/4
        #print(loss.shape)
        """
        #class0が0個のときはerrorflag[0]がTrueとなり1になるので、
        #1-errorflag[0]=0となり0個のclassのlossは計算されない
        loss0 = loss0.mean()*(1.-errorflag[0]) 
        loss1 = loss1.mean()*(1.-errorflag[1])
        loss2 = loss2.mean()*(1.-errorflag[2])
        #loss3 = loss3.mean()*(1.-errorflag[3])

        loss4 = (loss0 + loss1 + loss2 + loss3)/4
        #loss4 =(loss0 + loss1 + loss2)/4

        #return loss, h, w
        return loss4
        #return loss0, loss1, loss2, loss4
