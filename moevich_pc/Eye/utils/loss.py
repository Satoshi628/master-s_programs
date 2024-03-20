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


class Contrastive_Loss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, targets1, targets2):
        Distance = (out1 - out2)**2
        Distance = torch.sqrt(Distance.sum(dim=-1))
        match = (targets1 == targets2) * 1.0

        Loss = match * Distance**2 + (1 - match) * F.relu(Distance - self.margin, 0)**2
        return Loss.mean()


class SupCon_Loss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, targets1, targets2):
        out1 = F.normalize(out1, dim=1)
        out2 = F.normalize(out2, dim=1)
        similarity = torch.mm(out1, out2.tranpose(0, 1))
        
        lalbel = targets1[:, None] == targets2[None, :]

        mask = torch.eye(lalbel.shape[0], dtype=torch.bool).to(feature.device)
        lalbel = lalbel[~mask].view(lalbel.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        Distance = torch.sqrt(Distance.sum(dim=-1))
        match = (targets1 == targets2) * 1.0

        Loss = match * Distance**2 + (1 - match) * F.relu(Distance - self.margin, 0)**2
        return Loss.mean()



class simCLR_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feature):
        labels = torch.cat([torch.arange(feature.shape[0] // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(feature.device)

        feature = F.normalize(feature, dim=1)

        similarity_matrix = torch.matmul(feature, feature.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(feature.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1) / 0.07
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feature.device)

        Loss = self.criterion(logits, labels)
        return Loss


