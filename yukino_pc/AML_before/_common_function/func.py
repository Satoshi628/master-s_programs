#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as ttf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from _common_function.img_save import print_image
############################

# 二次元cos類似度、組合せ
# 1つの出力に対するフィルタのcos類似度
# portがFalseなら出力に対してのフィルタTrueなら入力に対してのフィルタに
# 独立性を適用する


def cosine_matrix(a, port=False, L=1):
    # x => [out_channel,in_channel,W*H]
    x = a.view(a.shape[0], a.shape[1], -1)
    if port:
        # x => [in_channel,out_channel,W*H]
        x = torch.transpose(x, 0, 1)
    # ベクトルの正規化
    x = x / (x.norm(dim=2, keepdim=True) + 1e-24)
    # x => [out_channel,in_channel,in_channel]
    x = torch.bmm(x, torch.transpose(x, 1, 2))
    #絶対値を取るか二乗を取るかL1 or L2正則化
    if L == 1:
        x = torch.abs(x)
    else:
        x = x ** 2
    #x = torch.abs(x)
    # x => [in_channel,in_channel]
    x = torch.triu(x, diagonal=1)
    x = x.sum(dim=(1, 2)) * 2 / (x.shape[1] * (x.shape[1] - 1))
    return x  # [out_channels]

def cov(tensor, port=False):
    # x => [out_channel,in_channel,W,H]
    x = tensor.view(tensor.shape[0], tensor.shape[1], -1)
    if port:
        # x => [in_channel,out_channel,W*H]
        x = torch.transpose(x, 0, 1)
    # x_はxの平均 x_ => [out_channel,in_channel,1]
    x_ = x.mean(2, keepdim=True)
    # xの平均とyの平均の積 x_y_ => [out_channel,in_channel,in_channel]
    x_y_ = torch.bmm(x_, torch.transpose(x_, 1, 2))
    # xとyの積の平均 x_y_ => [out_channel,in_channel,in_channel]
    xy_ = torch.bmm(x, torch.transpose(x, 1, 2)) / x.shape[2]
    return xy_ - x_y_  # [out_channel,in_channel,in_channel]

def corr(tensor):
    # x => [out_channel,in_channel,W,H]
    x = tensor.view(tensor.shape[0], tensor.shape[1],  -1)
    # x_はxの平均 x_ => [out_channel,in_channel,1]
    x_ = x.mean(2, keepdim=True)
    # xの平均とyの平均の積 x_y_ => [out_channel,in_channel,in_channel]
    x_y_ = torch.bmm(x_, torch.transpose(x_, 1, 2))
    # xとyの積の平均 x_y_ => [out_channel,in_channel,in_channel]
    xy_ = torch.bmm(x, torch.transpose(x, 1, 2)) / x.shape[2]
    cov = xy_ - x_y_
    # st_de:標準偏差[out_channel,in_channel]
    st_de = (torch.diagonal(cov, 0, dim1=1, dim2=2) + 1e-24).sqrt()
    st_de = st_de.view(st_de.shape[0], st_de.shape[1], 1)
    corr = cov / st_de / torch.transpose(st_de, 1, 2)
    return corr  # [out_channel,in_channel,in_channel]

# 1つの出力に対するフィルタの相関係数

def corr_loss(tensor, port=False, L=1):
    if port:
        # tensor => [in_channel,out_channel,W*H]
        tensor = torch.transpose(tensor, 0, 1)
    # corr_loss => [in_channel,out_channel,out_channel]
    corr_loss = corr(tensor)
    ##絶対値を取るか二乗を取るかL1 or L2正則化
    if L == 1:
        corr_loss = torch.abs(corr_loss)
    else:
        corr_loss = corr_loss.pow(2)
    #corr_loss = torch.gt(corr_loss,0.225)*corr_loss
    corr_loss = torch.triu(corr_loss, diagonal=1)
    corr_loss = corr_loss.sum(dim=(1, 2)) * 2 / (corr_loss.shape[1] * (corr_loss.shape[1] - 1))
    return corr_loss  # [out_channels]

class Weight_Normalization(nn.Module):
    def __init__(self,device, model, func='cos', port='input', L=1, alpha=0.001):
        super().__init__()
        self.device = device
        self.model = model
        if port in 'input':
            self.port = True
        elif port in 'output':
            self.port = False
        self.L = L
        self.alpha = alpha
        if func == 'cos':
            self.func = cosine_matrix
        else:
            self.func = corr_loss

    def forward(self):
        # self.model.state_dict()['conv??.weight'] => [out_channel,in_channel,W,H]
        # 全部、値ちがいます
        total = torch.tensor([]).cuda(self.device)
        for name, parameter in self.model.named_parameters():
            #if "down" in name and "weight" in name:
            if "weight" in name:
                if parameter.dim() == 4:
                    # 類似度、相関係数を貯める
                    total = torch.cat([total, self.func(parameter, port=self.port, L=self.L)])
        
        # 類似度、相関係数の平均をとる
        loss = self.alpha * total.mean()
        return loss

"""
for name, parameter in self.model.named_parameters():
        if "down" in name and "weight" in name:
            print(name)
"""

class Custom_MSELoss(nn.Module):
    def __init__(self, device, n_class=4, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)

        self.flag = self.flag.repeat(batch_size, 1, 1, 1)
        self.criterion1 = FocalLoss(gamma=1.0)
        #self.criterion1 = Custom_IoULoss(device, batch_size=8)
        self.criterion2 = My_weight_BCE(device, batch_size=8)

    def forward(self, outputs1, outputs2, targets):
        loss1 = self.criterion1(outputs1, targets)
        loss2 = self.criterion2(outputs2, targets)
        loss = loss1 + 0.01 * loss2
        return loss

class My_weight_MSE(nn.Module):
    def __init__(self, device, n_class=4, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)
        self.flag = self.flag.repeat(batch_size, 1, 1, 1)

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        #outputs = torch.sigmoid(outputs)
        # targets=>[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        #Myclass_weight
        weight = targets.sum(dim=(2, 3), keepdim=True) / targets.sum(dim=(1,2,3), keepdim=True)
        weight = torch.abs(targets - weight)
        weight = weight.clamp(0.01,1.0)
        #MSE
        loss = ((outputs - targets) ** 2) * weight * 4
        loss = loss.mean()
        
        return loss

class My_weight_BCE(nn.Module):
    def __init__(self, device, n_class=4, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)
        self.flag = self.flag.repeat(batch_size, 1, 1, 1)

    def forward(self, outputs, targets):
        #outputs = F.softmax(outputs, dim=1)
        outputs = torch.sigmoid(outputs)
        # targets=>[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        
        #focal_weight(outputsとは無関係の定数とする)定数にしたほうが上がりやした
        #focal_weight = Variable(torch.abs(targets - outputs))
        #focal_weight = Variable(targets + (targets == 0) * outputs)
        #focal_weight.requires_grad = False

        #weight(定数)
        weight = targets.sum(dim=(2, 3), keepdim=True) / targets.sum(dim=(1,2,3), keepdim=True)
        weight = targets + (targets == 0) * weight
        #weight = torch.abs(targets - weight)

        #weight = torch.min(weight, focal_weight)
        weight = torch.clamp(weight, 0., 1.)
        
        #BCE
        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        loss = weight * (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))

        loss = -loss.mean()
        
        return loss

class Funnel_weight_BCE(nn.Module):
    def __init__(self, device, Funnel, n_class=4,alpha=1.0, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)

        self.len = len(Funnel)
        self.flag = self.flag.repeat(batch_size, 1, 1, 1)
        self.funnel = Funnel_ave_pool(Funnel)
        if isinstance(alpha, list):
            self.alpha = self.flag.new_tensor(alpha).view(1,-1,1,1)
        self.alpha = self.flag.new_tensor(alpha)

    def forward(self, outputs, targets):
        #outputs = F.softmax(outputs, dim=1)
        outputs = torch.sigmoid(outputs)

        # targets=[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        
        #focal_weight(outputsとは無関係の定数とする)定数にしたほうが上がりやした
        #focal_weight = Variable(torch.abs(targets - outputs))
        #focal_weight = Variable(targets + (targets == 0) * outputs)
        #focal_weight.requires_grad = False


        weight = targets.sum(dim=(2, 3), keepdim=True) / targets.sum(dim=(1, 2, 3), keepdim=True)
        weight = (targets == 0) * weight
        weight_f = self.funnel(targets)
        weight_f = (targets == 0) * weight_f
        #weight_f = torch.pow(weight_f, self.alpha)
        #weight_f = (1 - torch.exp(-self.alpha * weight_f)) / (1 - torch.exp(-self.alpha))

        #weight_f = torch.sqrt(weight_f)
        #correction = torch.max(weight, weight_f).sum(dim=(2, 3),keepdim=True) - weight.sum(dim=(2, 3),keepdim=True)
        #correction = correction / (weight_f.sum(dim=(2, 3), keepdim=True)+1e-10)
        #weight_f = weight_f * correction

        alpha = (targets.sum(dim=(2, 3), keepdim=True)) ** 2 / (targets.size(2) * targets.size(3))
        weight_f = (1 - torch.exp(-alpha * weight_f)) / (1 - torch.exp(-alpha) + 1e-10)
        #alpha = torch.clamp(1 / (alpha + 1e-10), 1)
        #weight_f = torch.pow(weight_f, alpha)
        
        #BCE
        weight = torch.max(weight, weight_f)
        #weight = weight * (targets.sum(dim=(2, 3), keepdim=True) + 1e-10) / (weight.sum(dim=(2, 3), keepdim=True) + 1e-10)
        weight = targets + weight

        #クラス間のweightの大きさを均一に(意味なさそうじゃね？)
        #beta = weight.sum(dim=(2, 3), keepdim=True)
        #weight = weight * beta.mean(dim=1,keepdim=True)/ (beta + 1e-10)
        #weight = weight + weight_age

        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        loss = (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
        loss = loss * weight
        loss = -loss.mean()

        return loss


class Funnel_weight_BCE2(nn.Module):
    def __init__(self, device, Funnel, n_class=4,alpha=1.0, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)
        self.funnel_number = Funnel
        self.flag = self.flag.repeat(batch_size, 1, 1, 1)
        self.funnel = Funnel_ave_filter(device, Funnel, n_class)
        
        if isinstance(alpha, list):
            self.alpha = self.flag.new_tensor(alpha).view(1,-1,1,1)
        self.alpha = self.flag.new_tensor(alpha)

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)

        # targets=[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        
        weight = targets.sum(dim=(2, 3), keepdim=True) / targets.sum(dim=(1, 2, 3), keepdim=True)

        #weight_F = (targets == 0) * (self.funnel(targets) + weight)
        weight_F = (targets == 0) * (self.funnel(targets) + weight)
        #print_image(weight_F[1,0],file='test0')
        #print_image(weight_F[1,1],file='test1')
        #print_image(weight_F[1,2],file='test2')
        #print_image(weight_F[1, 3], file='test3')

        #[0,1]化
        weight_max = weight_F.view(weight_F.size(0), weight_F.size(1), -1).max(dim=-1, keepdim=True)[0]
        weight_max = weight_max.unsqueeze(-1)
        weight_F = weight_F / (weight_max + 1e-10)


        #ガンマ調整
        #alpha = (targets.sum(dim=(2, 3), keepdim=True)) ** 2 / (targets.size(2) * targets.size(3))
        #beta = 1 / self.funnel_number
        #alpha = 2 / (alpha + beta)
        #weight_F = torch.pow(weight_F, alpha)

        #過渡補正
        #alpha = (targets.sum(dim=(2, 3), keepdim=True)) ** 2 / (targets.size(2) * targets.size(3))
        #weight_F = (1 - torch.exp(-alpha * weight_F)) / (1 - torch.exp(-alpha) + 1e-10)
        
        #weight = torch.clamp(weight, 0., 1.)
        weight = weight_F + targets

        #BCE
        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        loss = (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
        loss = loss * weight
        loss = -loss.mean()

        return loss


class My_IoULoss(nn.Module):
    def __init__(self, device, n_class=4, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)

        self.flag = self.flag.repeat(batch_size, 1, 1, 1)
        self.scale = 8
        self.upsamp = nn.Upsample(scale_factor=self.scale, mode='nearest').cuda(device)

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        #outputs = torch.sigmoid(outputs)
        # targets=>[batch,256,256]
        targets = targets.unsqueeze(1)

        targets = self.flag.eq(targets) * 1.0

        
        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        loss = (intersectoin + 1e-24) / (union + 1e-24)
        loss = 1 - loss.mean()

        outputs_up = self.upsamp(outputs)
        targets_up = self.upsamp(targets)
        outputs_up = torch.stack(torch.split(outputs_up, 256, dim=2))
        targets_up = torch.stack(torch.split(targets_up, 256, dim=2))
        outputs_up = torch.cat(torch.split(outputs_up, 256, dim=4))
        targets_up = torch.cat(torch.split(targets_up, 256, dim=4))

        intersectoin = (outputs_up * targets_up).sum(dim=(3, 4))
        total = (outputs_up + targets_up).sum(dim=(3, 4))
        union = total - intersectoin
        loss_up = (intersectoin + 1e-24) / (union + 1e-24)
        loss_up = 1 - loss_up.mean()

        return (loss + loss_up)*0.5


class ACSL(nn.Module):
    def __init__(self, device, n_class=4, batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)

        self.flag = self.flag.repeat(batch_size, 1, 1, 1)

    def forward(self, outputs, targets):

        outputs = torch.sigmoid(outputs)
        # targets=>[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        #weight
        #論理式では勾配は計算されない(定数化)
        weight = targets + (targets == 0) * (outputs >= 0.7)
        #BCE
        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        loss = weight * (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
        
        loss = -loss.mean()
        
        return loss


class Custom_DiceLoss(nn.Module):
    def __init__(self, device, n_class=4, batch_size=8, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)

        self.flag = self.flag.repeat(batch_size, 1, 1, 1)

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets=>[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        intersection = (outputs * targets).sum()
        loss = 1 - (2 * intersection + self.gamma) / ((outputs ** 2).sum() + (targets ** 2).sum() + self.gamma)
        loss = loss.mean()
        return loss


class Custom_IoULoss(nn.Module):
    def __init__(self,device, n_class=4,batch_size=8):
        super().__init__()
        self.flag = torch.stack(
            [
                torch.full((256, 256), i, dtype=torch.float) for i in range(n_class)
            ]
        ).cuda(device)

        self.flag = self.flag.repeat(batch_size, 1, 1, 1)

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets=>[batch,256,256]
        targets = targets.unsqueeze(1)
        targets = self.flag.eq(targets) * 1.0
        
        intersectoin = (outputs * targets).sum(dim=(2,3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        loss = (intersectoin + 1e-24) / (union + 1e-24)
        loss = loss.mean()
        return 1-loss



def mixup(device, inputs, targets, alpha=1.0):

    ### Shuffle Minibatch ###
    indices = torch.randperm(inputs.size(0)).cuda(device)
    input_s, target_s = inputs[indices], targets[indices]

    lamb = np.random.beta(alpha, alpha)

    inputs = lamb * inputs + (1-lamb) * input_s

    return inputs, target_s, lamb

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
        pt = Variable(logpt.data.exp())
        

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class vote_loss(nn.Module):
    def __init__(self, gamma=0):
        super().__init__()
        self.criterion1 = FocalLoss(gamma=gamma)
        self.criterion2 = FocalLoss(gamma=gamma)
        self.criterion3 = FocalLoss(gamma=gamma)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0)
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, output1, output2, output3, targets):  # output2はsoftmaxをしてください

        loss1 = self.criterion1(output1, targets)
        loss2 = self.criterion2(output2, targets)
        loss3 = 0
        targets = targets.view(
            targets.shape[0], 1, targets.shape[1], targets.shape[2])

        for i in range(4):
            mine = self.conv(targets.eq(i) * 1.0)
            mine = mine.squeeze().long()
            loss3 += self.criterion3(output3[i], mine)

        loss = (loss1 + loss2) / 2 + loss3 / 4
        return loss


class IoU():
    def __init__(self, classification):
        self.__classification = classification
        self.__Intersection = 0
        self.__union = 0
        self.mode_list = ['softmax','sigmoid','linear']
        
    def collect(self, outputs, targets, mode='softmax'):
        assert mode in self.mode_list,"IoUのmodeが例外を発生させています。"
        AssertionError
        if mode == self.mode_list[0]: #softmax
            outputs = F.softmax(outputs, dim=1)
        elif mode == self.mode_list[1]: #sigmoid
            outputs = torch.sigmoid(outputs)
        elif mode == self.mode_list[2]: #linear
            pass


        _, predicted = outputs.max(1)

        self.__Intersection += torch.stack([(predicted.eq(i) & targets.eq(i)).sum() for i in range(self.__classification)])
        self.__union += torch.stack([(predicted.eq(i) | targets.eq(i)).sum() for i in range(self.__classification)])

    def output(self):#IoUとmean_IoUが出力される。
        IoU = (self.__Intersection / (self.__union + 1e-24)).float()
        mean_IoU = IoU.mean()

        self.__Intersection = 0
        self.__union = 0
        return IoU, mean_IoU

class IIC_loss(nn.Module):
    def __init__(self,alpha=1.0):
        super(IIC_loss, self).__init__()
        self.alpha = alpha
        self.EPS = 1e-10


    def compute_joint(self,x_out, x_tf_out):
        # x_out、x_tf_outは torch.Size([512, 10])。この二つをかけ算して同時分布を求める、torch.Size([2048, 10, 10])にする。
        # torch.Size([512, 10, 1]) * torch.Size([512, 1, 10])
        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)
        # p_i_j は　torch.Size([512, 10, 10])

        # 全ミニバッチを足し算する ⇒ torch.Size([10, 10])
        p_i_j = p_i_j.sum(dim=0)

        # 転置行列と足し算して割り算（対称化） ⇒ torch.Size([10, 10])
        p_i_j = (p_i_j + p_i_j.t()) / 2.

        # 規格化 ⇒ torch.Size([10, 10])
        p_i_j = p_i_j / p_i_j.sum()

        return p_i_j
        # 結局、p_i_jは通常画像の判定出力10種類と、変換画像の判定10種類の100パターンに対して、全ミニバッチが100パターンのどれだったのかの確率分布表を示す

    def forward(self, x_out, x_tf_out):
        # torch.Size([512, 10])、後ろの10は分類数なので、overclusteringのときは100
        bs, k = x_out.size()
        p_i_j = self.compute_joint(x_out, x_tf_out)  # torch.Size([10, 10])

        # 同時確率の分布表から、変換画像の10パターンをsumをして周辺化し、元画像だけの周辺確率の分布表を作る
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        # 同時確率の分布表から、元画像の10パターンをsumをして周辺化し、変換画像だけの周辺確率の分布表を作る
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        # 0に近い値をlogに入れると発散するので、避ける
        p_i_j = torch.where(p_i_j < self.EPS, torch.tensor(
            [self.EPS], device=p_i_j.device), p_i_j)
        p_j = torch.where(p_j < self.EPS, torch.tensor([self.EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < self.EPS, torch.tensor([self.EPS], device=p_i.device), p_i)

        # 元画像、変換画像の同時確率と周辺確率から、相互情報量を計算
        # ただし、マイナスをかけて最小化問題にする
        """
        相互情報量を最大化したい
        ⇒結局、x_out, x_tf_outが持ちあう情報量が多くなって欲しい
        ⇒要は、x_out, x_tf_outが一緒になって欲しい

        p_i_jはx_out, x_tf_outの同時確率分布で、ミニバッチが極力、10×10のいろんなパターン、満遍なく一様が嬉しい

        前半の項、torch.log(p_i_j)はp_ijがどれも1に近いと大きな値（0に近い）になる。
        どれかが1であと0でバラついていないと、log0で小さな値（負の大きな値）になる
        つまり前半の項は、

        後半の項は、元画像、もしくは変換画像について、それぞれ周辺化して10通りのどれになるかを計算した項。
        周辺化した10×10のパターンを引き算して、前半の項が小さくなるのであれば、
        x_outとx_tf_outはあまり情報を共有していなかったことになる。
        """
        # https://qiita.com/Amanokawa/items/0aa24bc396dd88fb7d2a
        # を参考に、重みalphaを追加
        # 同時確率分布表のばらつきによる罰則を小さく ＝ 同時確率の分布がバラつきやすくする
        alpha = 2.0  # 論文や通常の相互情報量の計算はalphaは1です

        loss = -1*(p_i_j * (torch.log(p_i_j) - alpha *
                            torch.log(p_j) - alpha*torch.log(p_i))).mean()

        return loss

#Funnel_weight用平滑化フィルタ
class Funnel_ave_pool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.level = len(kernel_size)
        self.Pyramid_pool = nn.ModuleList([
            nn.AvgPool2d(kernel_size[i],stride=1,padding=(kernel_size[i]-1)//2,count_include_pad=False) for i in range(self.level)
        ])

    def forward(self, x):  # x => [batch,channel,W,H]
        h = torch.stack([self.Pyramid_pool[i](x) for i in range(self.level)])
        h,_ = h.max(dim=0)
        return h

#Funnel_weight用加算平滑化フィルタ
class Funnel_ave_filter(nn.Module):
    def __init__(self, device, kernel_number,n_class):
        super().__init__()
        self.n_class = n_class
        self.kernel_number = kernel_number

        def def_kernel():  #関数内関数は外側関数の変数を参照できる。(書き換えは不可能らしい)
            if isinstance(kernel_number, list):
                assert len(kernel_number) == n_class,"Funnel_ave_filterエラー:カーネルの数とクラスの数が違います"
                self.kernel_number = max(kernel_number)
                #2次元tensor[n_class,4]
                pad_size = [
                    [[kernel_number[j] - 1 - i, kernel_number[j] - 1 - i, kernel_number[j] - 1 - i, kernel_number[j] - 1 - i] for i in range(1, kernel_number[j])]
                    for j in range (self.n_class)
                ]
                #4次元tensor[1,1,?,?]の配列
                kernel = [
                    torch.stack([F.pad(torch.full((1, 1, 2 * i + 1, 2 * i + 1), 1 / (2 * i + 1) ** 2, dtype=torch.float), pad_size[j][i - 1]) for i in range(1, kernel_number[j])])
                    for j in range(self.n_class)
                ]
                kernel = [kernel[j].sum(dim=0) for j in range(self.n_class)]
                kernel = torch.cat([F.pad(kernel[j],
                                          [(2 * self.kernel_number - 1 - kernel[j].size(-1))//2, (2 * self.kernel_number - 1 - kernel[j].size(-1))//2,(2 * self.kernel_number - 1 - kernel[j].size(-1))//2,(2 * self.kernel_number - 1 - kernel[j].size(-1))//2])
                                for j in range(self.n_class)], dim=0)
            else:
                pad_size = [[kernel_number - 1 - i, kernel_number - 1 - i, kernel_number - 1 - i, kernel_number - 1 - i] for i in range(1,kernel_number)]
                kernel = torch.stack([F.pad(torch.full((n_class, 1, 2*i+1, 2*i+1), 1 / (2*i+1) ** 2, dtype=torch.float),pad_size[i-1]) for i in range(1,kernel_number)])
                kernel = kernel.sum(dim=0)
            return kernel
        
        self.kernel = def_kernel().cuda(device)
            
    def forward(self, x):  # x => [batch,channel,W,H]
        h = F.conv2d(x, self.kernel, padding=self.kernel_number - 1, groups=self.n_class)
        return h


class one_hot_vector():
    def __init__(self, device="cuda:0", classification=4):
        self.flag = torch.stack(
            [
                torch.full((1, 1, 1), i, dtype=torch.float) for i in range(classification)
            ], dim=1).to(device)
        self.device = device
        self.classification = classification

    def change(self, targets):
        targets = targets.unsqueeze(1)
        targets = self.flag.cuda(targets.device).eq(targets) * 1.0
        return targets
