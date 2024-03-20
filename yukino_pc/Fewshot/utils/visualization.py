#coding: utf-8
#----- 標準ライブラリ -----#
import os

#----- 専用ライブラリ -----#
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tsne_torch import TorchTSNE as TSNE

#----- 自作モジュール -----#
#None


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



class Seg_image():
    def __init__(self,
                color,
                root_dir="images",
                input_norm_mean=[0.485, 0.456, 0.406],
                input_norm_std=[0.229, 0.224, 0.225]):
        
        self.__color = torch.tensor(color)
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        self.input_norm_mean = torch.tensor(input_norm_mean)
        self.input_norm_std = torch.tensor(input_norm_std)

        self.__number_image = {"input": 0, "label": 0, "output": 0}
    
    def _reset(self):
        self.__number_image = {"input": 0, "label": 0, "output": 0}

    def save_fig(self, tensor, mode="input"):
        """入力されたtensorを可視化する関数

        Args:
            tensor (tensor[batch,channel,H,W]): 可視化したい画像,入力画像、ラベル、予測画像を入力できる

            mode (str, optional): "input","label","output"かを指定する。label,outputはsegmentation結果を出力する. Defaults to "input".
        """        

        if not mode in ["input", "label", "output"]:
            raise ValueError("modeにinput,label,output以外の値が入力されました")
        
        if mode == "input":
            if tensor.dim() != 4:
                raise ValueError("inputモードですが入力が4次元ではありません")
            #float,cpuモードに変更
            tensor = tensor.float().to("cpu")

            #channelが3(RGB)なら正規化を元に戻す
            if tensor.size(1) == 3:
                tensor = self.input_norm_std[None, :, None, None] * tensor + self.input_norm_mean[None, :, None, None]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            tensor = (tensor * 255).permute(0, 2, 3, 1).detach().numpy().astype(np.uint8)
            
            for image in tensor:
                self.plot_image(image, mode)
        
        if mode == "label":
            if tensor.dim() != 3:
                raise ValueError("labelモードですが入力が3次元ではありません")
            #long,cpuモードに変更
            tensor = tensor.long().to("cpu")
            
            #seg_image.size() => [batch,H,W,3]
            seg_image = self.__color[tensor]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            seg_image = seg_image.detach().numpy().astype(np.uint8)
            
            for image in seg_image:
                self.plot_image(image, mode)
        
        if mode == "output":
            if tensor.dim() != 4:
                raise ValueError("outputモードですが入力が4次元ではありません")
            #float,cpuモードに変更
            tensor = tensor.float().to("cpu")

            class_image = tensor.max(dim=1)[1]
            
            #seg_image.size() => [batch,H,W,3]
            seg_image = self.__color[class_image]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            seg_image = seg_image.detach().numpy().astype(np.uint8)
            
            for image in seg_image:
                self.plot_image(image, mode)
    
    def plot_image(self, image, mode):
        cv2.imwrite("{}/{}{:04}.png".format(self.root_dir, mode, self.__number_image[mode]), image)

        self.__number_image[mode] += 1



class Edge_tSNE():
    def __init__(self,
                color,
                n_class,
                root_dir="images"):
        
        self.__color = torch.tensor(color)
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        self.n_class = n_class

        
        self.avgPool = nn.AvgPool2d(2 * 7 + 1, stride=1, padding=7)
        self.downsamp = nn.AvgPool2d(4)


    def __call__(self, feature, targets):
        onehot = one_hot_changer(targets, self.n_class, dim=1)

        smooth_map = self.avgPool(onehot)
        # 物体の曲線付近内側までを1.とするフラグを作成
        weight = onehot * smooth_map

        seg_image = self.__color[targets].cuda(targets.device).permute(0, 3, 1, 2)
        weight = self.downsamp(weight)
        weight = weight.sum(dim=1)

        seg_image = self.downsamp(seg_image.float()).long()
        feature = self.downsamp(feature)
        feature = feature.permute(0, 2, 3, 1).flatten(0, 2)
        weighted_color = (seg_image * weight.unsqueeze(1)).permute(0, 2, 3, 1).flatten(0, 2).cpu().detach().numpy()

        weighted_color = (weighted_color/255).tolist()
        

        emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(feature,device="cuda:0")

        plt.clf()
        for color, coord in zip(weighted_color, emb):
            plt.plot(coord[0], coord[1], marker=".", color=color)

        plt.savefig(f"edge_feature_Normal.jpg")
    
    def one_class(self, feature, targets):
        class_ids, counts = torch.unique(targets, return_counts=True)
        class_ids = class_ids[class_ids != 0]

        feature = feature.permute(0, 2, 3, 1)
        for class_id in class_ids:

            flag_map = targets == class_id
            
            onehot = one_hot_changer(targets, self.n_class, dim=1)

            smooth_map = self.avgPool(onehot)
            # 物体の曲線付近内側までを1.とするフラグを作成
            weight = onehot * smooth_map

            seg_image = self.__color[targets].cuda(targets.device).permute(0, 3, 1, 2)
            weight = weight.sum(dim=1)
            seg_image = seg_image.permute(0, 2, 3, 1)
            
            class_feature = feature[flag_map]
            weighted_color = (seg_image * weight.unsqueeze(-1))[flag_map].cpu().detach().numpy()
            weighted_color = (weighted_color/255).tolist()
            
            class_feature = class_feature.to("cpu")
            emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(class_feature,device="cpu")

            plt.clf()
            for color, coord in zip(weighted_color, emb):
                plt.plot(coord[0], coord[1], marker=".", color=color)

            plt.savefig(f"edge_feature_Normal_oneclass{class_id}.jpg")



class Edge_Seg_image():
    def __init__(self,
                color,
                n_class,
                root_dir="images",
                input_norm_mean=[0.485, 0.456, 0.406],
                input_norm_std=[0.229, 0.224, 0.225]):
        
        self.avgPool = nn.AvgPool2d(2 * 7 + 1, stride=1, padding=7)

        self.__color = torch.tensor(color)
        self.n_class = n_class
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        self.input_norm_mean = torch.tensor(input_norm_mean)
        self.input_norm_std = torch.tensor(input_norm_std)

        self.__number_image = {"input": 0, "label": 0, "output": 0}
    
    def _reset(self):
        self.__number_image = {"input": 0, "label": 0, "output": 0}

    def save_fig(self, tensor, mode="input"):
        """入力されたtensorを可視化する関数

        Args:
            tensor (tensor[batch,channel,H,W]): 可視化したい画像,入力画像、ラベル、予測画像を入力できる

            mode (str, optional): "input","label","output"かを指定する。label,outputはsegmentation結果を出力する. Defaults to "input".
        """        

        if not mode in ["input", "label", "output"]:
            raise ValueError("modeにinput,label,output以外の値が入力されました")
        
        if mode == "input":
            if tensor.dim() != 4:
                raise ValueError("inputモードですが入力が4次元ではありません")
            #float,cpuモードに変更
            tensor = tensor.float().to("cpu")

            #channelが3(RGB)なら正規化を元に戻す
            if tensor.size(1) == 3:
                tensor = self.input_norm_std[None, :, None, None] * tensor + self.input_norm_mean[None, :, None, None]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            tensor = (tensor * 255).permute(0, 2, 3, 1).detach().numpy().astype(np.uint8)
            
            for image in tensor:
                self.plot_image(image, mode)
        
        if mode == "label":
            if tensor.dim() != 3:
                raise ValueError("labelモードですが入力が3次元ではありません")
            #long,cpuモードに変更
            onehot = one_hot_changer(tensor.long(), self.n_class, dim=1)
            smooth_map = self.avgPool(onehot)
            # 物体の曲線付近内側までを1.とするフラグを作成
            weight = onehot * smooth_map
            weight = weight.sum(dim=1).to("cpu")
            tensor = tensor.long().to("cpu")

            #seg_image.size() => [batch,H,W,3]
            seg_image = self.__color[tensor]
            seg_image = (seg_image * weight.unsqueeze(-1))
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            seg_image = seg_image.detach().numpy().astype(np.uint8)
            
            for image in seg_image:
                self.plot_image(image, mode)
        
        if mode == "output":
            if tensor.dim() != 4:
                raise ValueError("outputモードですが入力が4次元ではありません")
            #float,cpuモードに変更
            tensor = tensor.float().to("cpu")

            class_image = tensor.max(dim=1)[1]
            
            #seg_image.size() => [batch,H,W,3]
            seg_image = self.__color[class_image]
            #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
            seg_image = seg_image.detach().numpy().astype(np.uint8)
            
            for image in seg_image:
                self.plot_image(image, mode)
    
    def plot_image(self, image, mode):
        cv2.imwrite("{}/{}{:04}.png".format(self.root_dir, mode, self.__number_image[mode]), image)

        self.__number_image[mode] += 1



"""供養,cv2のほうが書き込みが早く、余白もできないためmatplotを使わなくなった

    def plot_image(self, image, mode):
        plt.clf()
        if image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

        #メモリなし
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        #保存
        plt.savefig("{}/{}{:04}.png".format(self.root_dir, mode, self.__number_image[mode]),
                    bbox_inches='tight',
                    pad_inches=0)
        
        self.__number_image[mode] += 1
"""