#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random

#----- 専用ライブラリ -----#
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
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


def feature_Segmentation(feature_probability_map, file_name):
    """サンプリングされた特徴がどの場所を担当しているかの可視化

    Args:
        feature_probability_map (np.ndarray([sample_num, H, W])): サンプリングした特徴の分布
    """    
    feature_probability_map = torch.from_numpy(feature_probability_map)
    color = [[random.randint(0,255) for _ in range(3)] for _ in range(feature_probability_map.shape[0])]
    color = torch.tensor(color,dtype=torch.long)

    class_image = feature_probability_map.max(dim=0)[1]
    #seg_image.size() => [batch,H,W,3]
    seg_image = color[class_image][..., [2, 1, 0]]
    #[batch,H,W,3(RGB) or 1]にしてnumpy,unsigned char化
    seg_image = seg_image.detach().numpy().astype(np.uint8)
    
    cv2.imwrite(f"{file_name}.png", seg_image)
