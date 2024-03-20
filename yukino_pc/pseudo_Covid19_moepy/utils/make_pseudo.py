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

#----- 自作モジュール -----#
#None

#疑似ラベルのignore classを定義
IGNORE = 211

class Make_Pseudo():
    def __init__(self,
                root_dir,
                threshold,
                color=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]):
        
        self.__color = torch.tensor(color)
        self.__color = torch.cat([self.__color, torch.full([1, 3], 128)], dim=0)

        self.root_dir = root_dir
        self.threshold = threshold

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(root_dir + "/Label"):
            os.mkdir(root_dir + "/Label")
        if not os.path.exists(root_dir+"/Label_color"):
            os.mkdir(root_dir + "/Label_color")


        self.__number_image = 1
    
    def reset(self):
        self.__number_image = 1

    def update_rood_dir(self, root_dir):
        self.root_dir = root_dir
        if not os.path.exists(root_dir+"/Label"):
            os.mkdir(root_dir + "/Label")
        if not os.path.exists(root_dir+"/Label_color"):
            os.mkdir(root_dir + "/Label_color")

    def __call__(self, outputs):

        """入力された outputs を可視化する関数

        Args:
            outputs (tensor[batch,channel,H,W]): 可視化したい出力
        """        

        outputs = F.softmax(outputs, dim=1)
        value, index = outputs.max(dim=1)
        #valueが閾値以下だったらindexをIGNOREに書き換える
        index[value < self.threshold] = IGNORE

        #学習に使うラベル
        label = index.detach().to('cpu').numpy().astype(np.uint8)
        
        #可視化用画像作成
        index[index == IGNORE] = self.__color.size(0) - 1
        index = index.long()

        #可視化用画像
        color_img = self.__color[index].detach().to('cpu').numpy().astype(np.uint8)

        for lab, color in zip(label, color_img):
            #学習に使うラベル保存
            cv2.imwrite("{}/Label/img_{}.png".format(self.root_dir, self.__number_image), lab)
            #可視化用画像保存
            cv2.imwrite("{}/Label_color/img_{}.png".format(self.root_dir, self.__number_image), color)
            self.__number_image += 1

