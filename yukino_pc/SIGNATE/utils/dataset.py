#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random

#----- 専用ライブラリ -----#
from tqdm import tqdm
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import cv2
#----- 自作モジュール -----#
#None


##### SIGNATE TECHNOPRO dataset #####
class SIGNATE_TECHNOPRO_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = "{}/{}".format(root_dir, dataset_type)
        image_list = sorted(glob.glob(image_path + "/*.png"))
        self.image_path_list = image_list
        self.image_list = [Image.open(image_name).convert("RGB") for image_name in tqdm(image_list)]

        if dataset_type == "train":
            self.label_list = np.loadtxt(f"{root_dir}/train.csv", delimiter=',', dtype=str)[1:, 1].astype(np.int32)

        # self.dataset_type = dataset = 'train' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        if self.dataset_type == "train":
            label = self.label_list[index]
            return image, label
        else:
            return image, self.image_path_list[index].split("/")[-1]

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def get_class_count(self):
        label = np.array([np.array(label) for label in self.label_list])
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count
