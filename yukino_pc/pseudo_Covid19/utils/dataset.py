#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random

#----- 専用ライブラリ -----#
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
#----- 自作モジュール -----#
#None

##### covid19 dataset #####
class Covid19_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', pseudo_path=None, transform=None):
        # pseudo_path がないならtrainフォルダの物のみ使う

        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = "{}/covid19/Image/{}".format(root_dir, dataset_type)
        label_path = "{}/covid19/Label/{}".format(root_dir, dataset_type)

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        # pseudo_image_list = 疑似ラベルの画像のパスのリスト (sorted = 昇順)
        # pseudo_label_list = 疑似ラベルのラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*.png"))
        label_list = sorted(glob.glob(label_path + "/*.png"))

        #trainの時のみ疑似ラベルを使うため
        if pseudo_path is not None and dataset_type=='train':
            #疑似ラベル専用の画像と疑似ラベルのパスを取得
            pseudo_image_list = sorted(glob.glob(f"{root_dir}/covid19/Image/pseudo/*png"))
            pseudo_label_list = sorted(glob.glob(os.path.join(pseudo_path, "Label", "*png")))

            #リストに追加
            image_list.extend(pseudo_image_list)
            label_list.extend(pseudo_label_list)


        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB") for image_name in image_list]
        self.label_list = [Image.open(label_list).convert("L") for label_list in label_list]

        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index]
        label = self.label_list[index]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

