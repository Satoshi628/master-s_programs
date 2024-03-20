#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import json

#----- 専用ライブラリ -----#
from tqdm import tqdm
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

#----- 自作モジュール -----#
from utils.path import CURRENT_PATH

##### covid19 dataset #####
class tanacho_DataLoader(data.Dataset):
    # 初期設定
    def __init__(self, transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = sorted(glob.glob(CURRENT_PATH + "/data/images/**/*.jpg"))

        file_number = [int(path.split("/")[-2]) for path in image_path]

        with open(CURRENT_PATH + "/data/train_meta.json") as f:
            ano_info = json.load(f)
        label = {k: idx for idx, k in enumerate(ano_info.keys())}
        label = [label[str(fn).zfill(3)] for fn in file_number]
        self.label = np.array(label)

        # image_list = 画像のパスのリスト (sorted = 昇順)
        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB").resize([512, 512], Image.LANCZOS) for image_name in image_path]

        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index]
        label = self.label[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def get_class_count(self):
        label = np.array([np.array(label) for label in self.label])
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count
