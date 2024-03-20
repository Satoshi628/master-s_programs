#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import re

#----- 専用ライブラリ -----#
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
#----- 自作モジュール -----#
#None

##### IDRID dataset #####
class IDRID_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        if dataset_type == "train":
            image_path = f"{root_dir}/1. Original Images/a. Training Set"
            label_path = f"{root_dir}/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc"
        if dataset_type == "test":
            image_path = f"{root_dir}/1. Original Images/b. Testing Set"
            label_path = f"{root_dir}/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc"
        

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*.jpg"))
        label_list = sorted(glob.glob(label_path + "/*.tif"))

        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB").resize((945,628),Image.BILINEAR) for image_name in image_list]
        self.label_list = [Image.open(label_list).convert("L").resize((945,628),Image.NEAREST) for label_list in label_list]


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

        return image, label/76 #色がついているので対策


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)


##### IDRID dataset #####
class IDRID_multi_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',class_=0, transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        if dataset_type == "train":
            image_path = f"{root_dir}/1. Original Images/a. Training Set"
            if class_ == 0:
                #小動脈瘤
                label = f"{root_dir}/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms"
            if class_ == 1:
                #漏出性出血
                label = f"{root_dir}/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages"
            if class_ == 2:
                #ハードドルーゼン
                label = f"{root_dir}/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates"
            if class_ == 3:
                #ソフトドルーゼン
                label = f"{root_dir}/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates"
        if dataset_type == "test":
            image_path = f"{root_dir}/1. Original Images/b. Testing Set"
            if class_ == 0:
                label = f"{root_dir}/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms"
            if class_ == 1:
                label = f"{root_dir}/2. All Segmentation Groundtruths/b. Testing Set/2. Haemorrhages"
            if class_ == 2:
                label = f"{root_dir}/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates"
            if class_ == 3:
                label = f"{root_dir}/2. All Segmentation Groundtruths/b. Testing Set/4. Soft Exudates"

        

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*.jpg"))
        label = sorted(glob.glob(label + "/*"))

        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB").resize((945,628),Image.BILINEAR) for image_name in image_list]
        self.label_list = [np.array(Image.open(lab).convert("L").resize((945,628),Image.NEAREST)) for lab in label]

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

        return image, label/76


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)



##### Eye dataset #####
class Eye_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, transform=None):
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))

        self.images = [cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in image_paths]
        #self.image_paths = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        self.image_paths = [int(path[-8:-4]) for path in image_paths]

        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        path = self.image_paths[index]

        # もしself.transformが有効なら
        if self.transform:
            image = torch.from_numpy(image) / 255
            image = self.transform(image)
        
        return image, path


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)
