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
import cv2

#----- 自作モジュール -----#
#None

def get_label(root_dir):
    def call_count(i=[0]): #最初は1,[0]が更新されていく
        i[0] += 1
        return i[0]
    
    #get image class number ex) n07614500, n03662601
    image_folder_list = os.listdir(os.path.join(root_dir, "train"))

    with open(os.path.join(root_dir, "imagenet_class_index.json"),"r") as f:
        annotation_json = json.load(f)
    
    ja_dict = {item["num"]: [call_count()-1, item["ja"]]
                for item in annotation_json
                if item["num"] in image_folder_list}
    
    num_name_list = [v[1] for v in ja_dict.values()]
    return ja_dict, num_name_list

##### Tiny ImageNet dataset #####
class TinyImageNet_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        label_dict, _ = get_label(root_dir)
        
        if dataset_type == "train":
            image_folder_list = sorted(glob.glob(os.path.join(root_dir, "train", "*")))
            image_path_list = [sorted(glob.glob(os.path.join(path, "images", "*.JPEG")))
                                for path in image_folder_list]
            
            # 2dim array => 1dim array
            image_path_list = [path for paths in image_path_list for path in paths]
            label_list = [label_dict[path.split("/")[-3]][0] for path in image_path_list]
        
        if dataset_type == "val":
            image_folder_list = sorted(glob.glob(os.path.join(root_dir, "val", "images", "*")))
            image_path_list = [sorted(glob.glob(os.path.join(path, "*.JPEG")))
                                for path in image_folder_list]
            
            # 2dim array => 1dim array
            image_path_list = [path for paths in image_path_list for path in paths]
            label_list = [label_dict[path.split("/")[-2]][0] for path in image_path_list]

        #image => ndarray,[3(R,G,B),H,W]
        self.image_list = [cv2.imread(path).transpose(2, 0, 1)[[2, 1, 0], :, :]
                            for path in tqdm(image_path_list)]
        self.label_list = np.array(label_list)

        # self.dataset_type = dataset = 'train' or 'val'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index].copy()
        label = self.label_list[index].copy()

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)


##### cityscapse Segmentation dataset #####
class Cityscapse_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = "{}/image/{}".format(root_dir, dataset_type)
        label_path = "{}/gt_fine/{}".format(root_dir, dataset_type)

        image_list = sorted(glob.glob(image_path + "/*"))
        label_list = sorted(glob.glob(label_path + "/*"))

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = [glob.glob(path + "/*.png") for path in image_list]
        label_list = [glob.glob(path + "/*labelIds.png") for path in label_list]

        #2d array => 1d array
        image_list = sorted([path for paths in image_list for path in paths])
        label_list = sorted([path for paths in label_list for path in paths])

        #ignore_label define last class,because one_hot_changer dose not support -1
        ignore_label = 19

        #class id process
        self.label_mapping = {-1: ignore_label,
                            0: ignore_label,
                            1: ignore_label,
                            2: ignore_label,
                            3: ignore_label,
                            4: ignore_label,
                            5: ignore_label,
                            6: ignore_label,
                            7: 0,
                            8: 1,
                            9: ignore_label,
                            10: ignore_label,
                            11: 2,
                            12: 3,
                            13: 4,
                            14: ignore_label,
                            15: ignore_label,
                            16: ignore_label,
                            17: 5,
                            18: ignore_label,
                            19: 6,
                            20: 7,
                            21: 8,
                            22: 9,
                            23: 10,
                            24: 11,
                            25: 12,
                            26: 13,
                            27: 14,
                            28: 15,
                            29: ignore_label,
                            30: ignore_label,
                            31: 16,
                            32: 17,
                            33: 18}


        #画像読み込み
        print("Loading Cityscapse image data")
        # This code is faster than "Image.open(image_name).convert("RGB")"
        self.image_list = [cv2.imread(image_name)
                            for image_name in tqdm(image_list)]

        #transport BGR => RGB
        self.image_list = np.array(self.image_list).transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        

        print("Loading Cityscapse label data")
        self.label_list = [self.convert_label(cv2.imread(label_list, -1))
                        for label_list in tqdm(label_list)]

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

    def convert_label(self, label):
        temp = label.copy()
        for k, v in self.label_mapping.items():
            label[temp == k] = v
        return label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)
