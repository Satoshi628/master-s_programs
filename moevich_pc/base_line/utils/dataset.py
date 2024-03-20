#coding: utf-8
#----- Standard Library -----#
import os
import random
import json

#----- External Library -----#
import pydicom as pyd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

#----- My Module -----#
#None
"""
# Data Lack ,So no Val
train = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,29,
        15,16,17,18,19,20,21,22,23,24,25,26,27,28,36]
test = [30,31,32,33,34,35,
        37,38,39,40,41,42]
"""

# Data Lack ,So no Val,no uint8
train = [1,2,3,4,5,7,9,11,12,29,
        15,16,17,18,19,20,21,22,23,24,25,26,28,36]
test = [30,31,32,33,34,
        37,38,39,40,41,42]

##### PVT dataset #####
class PVT_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        with open(root_dir + "/annotation.json",mode="r") as f:
            data_json = json.load(f)

        if dataset_type == "train":
            #need .astype(np.int16) because exist uint16
            self.image_list = [pyd.dcmread(img_data['file_name']).pixel_array[None].astype(np.int16)
                        for img_data in data_json["images"]
                        if img_data["patient_id"] in train]
            
            self.label_list = [label_data['onset']
                        for label_data in data_json["annotations"]
                        if label_data["patient_id"] in train]
        
        if dataset_type == "test":
            self.image_list = [pyd.dcmread(img_data['file_name']).pixel_array[None].astype(np.int16)
                        for img_data in data_json["images"]
                        if img_data["patient_id"] in test]
            
            self.label_list = [label_data['onset']
                        for label_data in data_json["annotations"]
                        if label_data["patient_id"] in test]
        
        self.label_list = np.array(self.label_list)

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
            image = self.transform(image)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)


class PVT_Loader_plot(data.Dataset):
    def __init__(self,root_dir, transform=None):
        with open(root_dir + "/annotation.json",mode="r") as f:
            data_json = json.load(f)
        
        image_data = data_json["images"]
        anno_data = data_json["annotations"]

        self.image_list = [pyd.dcmread(img_data['file_name']).pixel_array[None].astype(np.int16)
                    for img_data in image_data
                    if img_data["frame_id"] in [0, img_data["seq_length"] // 2, img_data["seq_length"] - 1]]
        
        self.info = [{'onset': label_data['onset'],'frame_id':img_data["frame_id"],'patient_id':label_data['patient_id']}
                    for img_data, label_data in zip(image_data, anno_data)
                    if img_data["frame_id"] in [0, img_data["seq_length"] // 2, img_data["seq_length"] - 1]]

        print(len(self.image_list))
        print(len(self.info))
        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index]
        label = self.info[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.image_list)
