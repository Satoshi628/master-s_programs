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

# Data Lack ,So no Val
train = [1,2,3,4,5,7,9,11,12,29,
        15,16,17,18,19,20,21,22,23,24,
        25,26,28,36]
test = [30,31,32,33,34,
        37,38,39,40,41,
        42]


##### PVT dataset #####
class PVT_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, length=16, dataset_type='train', transform=None):
        self.length = length
        with open(root_dir + "/annotation.json",mode="r") as f:
            data_json = json.load(f)
        
        self.image_info = {img_data['id']:{
                            k:v
                            for k,v in img_data.items() if k != 'id'}
                                for img_data in data_json["images"]}

        if dataset_type == "train":
            #need .astype(np.int16) because exist uint16
            self.image_dict = {img_data['id']: pyd.dcmread(img_data['file_name']).pixel_array[None]
                        for img_data in data_json["images"]
                        if img_data["patient_id"] in train}
            
            self.label_dict = {label_data['image_id']: label_data['onset']
                        for label_data in data_json["annotations"]
                        if label_data["patient_id"] in train}
        
        if dataset_type == "test":
            self.image_dict = {img_data['id']: pyd.dcmread(img_data['file_name']).pixel_array[None]
                        for img_data in data_json["images"]
                        if img_data["patient_id"] in test}
            
            self.label_dict = {label_data['image_id']: label_data['onset']
                        for label_data in data_json["annotations"]
                        if label_data["patient_id"] in test}
        
        # fix
        self.image_dict = {k:2 * (v.astype(np.int16) - 1024) if v.dtype == np.uint16 else v  for k,v in self.image_dict.items()}

        self.ids = list(self.image_dict.keys())
        #delete straddling different data 
        self.ids = [idx for idx in self.ids[:-length+1]
                    if self.image_info[idx]['patient_id'] == self.image_info[idx + length - 1]['patient_id']]
        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        img_idx = self.ids[index]
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        images = [self.image_dict[img_idx + frame] for frame in range(self.length)]
        images = np.array(images)

        label = self.label_dict[img_idx]

        # もしself.transformが有効なら
        if self.transform:
            images = self.transform(images)
        
        #image.size()=>[C,T,H,W]
        images = images.transpose(0, 1)
        return images, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.ids)



##### PVT dataset #####
class PVT_Loader_plot(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, length=16, dataset_type='train', transform=None):
        self.length = length
        with open(root_dir + "/annotation.json",mode="r") as f:
            data_json = json.load(f)
        
        self.image_info = {img_data['id']:{
                            k:v
                            for k,v in img_data.items() if k != 'id'}
                                for img_data in data_json["images"]}

        if dataset_type == "train":
            #need .astype(np.int16) because exist uint16
            self.image_dict = {img_data['id']: pyd.dcmread(img_data['file_name']).pixel_array[None].astype(np.int16)
                        for img_data in data_json["images"]
                        if img_data["patient_id"] in train}
            
            self.label_dict = {label_data['image_id']: {'onset': label_data['onset'],'frame_id': img_data['frame_id'], 'patient_id':img_data['patient_id']}
                        for img_data, label_data in zip(data_json["images"], data_json["annotations"])
                        if label_data["patient_id"] in train}
        
        if dataset_type == "test":
            self.image_dict = {img_data['id']: pyd.dcmread(img_data['file_name']).pixel_array[None].astype(np.int16)
                        for img_data in data_json["images"]
                        if img_data["patient_id"] in test}
            
            self.label_dict = {label_data['image_id']: {'onset': label_data['onset'],'frame_id': img_data['frame_id'], 'patient_id':img_data['patient_id']}
                        for img_data, label_data in zip(data_json["images"], data_json["annotations"])
                        if label_data["patient_id"] in test}
        
        self.ids = list(self.image_dict.keys())
        #delete straddling different data 
        self.ids = [idx for idx in self.ids[:-length+1]
                    if self.image_info[idx]['patient_id'] == self.image_info[idx + length - 1]['patient_id']]
        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        img_idx = self.ids[index]
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        images = [self.image_dict[img_idx + frame] for frame in range(self.length)]
        images = np.array(images)

        label = self.label_dict[img_idx]

        # もしself.transformが有効なら
        if self.transform:
            images = self.transform(images)
        
        #image.size()=>[C,T,H,W]
        images = images.transpose(0, 1)
        return images, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.ids)

