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


##### PVT dataset #####
class PVT_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, transform=None):
        with open(root_dir + "/annotation.json",mode="r") as f:
            data_json = json.load(f)
        
        self.image_info = {img_data['id']:{
                            k:v
                            for k,v in img_data.items() if k != 'id'}
                                for img_data in data_json["images"]}

        #need .astype(np.int16) because exist uint16
        self.image_dict = {img_data['id']: pyd.dcmread(img_data['file_name']).pixel_array[None].astype(np.int16)
                    for img_data in data_json["images"]}
        
        self.label_dict = {label_data['image_id']: {'onset': label_data['onset'],'frame_id': img_data['frame_id'], 'patient_id':img_data['patient_id']}
                    for img_data, label_data in zip(data_json["images"], data_json["annotations"])}
        
        self.ids = list(self.image_dict.keys())

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        img_idx = self.ids[index]
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        images = self.image_dict[img_idx]

        label = self.label_dict[img_idx]

        # もしself.transformが有効なら
        if self.transform:
            images = self.transform(images)
        
        return images, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.ids)

