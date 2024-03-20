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

