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
import cv2
#----- 自作モジュール -----#


##### MVtecAD dataset #####
class MVtecAD_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, product, dataset_type='train', transform=None):
        self.product = product
        
        # image load
        image_path = os.path.join(root_dir,product, dataset_type,"*","*.png")
        image_path_list = sorted(glob.glob(image_path))

        class_names = set([path.split("/")[-2] for path in image_path_list])
        n = 1
        class_dict = {"good": 0}
        for key in class_names:
            if key != "good":
                class_dict[key] = n
                n += 1
        self.class_dict = class_dict
        print(class_dict)

        self.image_list = image_path_list

        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image_path = self.image_list[index]
        image = self.imload(image_path)

        label = self.class_dict[image_path.split("/")[-2]]

        if image_path.split("/")[-2] == "good":
            H, W = image.shape[1:]
            mask = np.zeros([1, H, W], dtype=np.float32)
        else:
            mask = cv2.imread(image_path.replace('test','ground_truth').replace('.png','_mask.png'), cv2.IMREAD_GRAYSCALE)[None]

        # もしself.transformが有効なら
        if self.transform:
            image, mask = self.transform(image, mask)
        mask = (mask/255).long()
        return image, mask, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def imload(self, file_name):
        try:
            img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)/255
        except:
            assert False, file_name
        img = img.transpose(2,0,1).astype(np.float32)
        return img


if __name__ == "__main__":
    dataset = MVtecAD_Loader("/mnt/kamiya/dataset/mvtec", "capsule", "test")