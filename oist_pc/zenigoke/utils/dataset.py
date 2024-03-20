#coding: utf-8
#----- Standard Library -----#
import re
import os
import sys
import random
import glob
import math

#----- Public Package -----#
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import Normalize
from torchvision.transforms.functional import rotate
from PIL import Image
import cv2

#----- Module -----#
#None


##### Zenigoke dataset #####
class Zenigoke_Loader_Exp(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/Experimental_data", video_name="CD47-GFP_01.avi"):
        norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        cap = cv2.VideoCapture(os.path.join(root_dir, video_name))
        
        if not cap.isOpened():
            # 正常に読み込めたのかチェックする
            # 読み込めたらTrue、失敗ならFalse
            print("動画の読み込み失敗")
            sys.exit()
        
        bar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.raw_images = []
        self.images = []
        while True:
            # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
            is_image, frame_img = cap.read()
            bar.update()
            if is_image:
                self.raw_images.append(frame_img)
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                self.images.append(norm(torch.from_numpy(frame_img).float() / 255))
            else:
                # フレーム画像が読込なかったら終了
                break
        bar.close()

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        
        images = self.images[index]
        
        return images


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)

    def get_videos(self):
        return self.raw_images



##### Zenigoke dataset #####
class Zenigoke_Loader_Exp_Res(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/Experimental_data", video_name="CD47-GFP_01.avi"):
        norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        image_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/code/my_exp/zenigoke/video_images/200107_Mppicalm-k-2#18x20_001", "*")))
        
        bar = tqdm(total=len(image_paths))

        self.raw_images = []
        self.images = []
        for path in image_paths:
            img = cv2.imread(path)
            bar.update()
            self.raw_images.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            self.images.append(norm(torch.from_numpy(img).float() / 255))
        bar.close()

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        
        images = self.images[index]
        
        return images


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)

    def get_videos(self):
        return self.raw_images





##### Prompt dataset #####
class Prompt_Loader(data.Dataset):
    # 初期設定
    def __init__(self, prompt_root_dir="/mnt/kamiya/dataset/OIST/Experimental_data"):
        # 30FPS => 120FPS case delay=4
        self.norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.rotate_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
        
        self.images_path = glob.glob(os.path.join(prompt_root_dir, "*"))

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        img = cv2.imread(self.images_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img = self.norm(torch.from_numpy(img).float() / 255)
        images = []
        for deg in self.rotate_degrees:
            images.append(rotate(img, deg))
        images = torch.stack(images)
        return images


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images_path)

