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
                self.images.append(torch.from_numpy(frame_img).float() / 255)
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



##### Zenigoke dataset #####
class Zenigoke_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/root/data", mode="train", split=0, length=16, delay=2, transform=None):
        # 30FPS => 120FPS case delay=4
        
        split_list = [{"train": [1, 2, 3, 4], "val": [5], "test": [6]},
                      {"train": [6, 1, 2, 3], "val": [4], "test": [5]},
                      {"train": [5, 6, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 5, 6, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 5, 6], "val": [1], "test": [2]}]
        
        self.length = length
        self.delay = delay
        
        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, f"sample{idx}") for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        CP_path = []
        path_length = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(os.path.join(path, "video", "*.png"))))
            path_length.append(len(glob.glob(os.path.join(path, "video", "*.png"))))
            #imgパスからCPパスを取得する
        
        self.images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for paths in img_paths for path in paths]
        
        if mode == "test":
            self.video_file = glob.glob(os.path.join(use_paths[0], "*.avi"))[0]
        
        #EP保存
        EP_images = [np.load(os.path.join(path, "EPmaps.npz"))['arr_0'] for path in use_paths]
        EP_images = np.concatenate(EP_images, axis=0)

        self.EP_images = EP_images

        #Cell Point保存
        CP_data = [torch.from_numpy(np.loadtxt(os.path.join(path, "det.txt"), delimiter=',')).float()
                    for path in use_paths]
        
        
        self.CP_batch_data = [Points[Points[:,0] == idx] for video_len, Points in zip(path_length, CP_data) for idx in range(video_len)]
        #track idの被りを無くす
        #track id の最初は1
        track_id_max = 0
        for idx in range(len(CP_data)):
            CP_data[idx][:, 1] += track_id_max
            track_id_max = CP_data[idx][:, 1].max()


        #削除するindexを求めるlength-1コ消す
        delete_index = [frame for frame in path_length]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(len(self.images)))
        #最後のidxから削除
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        #val,testは同じ画像を入力する必要がない追跡アルゴリズムの場合
        #1iter 0,1,2,3, 2iter 4,5,6,7,...とする
        if mode == "val" or mode == "test":
            self.idx_transfer = [idx for idx in self.idx_transfer if idx%(length) == 0]
        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]
        image = np.stack([self.images[data_index + length] for length in range(self.length) if length %self.delay == 0], axis=1)
        label = np.stack([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = [self.CP_batch_data[data_index + length].clone() for length in range(self.length)]

        image = torch.from_numpy(image).float() / 255.
        label = torch.from_numpy(label) 

        if self.transform:
            image, label, point = self.transform(image, label, point)

        label = label.transpose(0, 1)
        return image, label, point


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_batch_data

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_images


##### Zenigoke dataset #####
class Zenigoke_Loader_Res(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/root/data", split=0):
        
        split_list = [1, 2, 3, 4, 5, 6]
        
        image_paths = sorted(glob.glob(os.path.join(root_dir, f"sample{split_list[split]}", "video", "*.png")))

        self.video_file = glob.glob(os.path.join(root_dir, f"sample{split_list[split]}", "*.avi"))[0]
        
        bar = tqdm(total=len(image_paths))

        self.raw_images = []
        self.images = []
        for path in image_paths:
            img = cv2.imread(path)
            bar.update()
            self.raw_images.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            self.images.append(torch.from_numpy(img).float() / 255)
        bar.close()

        CP_data = np.loadtxt(os.path.join(root_dir, f"sample{split_list[split]}", "det.txt"), delimiter=',')
        self.CP_data = [CP_data[CP_data[:,0] == f] for f in range(len(image_paths))]


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        
        images = self.images[index]
        CP_data = self.CP_data[index]
        
        return images, CP_data


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)

    def get_videos(self):
        return self.raw_images

def collate_delete_PAD(batch):
    image, label, point = list(zip(*batch))

    image = torch.stack(image)
    label = torch.stack(label)

    return image, label, point