#coding: utf-8
#----- Standard Library -----#
import re
import os
import random
import math

#----- Public Package -----#
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import cv2

#----- Module -----#
#None

PAD_ID = -1

##### OIST dataset #####
class OIST_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/NC1032", mode="train", length=16, delay=2, transform=None):
        
        self.length = length
        self.delay = delay

        image_paths = sorted(glob.glob(os.path.join(root_dir, "imgs", "*")))
        path_length = [len(image_paths)]

        self.images = [(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float32)
                        for path in image_paths]
        
        #EP保存
        self.EP_maps = np.load(os.path.join(root_dir, "gt_maps.npz"))['arr_0'][:, 0]

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
            self.idx_transfer = [idx for idx in self.idx_transfer if idx % (length - 1) == 0]
        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        image = np.stack([self.images[data_index + length] for length in range(self.length) if length % self.delay == 0], axis=0)
        label = np.stack([self.EP_maps[data_index + length] for length in range(self.length)], axis=0)

        if self.transform:
            image, label = self.transform(image[None], label[None])

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_maps


##### OIST dataset #####
class OIST_Loader_Test(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/NC1032", length=16, transform=None):
        self.length = length

        image_paths = sorted(glob.glob(os.path.join(root_dir, "imgs", "*")))
        path_length = [len(image_paths)]

        self.images = [(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float32)
                        for path in image_paths]
        
        #EP保存
        self.EP_maps = np.load(os.path.join(root_dir, "gt_maps.npz"))['arr_0'][:, 0]


        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(0, len(self.images), length))

        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        image = []
        label = []
        for length in range(self.length):
            idx = data_index + length
            idx = min(idx, len(self.images) - 1)

            image.append(self.images[idx])
            label.append(self.EP_maps[idx])

        image = np.stack(image)
        label = np.stack(label)
        if self.transform:
            image, label = self.transform(image[None], label[None])

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_maps

    def get_img_size(self):
        return self.images[0].shape



def collate_delete_PAD(batch):
    image, label, point = list(zip(*batch))

    image = torch.stack(image)
    label = torch.stack(label)
    point = torch.stack(point)

    # Make PAD as low as possible.
    sorted_idx = torch.argsort(point[:, :, :, 0], dim=2, descending=True)
    point = torch.take_along_dim(point, sorted_idx[:, :, :, None], dim=2)
    max_point_num = (point[:, :, :, 0] != -1).sum(dim=-1).max()
    point = point[:, :, :max_point_num]

    return image, label, point

if __name__ == "__main__":
    import transforms as tf
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    train_dataset = OIST_Loader(Staining="GFP-Mid", mode="train", split=0, length=16, delay=4, transform=train_transform)
    max_move = train_dataset.get_max_move_speed()
    input(max_move)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=8,
        shuffle=True,
        #num_workers=2,
        drop_last=True)
    for batch_idx, (inputs, targets, point) in enumerate(train_loader):
        print(inputs.shape)
        print(targets.shape)
        print(point.shape)
        input()