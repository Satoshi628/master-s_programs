#coding: utf-8
#----- Standard Library -----#
import os
import random
import json
import re
#----- External Library -----#
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import (Compose,
    Normalize,
    Resize,
    RandomCrop,
    CenterCrop,
    ColorJitter,
    Pad,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomPerspective,
    RandomRotation)
#----- My Module -----#
from utils.utils import GaussianBlur, ndarrayToTensor
# from utils import GaussianBlur, ndarrayToTensor

#男性:1 女性:0

##### Eye dataset #####
class Eye_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        mask_paths = sorted(glob.glob(os.path.join(root_dir,"mask","*.png")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = 2*label[:,0,0] + label[:,0,1]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        self.labels = np.array([label[idx]  for idx in split_idx])
        if use_mask:
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_mask = use_mask

        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.labels[index]

        if self.use_mask:
            mask = self.mask[index]
            if random.random() > 0.5:
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_2classification2(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', split=1, transform=None, use_age=False, use_sex=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        train_paths = sorted(glob.glob(os.path.join(root_dir,"train_images","*")))
        test_paths = sorted(glob.glob(os.path.join(root_dir,"test_images","*.TIF")))
        image_paths = train_paths + test_paths

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        
        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx,5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        self.labels = np.array([label[idx]  for idx in split_idx])[:, 0]
        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex

        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.labels[index]
        

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)
        
        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_ARMS2(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        mask_paths = sorted(glob.glob(os.path.join(root_dir,"mask","*.png")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = label[:,0,0]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        self.labels = np.array([label[idx]  for idx in split_idx])
        if use_mask:
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_mask = use_mask

        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.labels[index]

        if self.use_mask:
            mask = self.mask[index]
            if random.random() > 0.5:
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT200","*.tif")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)

        OCT_num = [int(re.search(r"\d+.tif",path).group()[:-4]) for path in OCT_paths]
        OCT_num = np.array(OCT_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = label[:,0,0]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        
        self.OCT = []
        for idx in split_idx:
            if len(OCT_paths) > idx:
                try:
                    self.OCT.append(cv2.cvtColor(cv2.imread(OCT_paths[idx]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                except:
                    self.OCT.append(np.zeros([3, 945, 945]))
                    print(OCT_paths[idx])
            else:
                self.OCT.append(np.zeros([3, 945, 945]))

        self.labels = np.array([label[idx]  for idx in split_idx])
        if use_mask:
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_mask = use_mask

        self.transform = transform
        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                RandomCrop(size=(800, 800)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                CenterCrop(size=(800, 800)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT = self.OCT[index]
        label = self.labels[index]

        if self.use_mask:
            mask = self.mask[index]
            if random.random() > 0.5:
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, OCT, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT_Test(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        OCT_paths = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT200","*.tif")))


        OCT_num = [int(re.search(r"\d+.tif",path).group()[:-4]) for path in OCT_paths]
        OCT_num = np.array(OCT_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in OCT_num]
        label = np.array(label)
        # bainary => Decimal
        label = label[:,0,0]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in OCT_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(OCT_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.OCT = []
        for idx in split_idx:
            if len(OCT_paths) > idx:
                try:
                    self.OCT.append(cv2.cvtColor(cv2.imread(OCT_paths[idx]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                except:
                    self.OCT.append(np.zeros([3, 945, 945]))
                    print(OCT_paths[idx])
            else:
                self.OCT.append(np.zeros([3, 945, 945]))

        self.labels = np.array([label[idx]  for idx in split_idx])

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                RandomCrop(size=(800, 800)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                CenterCrop(size=(800, 800)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        OCT = self.OCT[index]
        label = self.labels[index]

        OCT = self.OTC_transform(OCT)
        
        return OCT, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.OCT)



##### Eye dataset #####
class Eye_Loader_OCT_VIT(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT200","*.tif")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)

        OCT_num = [int(re.search(r"\d+.tif",path).group()[:-4]) for path in OCT_paths]
        OCT_num = np.array(OCT_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = label[:,0,0]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        
        self.OCT = []
        for idx in split_idx:
            if len(OCT_paths) > idx:
                try:
                    self.OCT.append(cv2.cvtColor(cv2.imread(OCT_paths[idx]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                except:
                    self.OCT.append(np.zeros([3, 945, 945]))
                    print(OCT_paths[idx])
            else:
                self.OCT.append(np.zeros([3, 945, 945]))

        self.labels = np.array([label[idx]  for idx in split_idx])
        if use_mask:
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_mask = use_mask

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT = self.OCT[index]
        label = self.labels[index]

        if self.use_mask:
            mask = self.mask[index]
            if random.random() > 0.5:
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, OCT, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT_VIT_ALL(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT200","*.tif")))
        OCT_paths2 = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT400","*.tif")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)

        OCT_num = [int(re.search(r"/OCT\d+",path).group()[4:]) for path in OCT_paths]
        OCT_num = np.array(OCT_num)

        OCT_num2 = [int(re.search(r"/OCT\d+",path).group()[4:]) for path in OCT_paths2]
        OCT_num2 = np.array(OCT_num2)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = label[:,0,0]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        
        self.OCT = [[] for _ in split_idx]
        for idx in split_idx:
            if len(OCT_paths) > idx:
                OCT_idx = np.nonzero(OCT_num == idx)[0]
                for OCT_i in OCT_idx:
                    try:
                        self.OCT[idx].append(cv2.cvtColor(cv2.imread(OCT_paths[OCT_i]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                    except:
                        self.OCT[idx].append(np.zeros([3, 945, 945]))
            else:
                self.OCT[idx].append(np.zeros([3, 945, 945]))
            
            if len(OCT_paths) > idx:
                OCT_idx2 = np.nonzero(OCT_num2 == idx)[0]
                for OCT_i in OCT_idx2:
                    try:

                        self.OCT[idx].append(cv2.cvtColor(cv2.imread(OCT_paths2[OCT_i]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                    except:
                        self.OCT[idx].append(np.zeros([3, 945, 945]))
            else:
                self.OCT[idx].append(np.zeros([3, 945, 945]))

        self.labels = np.array([label[idx]  for idx in split_idx])
        if use_mask:
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_mask = use_mask

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT = self.OCT[index]
        label = self.labels[index]

        if self.use_mask:
            mask = self.mask[index]
            if random.random() > 0.5:
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, OCT, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)



##### Eye dataset #####
class Eye_Loader_OCT_VIT_EDIT(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths_temp = sorted(glob.glob(os.path.join(root_dir,"OCT","*.tif")))
        label = np.loadtxt(os.path.join(root_dir, "label.txt"), delimiter=",")

        OCT_paths = [[] for _ in image_paths]
        for path in OCT_paths_temp:
            n = int(path.split("/")[-1][:4])
            OCT_paths[n].append(path)

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        age_mean = label[:,1].mean()
        age_std = np.std(label[:,1])

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
            {"train":[0, 1, 2, 3, 4],"val":[0, 1, 2, 3, 4],"test":[0, 1, 2, 3, 4]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        OCT_paths = [OCT_paths[idx] for idx in split_idx]
        self.OCT = [[cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in OCT_path] for OCT_path in OCT_paths]

        # bainary => Decimal
        self.labels = np.array([2*label[idx,-2] + label[idx,-1] for idx in split_idx])
        #self.labels = np.array([label[idx,-2] for idx in split_idx])
        
        self.age = np.array([(label[idx,1] - age_mean)/age_std  for idx in split_idx])
        self.sex = np.array([label[idx,2]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT_imgs = self.OCT[index]
        if self.dataset_type == "train":
            OCT = random.choice(OCT_imgs)
        else:
            OCT = OCT_imgs[0]
        label = self.labels[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, OCT, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT_VIT_EDIT2(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths_temp = sorted(glob.glob(os.path.join(root_dir,"OCT","*.tif")))
        label = np.loadtxt(os.path.join(root_dir, "label.txt"), delimiter=",")

        OCT_paths = [[] for _ in image_paths]
        for path in OCT_paths_temp:
            n = int(path.split("/")[-1][:4])
            OCT_paths[n].append(path)

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        age_mean = label[:,1].mean()
        age_std = np.std(label[:,1])

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
            {"train":[0, 1, 2, 3, 4],"val":[0, 1, 2, 3, 4],"test":[0, 1, 2, 3, 4]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        OCT_paths = [OCT_paths[idx] for idx in split_idx]
        self.OCT = [[cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in OCT_path] for OCT_path in OCT_paths]

        # bainary => Decimal
        self.labels = np.array([2*label[idx,-2] + label[idx,-1] for idx in split_idx])
        #self.labels = np.array([label[idx,-2] for idx in split_idx])
        
        self.age = np.array([(label[idx,1] - age_mean)/age_std  for idx in split_idx])
        self.sex = np.array([label[idx,2]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT_imgs = self.OCT[index]
        OCT = OCT_imgs[0]
        label = self.labels[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        if self.use_age:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image = torch.cat([image, torch.full([1, *image.shape[-2:]], self.sex[index])], dim=0)

        return image, OCT, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT_VIT_CHECK(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_mask=False):
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths_temp = sorted(glob.glob(os.path.join(root_dir,"OCT","*.tif")))
        label = np.loadtxt(os.path.join(root_dir, "label.txt"), delimiter=",")

        OCT_paths = [[] for _ in image_paths]
        for path in OCT_paths_temp:
            n = int(path.split("/")[-1][:4])
            OCT_paths[n].append(path)

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
            {"train":[0, 1, 2, 3, 4],"val":[0, 1, 2, 3, 4],"test":[0, 1, 2, 3, 4]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        OCT_paths = [OCT_paths[idx] for idx in split_idx]
        self.OCT = [[cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in OCT_path] for OCT_path in OCT_paths]

        # bainary => Decimal
        self.labels = np.array([2*(label[idx,-3] // 2) + (label[idx,-2] // 2) for idx in split_idx])
        #self.labels = np.array([label[idx,-2] for idx in split_idx])
        
        self.age = np.array([label[idx,0]  for idx in split_idx])
        self.sex = np.array([label[idx,1]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT_imgs = self.OCT[index]
        OCT = OCT_imgs[0]
        label = self.labels[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        age = self.age[index]
        sex = self.sex[index]

        return image, OCT, label, age, sex


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT_VIT_SMOKE(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_smoke=False, use_AMD=False, use_mask=None):
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths_temp = sorted(glob.glob(os.path.join(root_dir,"OCT","*.tif")))
        label = np.loadtxt(os.path.join(root_dir, "label.txt"), delimiter=",")

        OCT_paths = [[] for _ in image_paths]
        for path in OCT_paths_temp:
            n = int(path.split("/")[-1][:4])
            OCT_paths[n].append(path)

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        age_mean = label[:,0].mean()
        age_std = np.std(label[:,0])

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
            {"train":[0, 1, 2, 3, 4],"val":[0, 1, 2, 3, 4],"test":[0, 1, 2, 3, 4]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        OCT_paths = [OCT_paths[idx] for idx in split_idx]
        self.OCT = [[cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in OCT_path] for OCT_path in OCT_paths]

        if use_mask:
            mask_paths = sorted(glob.glob(os.path.join(root_dir, f"{use_mask}_mask", "*.png")))
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        # bainary => Decimal
        self.labels = np.array([2*(label[idx,2] // 2) + (label[idx,3] // 2) for idx in split_idx])
        #self.labels = np.array([label[idx,-2] for idx in split_idx])
        
        #self.age = np.array([(label[idx,0] - age_mean)/age_std  for idx in split_idx]).astype(np.float32)
        self.age = np.array([label[idx,0]  for idx in split_idx]).astype(np.float32)
        self.sex = np.array([label[idx,1]  for idx in split_idx]).astype(np.float32)
        self.smoke = np.array([label[idx,4]  for idx in split_idx]).astype(np.float32)
        self.AMD = np.array([label[idx,4]  for idx in split_idx]).astype(np.float32)

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_smoke = use_smoke
        self.use_mask = use_mask
        self.use_AMD = use_AMD

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT_imgs = self.OCT[index]
        OCT = OCT_imgs[0]
        label = self.labels[index]

        
        if self.use_mask:
            if random.random() > 0.0:
                mask = self.mask[index]
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        info = {}
        if self.use_age:
            info["age"] = self.age[index][None]
        if self.use_sex:
            info["gender"] = self.sex[index][None]
        if self.use_smoke:
            info["smoke"] = self.smoke[index][None]
        if self.use_AMD:
            info["AMD"] = self.smoke[index][None]
        return image, OCT, label, info


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_OCT_VIT_MASK(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, transform=None, use_age=False, use_sex=False, use_smoke=False, use_AMD=False, use_mask=None):
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        OCT_paths_temp = sorted(glob.glob(os.path.join(root_dir,"OCT","*.tif")))
        label = np.loadtxt(os.path.join(root_dir, "label.txt"), delimiter=",")

        OCT_paths = [[] for _ in image_paths]
        for path in OCT_paths_temp:
            n = int(path.split("/")[-1][:4])
            OCT_paths[n].append(path)

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        age_mean = label[:,0].mean()
        age_std = np.std(label[:,0])

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
            {"train":[0, 1, 2, 3, 4],"val":[0, 1, 2, 3, 4],"test":[0, 1, 2, 3, 4]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        OCT_paths = [OCT_paths[idx] for idx in split_idx]
        self.OCT = [[cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in OCT_path] for OCT_path in OCT_paths]

        if use_mask:
            mask_paths = sorted(glob.glob(os.path.join(root_dir, f"HE_mask", "*.png")))
            self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]
            
            mask_paths = sorted(glob.glob(os.path.join(root_dir, f"SE_mask", "*.png")))
            self.mask2 = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        # bainary => Decimal
        self.labels = np.array([2*(label[idx,2] // 2) + (label[idx,3] // 2) for idx in split_idx])
        #self.labels = np.array([label[idx,-2] for idx in split_idx])
        
        #self.age = np.array([(label[idx,0] - age_mean)/age_std  for idx in split_idx]).astype(np.float32)
        self.age = np.array([label[idx,0]  for idx in split_idx]).astype(np.float32)
        self.sex = np.array([label[idx,1]  for idx in split_idx]).astype(np.float32)
        self.smoke = np.array([label[idx,4]  for idx in split_idx]).astype(np.float32)
        self.AMD = np.array([label[idx,4]  for idx in split_idx]).astype(np.float32)

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_smoke = use_smoke
        self.use_mask = use_mask
        self.use_AMD = use_AMD

        self.transform = transform

        if dataset_type=="train":
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                RandomCrop(size=(288, 288)),
                                RandomHorizontalFlip(p=0.5),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288, 288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.OTC_transform = Compose(augment_list)


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        OCT_imgs = self.OCT[index]
        OCT = OCT_imgs[0]
        label = self.labels[index]

        
        if self.use_mask:
            if random.random() > 0.0:
                mask = self.mask[index]
                mask2 = self.mask2[index]
                mask = np.stack([mask, mask2], axis=0)

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        OCT = self.OTC_transform(OCT)
        
        info = {}
        if self.use_age:
            info["age"] = self.age[index][None]
        if self.use_sex:
            info["gender"] = self.sex[index][None]
        if self.use_smoke:
            info["smoke"] = self.smoke[index][None]
        if self.use_AMD:
            info["AMD"] = self.smoke[index][None]
        return image, OCT, mask, label, info


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)




##### Eye dataset #####
class Eye_Loader_simCLR(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train',split=1, use_age=False, use_sex=False, use_mask=False):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images","*")))
        mask_paths = sorted(glob.glob(os.path.join(root_dir,"mask","*.png")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)
        
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = label[:,0,0]

        std = 76.5339018952063
        mean = 73.1766666666667
        input_info = [info.query(f"AI_number == {num}").iloc[:,[2,3]].values for num in image_num]
        input_info = np.array(input_info).squeeze(1)
        normed_age = (input_info[:, 0] - mean) / std
        sex = input_info[:,1]

        split_idx = np.arange(len(image_paths))
        split_idx = np.array_split(split_idx, 5)

        split_pattern = [
            {"train":[0, 1, 2],"val":[3],"test":[4]},
            {"train":[4, 0, 1],"val":[2],"test":[3]},
            {"train":[3, 4, 0],"val":[1],"test":[2]},
            {"train":[2, 3, 4],"val":[0],"test":[1]},
            {"train":[1, 2, 3],"val":[4],"test":[0]},
        ]
        split_idx = np.concatenate([split_idx[idx] for idx in split_pattern[split][dataset_type]])

        self.images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        self.labels = np.array([label[idx]  for idx in split_idx])
        self.mask = [cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE) != 0 for idx in split_idx]

        self.age = np.array([normed_age[idx]  for idx in split_idx])
        self.sex = np.array([sex[idx]  for idx in split_idx])

        self.dataset_type = dataset_type
        self.use_age = use_age
        self.use_sex = use_sex
        self.use_mask = use_mask
        
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transform = transforms.Compose([ndarrayToTensor(),
                                            transforms.CenterCrop(size=(800,800)),
                                            transforms.RandomCrop(size=(256, 256)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * 256))])


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.labels[index]
        mask = self.mask[index]


        if self.use_mask:
            if random.random() > 0.5:
                image[:, mask] = 0

        # もしself.transformが有効なら
        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        
        if self.use_age:
            image1 = torch.cat([image1, torch.full([1, *image1.shape[-2:]], self.age[index])], dim=0)
        if self.use_sex:
            image1 = torch.cat([image1, torch.full([1, *image1.shape[-2:]], self.sex[index])], dim=0)

        return image1, image2, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)



##### Eye dataset #####
class Eye_Loader_traintest(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,f"{dataset_type}_images","*.TIF")))
        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        # bainary => Decimal
        label = 2*label[:,0,0] + label[:,0,1]
        
        self.images = [cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in image_paths]
        self.labels = label
        
        self.dataset_type = dataset_type

        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.labels[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Eye dataset #####
class Eye_Loader_2classification2_traintest(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        info = pd.read_csv(os.path.join(root_dir, "info.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,f"{dataset_type}_images","*.TIF")))
        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)
        label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
        label = np.array(label)
        
        self.images = [cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).transpose(2,0,1) for path in image_paths]
        self.labels = label[:, 0]
        self.dataset_type = dataset_type

        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.images[index]
        label = self.labels[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)
        
        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


############## main ##############
if __name__ == '__main__':
    train_dataset = Eye_Loader_OCT_VIT_SMOKE(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='train', split=5)
    image = train_dataset.images
    OCT = [octs[0] for octs in train_dataset.OCT]

    for i in range(len(image)):
        image[i] = cv2.resize(image[i].transpose(1,2,0), (256, 256)).transpose(2,0,1)
        OCT[i] = cv2.resize(OCT[i].transpose(1,2,0), (256, 256)).transpose(2,0,1)

    image = np.stack(image)
    OCT = np.stack(OCT)

    age = train_dataset.age
    sex = train_dataset.sex
    label = train_dataset.labels

    ARMS2 = label//2
    CFH = label%2
    flag_50_down = age < 50
    flag_50 = (age >= 50) & (age < 60)
    flag_60 = (age >= 60) & (age < 70)
    flag_70 = (age >= 70) & (age < 80)
    flag_80 = age >= 80
    
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    img_emb = PCA(n_components=2).fit_transform(image.reshape(image.shape[0],-1))
    oct_emb = PCA(n_components=2).fit_transform(OCT.reshape(OCT.shape[0],-1))
    # img_emb = TSNE(n_components=2, random_state=0).fit_transform(image.reshape(image.shape[0],-1))
    # oct_emb = TSNE(n_components=2, random_state=0).fit_transform(OCT.reshape(OCT.shape[0],-1))

    color = np.zeros([img_emb.shape[0],3])
    color[flag_50_down] = np.array([1.,0,0])
    color[flag_50] = np.array([0.,0.,0])
    color[flag_60] = np.array([0,1.,0])
    color[flag_70] = np.array([0,0,1.])
    color[flag_80] = np.array([1.,0,1.])


    plt.clf()
    plt.scatter(img_emb[:,0], img_emb[:,1],s=10, c=color)
    plt.savefig(f"img_age.png")

    plt.clf()
    plt.scatter(oct_emb[:,0], oct_emb[:,1],s=10, c=color)
    plt.savefig(f"oct_age.png")