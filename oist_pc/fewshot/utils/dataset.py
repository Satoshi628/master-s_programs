#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random

#----- 専用ライブラリ -----#
from tqdm import tqdm
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import cv2
#----- 自作モジュール -----#
from .utils import size_Pad

##### covid19 dataset #####
class Covid19_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, dataset_type='train', transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = "{}/Image/{}".format(root_dir, dataset_type)
        label_path = "{}/Label/{}".format(root_dir, dataset_type)

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*.png"))
        label_list = sorted(glob.glob(label_path + "/*.png"))

        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB") for image_name in image_list]
        self.label_list = [Image.open(label_list).convert("L") for label_list in label_list]

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

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def get_class_count(self):
        label = np.array([np.array(label) for label in self.label_list])
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count
    
    def few_choise(self, use_ratio=1.00):
        label_np = np.array([np.array(label, dtype=np.long) for label in self.label_list])

        image_idx = np.arange(label_np.shape[0])

        #全クラス必ず1ピクセルは含むと仮定
        num, count = np.unique(label_np, return_counts=True)
        image_class_num = np.array([np.unique(label).shape[0] for label in label_np])
        image_class_min = np.array([np.unique(label, return_counts=True)[1].min() for label in label_np])
        
        class_rank_idx = [image_idx[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        class_rank = [image_class_min[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        
        #sort 降順
        class_rank_sort_idx = [np.argsort(rank)[::-1] for rank in class_rank]
        class_rank_idx_sorted = [class_rank_idx[i][class_rank_sort_idx[i]] for i in range(len(class_rank_sort_idx))]
        
        image_rank = np.concatenate(class_rank_idx_sorted)
        image_rank = image_rank[: int(use_ratio * len(self.label_list))].tolist()
        self.image_list = [self.image_list[idx] for idx in image_rank]
        self.label_list = [self.label_list[idx] for idx in image_rank]

##### Drosophila dataset #####
class Drosophila_Loader():
    def __init__(self, rootdir="", val_area="2", split='train', iteration_number=None):
        self.split = split
        self.training = True if split == 'train' else False

        filelist_train = []
        filelist_val = []
        filelist_test = []
        test_area = val_area + 1 if val_area != 5 else 1

        for i in range(1, 6):
            dataset = [os.path.join(f'{rootdir}/5-fold/Area_{i}/{data}')
                        for data in os.listdir(f'{rootdir}/5-fold/Area_{i}')]
            if i == val_area:
                filelist_test = filelist_test + dataset
            elif i == test_area:
                filelist_val = filelist_val + dataset
            else:
                filelist_train = filelist_train + dataset
        

        if split == 'train':
            self.filelist = filelist_train
        elif split == 'val':
            self.filelist = filelist_val
        elif split == 'test':
            self.filelist = filelist_test

        if self.training:
            self.number_of_run = 1
            self.iterations = iteration_number
        else:
            self.number_of_run = 16
            self.iterations = None

        print(f'val_area : {val_area} test_area : {test_area} ', end='')
        print(f"{split} files : {len(self.filelist)}")

    def __getitem__(self, index):
        # print(index)
        if self.training:
            index = random.randint(0, len(self.filelist) - 1)
            dataset = self.filelist[index]

        else:
            dataset = self.filelist[index // self.number_of_run]

        # load files
        filename_data = os.path.join(dataset)
        inputs = np.load(filename_data)
        #4(12)枚の大きな画像から256*256の画像を生成
        #4(12)*(1024/256)*(1024/256)=64(192)
        #実質64(192)枚学習
        # print(len(inputs))   #1024
        # print(inputs.shape)  # (1024, 1024, 2)
        # print(inputs.dtype)  # float32

        # split features labels
        if self.training:
            x = random.randint(0, inputs.shape[0] - 256)
            y = random.randint(0, inputs.shape[0] - 256)
        else:
            x = index % self.number_of_run//4 * 256
            y = index % self.number_of_run % 4 * 256

        features = inputs[x: x + 256, y: y + 256, 0:1].transpose(2, 0, 1).astype(np.float32)
        features /= 255.0
        
        labels = inputs[x: x + 256, y: y + 256, -1].astype(int)

        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(labels).long()

        return fts, lbs

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations

    def get_class_count(self):
        label = np.array([np.load(data_path)[:, :, -1] for data_path in self.filelist])
        
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]

        return label_count


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
        #this code is faster than "Image.open(image_name).convert("RGB")"
        self.ToPIL = transforms.ToPILImage()
        self.image_list = [self.ToPIL(cv2.imread(image_name)).convert("RGB") for image_name in tqdm(image_list)]

        print("Loading Cityscapse label data")
        self.label_list = [self.ToPIL(self.convert_label(cv2.imread(label_list, -1)))
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

    def get_class_count(self):
        label = np.array([np.array(label) for label in self.label_list])
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count

##### Drive dataset #####
class Drive_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, mode="train", split=1, transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = os.path.join(root_dir, f"{mode}_{split}", "images")
        label_path = os.path.join(root_dir, f"{mode}_{split}", "1st_manual")
        # 眼底画像のみ映っている所のmaskデータは使用しない
        #mask_path = os.path.join(root_dir, split, "mask")

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*"))
        label_list = sorted(glob.glob(label_path + "/*"))

        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB") for image_name in image_list]
        self.label_list = [Image.open(label_list).convert("L") for label_list in label_list]

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

        # class0 = 0 class1 = 255
        label = label / 255
        
        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def get_class_count(self):
        label = np.array([np.array(label) for label in self.label_list])
        label_num, label_count = np.unique(label / 255, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count

    def few_choise(self, use_ratio=1.00):
        image_idx = np.arange(self.label_np.shape[0])

        #全クラス必ず1ピクセルは含むと仮定
        num, count = np.unique(self.label_np, return_counts=True)
        image_class_num = np.array([np.unique(label).shape[0] for label in self.label_np])
        image_class_min = np.array([np.unique(label, return_counts=True)[1].min() for label in self.label_np])
        
        class_rank_idx = [image_idx[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        class_rank = [image_class_min[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        
        #sort 降順
        class_rank_sort_idx = [np.argsort(rank)[::-1] for rank in class_rank]
        class_rank_idx_sorted = [class_rank_idx[i][class_rank_sort_idx[i]] for i in range(len(class_rank_sort_idx))]
        
        image_rank = np.concatenate(class_rank_idx_sorted)
        image_rank = image_rank[: int(use_ratio * len(self.label_np))].tolist()
        
        self.image_np = np.array([self.image_np[idx] for idx in image_rank])
        self.label_np = np.array([self.label_np[idx] for idx in image_rank])

##### Drive dataset test #####
class Drive_Loader_Test(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = os.path.join(root_dir, "test", "images")
        # 眼底画像のみ映っている所のmaskデータは使用しない
        #mask_path = os.path.join(root_dir, split, "mask")

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*"))

        #画像読み込み
        self.image_list = [Image.open(image_name).convert("RGB") for image_name in image_list]

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index]
        label = self.image_list[index].convert("L")

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)

        # class0 = 0 class1 = 255
        label = label / 255
        
        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

##### ACDC dataset #####
class ACDC_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, mode="train", split=1, transform=None):
        # データ取得
        self.image_np = np.load(os.path.join(root_dir, "datamodel", f"{mode}_data_{split}.npy"))
        self.label_np = np.load(os.path.join(root_dir, "datamodel", f"{mode}_label_{split}.npy"))
        
        # self.transform = transform
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_np[index]
        label = self.label_np[index]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_np)

    def get_class_count(self):
        label_num, label_count = np.unique(self.label_np, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count

    def few_choise(self, use_ratio=1.00):
        image_idx = np.arange(self.label_np.shape[0])

        #全クラス必ず1ピクセルは含むと仮定
        num, count = np.unique(self.label_np, return_counts=True)
        image_class_num = np.array([np.unique(label).shape[0] for label in self.label_np])
        image_class_min = np.array([np.unique(label, return_counts=True)[1].min() for label in self.label_np])
        
        class_rank_idx = [image_idx[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        class_rank = [image_class_min[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        
        #sort 降順
        class_rank_sort_idx = [np.argsort(rank)[::-1] for rank in class_rank]
        class_rank_idx_sorted = [class_rank_idx[i][class_rank_sort_idx[i]] for i in range(len(class_rank_sort_idx))]
        
        image_rank = np.concatenate(class_rank_idx_sorted)
        image_rank = image_rank[: int(use_ratio * len(self.label_np))].tolist()
        
        self.image_np = np.array([self.image_np[idx] for idx in image_rank])
        self.label_np = np.array([self.label_np[idx] for idx in image_rank])



# Same as ACDC_Loader
##### Synapse_Multi_Organ dataset #####
class Synapse_Multi_Organ_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, mode="train", split=1, transform=None):

        # データ取得
        self.image_np = np.load(os.path.join(root_dir, "datamodel", f"{mode}_data_{split}.npy"))
        self.label_np = np.load(os.path.join(root_dir, "datamodel", f"{mode}_label_{split}.npy"))

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_np[index]
        label = self.label_np[index]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_np)

    def get_class_count(self):
        label_num, label_count = np.unique(self.label_np, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count

    def few_choise(self, use_ratio=1.00):
        image_idx = np.arange(self.label_np.shape[0])

        #全クラス必ず1ピクセルは含むと仮定
        num, count = np.unique(self.label_np, return_counts=True)
        image_class_num = np.array([np.unique(label).shape[0] for label in self.label_np])
        image_class_min = np.array([np.unique(label, return_counts=True)[1].min() for label in self.label_np])
        
        class_rank_idx = [image_idx[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        class_rank = [image_class_min[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        
        #sort 降順
        class_rank_sort_idx = [np.argsort(rank)[::-1] for rank in class_rank]
        class_rank_idx_sorted = [class_rank_idx[i][class_rank_sort_idx[i]] for i in range(len(class_rank_sort_idx))]
        
        image_rank = np.concatenate(class_rank_idx_sorted)
        image_rank = image_rank[: int(use_ratio * len(self.label_np))].tolist()
        
        self.image_np = np.array([self.image_np[idx] for idx in image_rank])
        self.label_np = np.array([self.label_np[idx] for idx in image_rank])


##### BRATS2013 dataset #####
class BRATS2013_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, mode='train', split=0, transform=None):

        #入力画像とラベルの上位ディレクトリ取得
        image_path = f"{root_dir}/data/{mode}_image_{split}.npz"
        label_path = f"{root_dir}/data/{mode}_label_{split}.npz"

        #画像読み込み
        self.image_np = np.load(image_path)["arr_0"].astype(np.float32)
        self.label_np = np.load(label_path)["arr_0"]

        self.mode = mode

        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_np[index]
        label = self.label_np[index]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_np)

    def get_class_count(self):
        label_num, label_count = np.unique(self.label_np, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count

    def few_choise(self, use_ratio=1.00):
        image_idx = np.arange(self.label_np.shape[0])

        #全クラス必ず1ピクセルは含むと仮定
        num, count = np.unique(self.label_np, return_counts=True)
        image_class_num = np.array([np.unique(label).shape[0] for label in self.label_np])
        image_class_min = np.array([np.unique(label, return_counts=True)[1].min() for label in self.label_np])
        
        class_rank_idx = [image_idx[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        class_rank = [image_class_min[image_class_num == i] for i in range(num.shape[0], 0, -1)]
        
        #sort 降順
        class_rank_sort_idx = [np.argsort(rank)[::-1] for rank in class_rank]
        class_rank_idx_sorted = [class_rank_idx[i][class_rank_sort_idx[i]] for i in range(len(class_rank_sort_idx))]
        
        image_rank = np.concatenate(class_rank_idx_sorted)
        image_rank = image_rank[: int(use_ratio * len(self.label_np))].tolist()
        
        self.image_np = np.array([self.image_np[idx] for idx in image_rank])
        self.label_np = np.array([self.label_np[idx] for idx in image_rank])


##### rTg4510 dataset #####
class rTg4510_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir, mode='train', split=0, transform=None):
        padding_func = size_Pad([288,288])
        split_list = [[0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]]
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = "{}/rTg4510/rTg4510_ex_vivo/template".format(root_dir)
        label_path = "{}/rTg4510/rTg4510_ex_vivo/label".format(root_dir)

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = sorted(glob.glob(image_path + "/*.nii.gz"))
        label_list = sorted(glob.glob(label_path + "/*.nii.gz"))
        
        if mode == "train":
            image_list = [image_list[idx] for idx in range(len(image_list)) if not idx in split_list[split]]
            label_list = [label_list[idx] for idx in range(len(label_list)) if not idx in split_list[split]]
        elif mode == "val":
            image_list = [image_list[idx] for idx in range(len(image_list)) if idx in split_list[split][:2]]
            label_list = [label_list[idx] for idx in range(len(label_list)) if idx in split_list[split][:2]]
        elif mode == "test":
            image_list = [image_list[idx] for idx in range(len(image_list)) if idx in split_list[split][2:]]
            label_list = [label_list[idx] for idx in range(len(label_list)) if idx in split_list[split][2:]]
        
        #画像読み込み

        self.image_list = [padding_func(torch.tensor(nib.load(image).get_fdata())) for image in image_list]
        self.label_list = [padding_func(torch.tensor(nib.load(label).get_fdata())) for label in label_list]

        self.image_list = torch.cat(self.image_list, dim=0)[:, None].float()
        self.label_list = torch.cat(self.label_list, dim=0)[:, None]
        
        # self.mode = dataset = 'train' or 'val' or 'test'
        self.mode = mode

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

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def get_class_count(self):
        label_num, label_count = torch.unique(self.label_list, return_counts=True)
        sort_idx = torch.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count