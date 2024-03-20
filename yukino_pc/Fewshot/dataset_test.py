#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import (Covid19_Loader,
    Drosophila_Loader,
    Cityscapse_Loader,
    Drive_Loader,
    ACDC_Loader,
    Synapse_Multi_Organ_Loader,
    BRATS2013_Loader,
    rTg4510_Loader)
from utils.evaluation import IoU, Dice
import utils.utils as ut
from utils.loss import IoULoss, get_class_rate_weight, get_class_balancing_weight, Contrastive_Loss_moepy, Contrastive_Loss_moepy7
from models.Unet import UNet_class_con
from models.Deeplabv3 import Deeplabv3


def build_transform(dataset_name):
    ### data augmentation + preprocceing ###
    if dataset_name == "Covid19":
        train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    
    if dataset_name == "Drosophila":
        train_transform = None
        val_transform = None
    
    if dataset_name == "CityScapes":
        train_transform = ut.Compose([ut.RandomCrop(size=(512, 1024)),  # ランダムクロップ(size=512x1024)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    
    if dataset_name == "DRIVE":
        train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.DRIVE_Pad(),
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    if dataset_name == "ACDC":
        train_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.RandomCrop_tensor(size=(224, 224)),  # ランダムクロップ(size=224x224)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    if dataset_name == "Synapse_Multi_Organ":
        train_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.RandomCrop_tensor(size=(224, 224)),  # ランダムクロップ(size=224x224)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    if dataset_name == "BRATS2013":
        train_transform = ut.Compose([ut.Numpy2Tensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.RandomCrop_tensor(size=(224, 224)),  # ランダムクロップ(size=224x224)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ])

        val_transform = ut.Compose([ut.Numpy2Tensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])
    
    if dataset_name == "Mouse_brain":
        train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5)
                                        ])

        val_transform = ut.Compose([])

    return train_transform, val_transform

############## dataloader関数##############
def dataload(dataset_name, split):
    ### data augmentation + preprocceing ###
    train_transform, val_transform = build_transform(dataset_name)

    if dataset_name == "Covid19":
        train_dataset = Covid19_Loader(root_dir="/mnt/kamiya/dataset/covid19", dataset_type='train', transform=train_transform)
        val_dataset = Covid19_Loader(root_dir="/mnt/kamiya/dataset/covid19", dataset_type='val', transform=val_transform)
    
    if dataset_name == "ACDC":
        train_dataset = ACDC_Loader(root_dir="/mnt/kamiya/dataset/ACDC", mode="train", split=split, transform=train_transform)
        val_dataset = ACDC_Loader(root_dir="/mnt/kamiya/dataset/ACDC", mode="val", split=split, transform=val_transform)
        test_dataset = ACDC_Loader(root_dir="/mnt/kamiya/dataset/ACDC", mode="test", split=split, transform=val_transform)
    
    if dataset_name == "Synapse_Multi_Organ":
        train_dataset = Synapse_Multi_Organ_Loader(root_dir="/mnt/kamiya/dataset/Synapse_Multi_Organ", mode="train", split=split, transform=train_transform)
        val_dataset = Synapse_Multi_Organ_Loader(root_dir="/mnt/kamiya/dataset/Synapse_Multi_Organ", mode="val", split=split, transform=val_transform)
        test_dataset = Synapse_Multi_Organ_Loader(root_dir="/mnt/kamiya/dataset/Synapse_Multi_Organ", mode="test", split=split, transform=val_transform)
    
    if dataset_name == "BRATS2013":
        train_dataset = BRATS2013_Loader(root_dir="/mnt/kamiya/dataset/BRATS2013", mode="train", split=split, transform=train_transform)
        val_dataset = BRATS2013_Loader(root_dir="/mnt/kamiya/dataset/BRATS2013", mode="val", split=split, transform=val_transform)
    
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

def main():
    # データ読み込み+初期設定
    dataload("Synapse_Multi_Organ", 3)


############## main ##############
if __name__ == '__main__':
    main()