#coding: utf-8
#----- 標準ライブラリ -----#
import matplotlib.pyplot as plt
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.dataset import Covid19_Loader, ACDC_Loader, Synapse_Multi_Organ_Loader, BRATS2013_Loader
import utils.utils as ut
from utils.visualization import Seg_image



def build_transform(dataset_name):
    ### data augmentation + preprocceing ###
    if dataset_name == "Covid19":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    
    if dataset_name == "Drosophila":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])
    
    if dataset_name == "CityScapes":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    if dataset_name == "ACDC":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    if dataset_name == "Synapse_Multi_Organ":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])
    
    if dataset_name == "BRATS2013":
        test_transform = ut.Compose([ut.Numpy2Tensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    return test_transform



def dataload(dataset_name):
    ### data augmentation + preprocceing ###
    test_transform = build_transform(dataset_name)

    if dataset_name == "Covid19":
        test_dataset = Covid19_Loader(root_dir="/mnt/kamiya/dataset/covid19", dataset_type='test', transform=test_transform)
    
    if dataset_name == "ACDC":
        test_dataset = ACDC_Loader(root_dir="/mnt/kamiya/dataset/ACDC", mode="test", split=1, transform=test_transform)

    if dataset_name == "Synapse_Multi_Organ":
        test_dataset = Synapse_Multi_Organ_Loader(root_dir="/mnt/kamiya/dataset/Synapse_Multi_Organ", mode="test", split=1, transform=test_transform)

    if dataset_name == "BRATS2013":
        test_dataset = BRATS2013_Loader(root_dir="/mnt/kamiya/dataset/BRATS2013", mode="test", split=3, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            drop_last=False)
    
    return test_loader


############## test関数 ##############
def test(test_loader, visual):
    tar = 275
    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
        # 教師ラベルをlongモードに変換
        targets = targets.long()
        if batch_idx <= tar:
            continue
        visual.save_fig(inputs[:, :1], mode="input")
        visual.save_fig(targets, mode="label")
        if batch_idx >= tar + 20:
            break



def main():
    
    dataset_name = "BRATS2013"

    n_class = 5

    #データセットの入力チャンネルとクラス数,セグメンテーション色を選択
    # color list = [B,G,R]
    if dataset_name == "Covid19":
        color = [[0, 0, 0],
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0]]
    
    if dataset_name == "ACDC":
        color = [[0, 0, 0],
                 [0, 0, 255],
                 [0, 255, 0],
                 [255, 0, 0]]

    if dataset_name == "Synapse_Multi_Organ":
        color = [[0, 0, 0],
                 [189, 220, 254],
                 [156, 206, 144],
                 [0, 255, 0],
                 [0, 255, 255],
                 [128, 0, 128],
                 [253, 254, 189],
                 [255, 0, 0],
                 [255, 73, 143]]

    if dataset_name == "BRATS2013":
        color = [[0, 0, 0],
                 [0, 0, 255],
                 [0, 255, 0],
                 [255, 0, 0],
                 [255, 0, 0]]



    #可視化クラス定義
    segmentation_visualer = Seg_image(color)

    # データ読み込み+初期設定
    test_loader = dataload(dataset_name)

    test(test_loader, segmentation_visualer)


main()