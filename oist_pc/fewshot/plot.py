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
from utils.dataset import Covid19_Loader, Drosophila_Loader, Cityscapse_Loader, Drive_Loader, ACDC_Loader, Synapse_Multi_Organ_Loader, BRATS2013_Loader
from utils.visualization import Edge_tSNE, Edge_Seg_image
from utils.evaluation import IoU, Dice, HD
import utils.utils as ut
from models.Unet import UNet, UNet_class_con, UNet_contrastive
from models.Deeplabv3 import Deeplabv3

############## dataloader関数##############


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


def dataload(dataset_name, cfg):
    ### data augmentation + preprocceing ###
    test_transform = build_transform(dataset_name)

    if dataset_name == "Covid19":
        test_dataset =Covid19_Loader(root_dir=cfg.root_dir, dataset_type='test', transform=test_transform)

    if dataset_name == "Drosophila":
        test_dataset = Drosophila_Loader(rootdir=cfg.root_dir, val_area=cfg.val_area, split='test')

    if dataset_name == "CityScapes":
        test_dataset = Cityscapse_Loader(root_dir=cfg.root_dir, dataset_type='val', transform=test_transform)

    if dataset_name == "ACDC":
        test_dataset = ACDC_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=test_transform)

    if dataset_name == "Synapse_Multi_Organ":
        test_dataset = Synapse_Multi_Organ_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=test_transform)

    if dataset_name == "BRATS2013":
        test_dataset = BRATS2013_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            drop_last=False)
    
    return test_loader


############## test関数 ##############
def test(model, test_loader, visual,seg, Eval_calc_list, device):
    # モデル→推論モード
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            print("ok")
            if len(torch.unique(targets)) != 4:
                continue
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            feature = model.get_feature(inputs)
            #seg.save_fig(targets, mode="label")
            
            visual(feature, targets)
            visual.one_class(feature, targets)
            print("ok")
            break



@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.test_conf.gpu}")
    print(f"multi GPU   :{cfg.test_conf.multi_gpu}")
    print(f"dataset     :{cfg.dataset.name}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.test_conf.gpu) if torch.cuda.is_available() else 'cpu')


    #データセットの設定
    dataset_name = cfg.dataset.name
    dataset_cfg = eval(f"cfg.dataset.{dataset_name}")
    model_name = dataset_cfg.model
    in_channel = dataset_cfg.input_channel
    n_class = dataset_cfg.classes

    #データセットの入力チャンネルとクラス数,セグメンテーション色を選択
    # color list = [B,G,R]
    if dataset_name == "Covid19":
        color = [[0, 0, 0],
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0]]
    
    if dataset_name == "Drosophila":
        color = [[255, 255, 255],
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0],
                [0, 0, 0]]

    if dataset_name == "ACDC":
        color = [[0, 0, 0],
                 [0, 0, 255],
                 [0, 255, 0],
                 [255, 0, 0]]

    if dataset_name == "Synapse_Multi_Organ":
        color = [[128, 128, 128],
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

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test.txt"
    with open(PATH, mode='w') as f:
        f.write("")


    # モデル設定
    if model_name == "U-Net":
        model = UNet(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "U-Net-moepy":
        model = UNet_class_con(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "U-Net-CLLE":
        model = UNet_contrastive(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "DeepLabv3":
        model = Deeplabv3(n_class).cuda(device)


    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #可視化クラス定義
    tSNE_visual = Edge_tSNE(color, n_class)
    seg = Edge_Seg_image(color, n_class)

    # データ読み込み+初期設定
    test_loader = dataload(dataset_name, dataset_cfg)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    test(model, test_loader, tSNE_visual, seg, cfg, device)

############## main ##############
if __name__ == '__main__':
    main()
