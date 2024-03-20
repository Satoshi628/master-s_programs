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
from torchvision.transforms import (Compose,
    Normalize,
    CenterCrop,
    Pad)
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.dataset import Eye_Loader
from utils.visualization import IoU_image
from utils.evaluation import IoU
import utils.utils as ut
from models.Unet import UNet
from models.Deeplabv3 import Deeplabv3

############## dataloader関数##############


############## dataloader関数##############
def dataload():
    transform = Compose([
                                CenterCrop(size=(928, 928)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    dataset = Eye_Loader(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=4,
                                                shuffle=False,
                                                drop_last=False,
                                                #num_workers=2,
                                                pin_memory=True)
    
    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return dataloader


############## test関数 ##############
def test(model, test_loader,visual, device):
    # モデル→推論モード
    model.eval()
    transform = Pad([9,9,8,8])

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, paths) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            # 入力画像をモデルに入力
            output = model(inputs)
            inputs = transform(inputs)
            output = transform(output)

            #可視化
            #visual.save_fig(inputs, paths, mode="input")
            visual.save_fig(output, paths, mode="output")


@hydra.main(config_path="config", config_name='generator.yaml')
def main(cfg):
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')


    #データセットの入力チャンネルとクラス数,セグメンテーション色を選択
    color = [[0, 0, 0],
            [255, 255, 255]]

    if not os.path.exists("result"):
        os.mkdir("result")


    # モデル設定
    model = UNet(in_channels=3, n_classes=2, channel=32).cuda(device)
    #model = Deeplabv3(n_class).cuda(device)

    #マルチGPU
    model_path = "result/model.pth"
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    model.load_state_dict(torch.load(model_path))


    #可視化クラス定義
    segmentation_visualer = IoU_image(color)

    # データ読み込み+初期設定
    test_loader = dataload()

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    test(model, test_loader, segmentation_visualer, device)



############## main ##############
if __name__ == '__main__':
    main()
