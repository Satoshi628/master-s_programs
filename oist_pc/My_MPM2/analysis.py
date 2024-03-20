#coding: utf-8
#----- 標準ライブラリ -----#
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.dataset import OISTLoader_raw_data
from utils.MPM_to_track import MPM_to_Cell, Cell_Tracker, division_process, Cell_Tracker_non_division
from utils.evaluation import Precition_Recall, Object_Tracking, Object_Detection
from U_net import UNet

############## dataloader関数##############
def dataload(cfg):
    test_loader = OISTLoader_raw_data()
    
    return test_loader


############## validation関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0
    
    predict_MPM = torch.tensor([])
    predict_MPM = []

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs) in tqdm(enumerate(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)

            # 入力画像をモデルに入力
            output = model(inputs)
            predict_MPM.append(output.to("cpu"))
            
        predict_MPM = torch.cat(predict_MPM, dim=0)
        cell = MPM_to_Cell(predict_MPM)
        MPM_track = Cell_Tracker(predict_MPM, cell)
        #MPM_track = Cell_Tracker_non_division(predict_MPM, cell)
        #MPM_track = division_process(MPM_track) #増えすぎだめ


        save_track = MPM_track[:, [0, 1, 2, 3]].to('cpu').numpy().copy()
        np.save('result/track', save_track)


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')


    # モデル設定
    model = UNet(n_channels=2, n_classes=3, bilinear=True).cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # データ読み込み+初期設定
    test_loader = dataload(cfg)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    test(model, test_loader, device)

############## main ##############
if __name__ == '__main__':
    main()
    
