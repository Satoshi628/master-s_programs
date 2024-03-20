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
import torch.nn as nn
import optuna

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Covid19_Loader, Drosophila_Loader
from utils.evaluation import IoU
import utils.utils as ut
from utils.loss import FocalLoss, IoULoss, Edge_IoULoss
from models.Unet import UNet

optuna.logging.disable_default_handler()

############## dataloader関数##############


def dataload():
    ### data augmentation + preprocceing ###
    train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                  # ランダムに左右反転(p=probability)
                                  ut.RandomHorizontalFlip(p=0.5),
                                  ut.RandomVerticalFlip(p=0.5),
                                  ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                  ut.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                      0.229, 0.224, 0.225]),  # データ標準化
                                  ])

    val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                ut.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                             0.229, 0.224, 0.225]),  # データ標準化
                                ])

    if dataset == "Covid19":
        train_dataset = Covid19_Loader(
            root_dir="/mnt/kamiya/dataset/covid19", dataset_type='train', transform=train_transform)
        val_dataset = Covid19_Loader(
            root_dir="/mnt/kamiya/dataset/covid19", dataset_type='val', transform=val_transform)
        test_dataset = Covid19_Loader(
            root_dir="/mnt/kamiya/dataset/covid19", dataset_type='test', transform=val_transform)

    if dataset == "Drosophila":
        train_dataset = Drosophila_Loader(
            rootdir="/mnt/kamiya/dataset/drosophila/data", val_area=val_area, split='train', iteration_number=12 * 8)
        val_dataset = Drosophila_Loader(
            rootdir="/mnt/kamiya/dataset/drosophila/data", val_area=val_area, split='val')
        test_dataset = Drosophila_Loader(
            rootdir="/mnt/kamiya/dataset/drosophila/data", val_area=val_area, split='test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             drop_last=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              drop_last=False)

    # これらをやると
    # num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader, test_loader


############## train関数 ##############
def train(model, train_loader, criterion, optimizer, device):
    # モデル→学習モード
    model.train()

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()

    return


############## validation関数 ##############
def val(model, val_loader, IoU_calculator, device):
    # モデル→推論モード
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output = model(inputs)

            ###精度の計算###
            IoU_calculator.update(output, targets, mode="softmax")

        IoU, mean_IoU = IoU_calculator()
    return mean_IoU


############## test関数 ##############
def test(model, test_loader, IoU_calculator, device):
    # モデル→推論モード
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for inputs, targets in test_loader:
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output = model(inputs)

            ###精度の計算###
            IoU_calculator.update(output, targets, mode="softmax")

        IoU, mean_IoU = IoU_calculator()
    return mean_IoU


def objective(trial):
    # 変更パラメータ定義
    edge_range_param = trial.suggest_int('edge_rage', 1, 10)
    lamda_param = trial.suggest_uniform('lamda', 0.0, 2.0)

    # モデル設定
    model = UNet(in_channels=in_channel, n_classes=n_class,
                 channel=32).cuda(device)

    # マルチGPU
    model = torch.nn.DataParallel(model)

    #criterion = IoULoss(n_class)
    criterion = Edge_IoULoss(n_class,
                             edge_range=edge_range_param,
                             lamda=lamda_param)

    print("開始:5分休憩")
    sleep(5 * 60)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=100,
                                              cycle_mult=2,
                                              max_lr=0.001,
                                              min_lr=0.0,
                                              warmup_steps=0,
                                              gamma=0.5)

    best_IoU = 0.0
    best_test_IoU = 0.0
    # args.num_epochs = max epoch
    for epoch in range(700):

        # train
        train(model, train_loader, criterion, optimizer, device)
        # validation
        val_mIoU = val(model, val_loader, IoU_calculator, device)

        if val_mIoU > best_IoU:
            best_IoU = val_mIoU
            best_test_IoU = test(model, test_loader, IoU_calculator, device)

        scheduler.step()

    return -best_test_IoU


############## main ##############
if __name__ == '__main__':

    # dataset判定
    #dataset = ["Covid19", "Drosophila"]
    dataset = "Covid19"

    ##### GPU設定 #####
    device = torch.device('cuda:0')

    # データセットの入力チャンネルとクラス数を選択
    if dataset == "Covid19":
        in_channel = 3
        n_class = 4
    if dataset == "Drosophila":
        in_channel = 1
        n_class = 5
        val_area = 1

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # 精度計算クラス定義
    IoU_calculator = IoU(n_class)

    # データ読み込み+初期設定
    train_loader, val_loader, test_loader = dataload()

    # 高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    TRIAL_SIZE = 40
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    with open(f"optuna_{dataset}.txt", mode='w') as f:
        for key, value in study.best_params.items():
            f.write(f"{key}\t{value}\n")

        f.write(f"best IoU:{-study.best_value}")
